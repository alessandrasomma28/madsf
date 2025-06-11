#!/bin/bash

# Fixed parameters and paths
BASE_DATE='2021-01-14'
SCENARIO='normal'
ACTIVE_GUI='no'
VERBOSE='no'
MODE='social_groups'
DURATIONS=(60 360 720) # Minutes
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENV_PATH="${ROOT_DIR}/.env"

# Convert HH:MM to total minutes
time_to_minutes() {
  IFS=':' read -r h m <<< "$1"
  echo $((10#$h * 60 + 10#$m))
}

# Convert total minutes to HH:MM
minutes_to_time() {
  local total=$1
  local h=$((total / 60 % 24))
  local m=$((total % 60))
  printf "%02d:%02d" $h $m
}

# Add 1 day to a date
next_day() {
  date -j -f "%Y-%m-%d" "$1" -v+1d "+%Y-%m-%d"
}

for DURATION in "${DURATIONS[@]}"; do

  # === DAY SHIFT (8AM - 8PM) ===
  for START_MIN in $(seq $(time_to_minutes 08:00) 60 $(time_to_minutes 19:00)); do
    END_MIN=$((START_MIN + DURATION))
    if [ $END_MIN -gt $(time_to_minutes 20:00) ]; then continue; fi
    START_TIME=$(minutes_to_time $START_MIN)
    END_TIME=$(minutes_to_time $END_MIN)
    cat <<EOF > "$ENV_PATH"
START_DATE=${BASE_DATE}
END_DATE=${BASE_DATE}
START_TIME=${START_TIME}
END_TIME=${END_TIME}
SCENARIO=${SCENARIO}
MODE=${MODE}
ACTIVE_GUI=${ACTIVE_GUI}
VERBOSE=${VERBOSE}
EOF
    echo "▶️  DAY [$MODE] $START_TIME-$END_TIME (${DURATION} min)"
    export $(grep -v '^#' "$ENV_PATH" | xargs)
    python "${ROOT_DIR}/main.py"
  done

  # === NIGHT SHIFT (8PM - 8AM) ===
  LIMIT_MIN=$(( $(time_to_minutes 08:00) + 1440 ))
  for START_MIN in $(seq $(time_to_minutes 20:00) 60 $(time_to_minutes 23:00)); do
    END_MIN=$((START_MIN + DURATION))
    if [ $END_MIN -gt $LIMIT_MIN ]; then continue; fi
    START_TIME=$(minutes_to_time $START_MIN)
    if [ $END_MIN -lt 1440 ]; then
      END_TIME=$(minutes_to_time $END_MIN)
      END_DATE=$BASE_DATE
    else
      END_TIME=$(minutes_to_time $((END_MIN % 1440)))
      END_DATE=$(next_day "$BASE_DATE")
    fi
    cat <<EOF > "$ENV_PATH"
START_DATE=${BASE_DATE}
END_DATE=${END_DATE}
START_TIME=${START_TIME}
END_TIME=${END_TIME}
SCENARIO=${SCENARIO}
MODE=${MODE}
ACTIVE_GUI=${ACTIVE_GUI}
VERBOSE=${VERBOSE}
EOF
    echo "▶️  NIGHT [$MODE] $START_TIME-$END_TIME (${DURATION} min)"
    export $(grep -v '^#' "$ENV_PATH" | xargs)
    python "${ROOT_DIR}/main.py"
  done
done
