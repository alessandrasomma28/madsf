#!/bin/bash

# Dates for the runs
BASE_DATES=('2021-11-10' '2021-11-12' '2021-11-14' '2021-10-06' '2021-10-08' '2021-10-10' '2021-06-23' '2021-06-25' '2021-06-27')
SCENARIO='normal'
ACTIVE_GUI='no'
VERBOSE='no'
MODE='social_groups'    # sumo, multi_agent, social_groups
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENV_PATH="${ROOT_DIR}/.env"

time_to_minutes() {
  IFS=':' read -r h m <<< "$1"
  echo $((10#$h * 60 + 10#$m))
}

minutes_to_time() {
  local total=$1
  local h=$((total / 60 % 24))
  local m=$((total % 60))
  printf "%02d:%02d" $h $m
}

# Durations in minutes
DAY_DURATIONS=(60 180 360 720)     # 08:00–09:00, 08:00-11:00, 08:00–14:00, 08:00–20:00
NIGHT_DURATIONS=(60 180 360 720)   # 20:00–21:00, 20:00-23:00, 20:00–02:00, 20:00–08:00
DAY_START_MIN=$(time_to_minutes 08:00)
NIGHT_START_MIN=$(time_to_minutes 20:00)

for BASE_DATE in "${BASE_DATES[@]}"; do

  # === DAY RUNS ===
  for DURATION in "${DAY_DURATIONS[@]}"; do
    START_TIME=$(minutes_to_time $DAY_START_MIN)
    END_MIN=$((DAY_START_MIN + DURATION))
    END_TIME=$(minutes_to_time $END_MIN)
    FOLDER_NAME="$(date -jf "%Y-%m-%d %H:%M" "${BASE_DATE} ${START_TIME}" "+%y%m%d%H")_$(date -jf "%Y-%m-%d %H:%M" "${BASE_DATE} ${END_TIME}" "+%y%m%d%H")"
    FOLDER_PATH="${ROOT_DIR}/sumoenv/scenarios/${SCENARIO}/${MODE}/${FOLDER_NAME}"
    if [ -d "$FOLDER_PATH" ]; then
      echo "⏭️  Skipping existing DAY run: $FOLDER_NAME"
      continue
    fi
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
    echo "▶️  DAY [$MODE] [$SCENARIO] $START_TIME-$END_TIME on $BASE_DATE (${DURATION} min)"
    set -a
    source "$ENV_PATH"
    set +a
    python "${ROOT_DIR}/main.py"
    "${ROOT_DIR}/clean.sh"
  done

  # === NIGHT RUNS ===
  for DURATION in "${NIGHT_DURATIONS[@]}"; do
    START_TIME=$(minutes_to_time $NIGHT_START_MIN)
    END_MIN=$((NIGHT_START_MIN + DURATION))
    if [ $END_MIN -lt 1440 ]; then
      END_DATE=$BASE_DATE
      END_TIME=$(minutes_to_time $END_MIN)
    else
      END_DATE=$(date -j -v+1d -f "%Y-%m-%d" "$BASE_DATE" "+%Y-%m-%d")
      END_TIME=$(minutes_to_time $((END_MIN % 1440)))
    fi
    FOLDER_NAME="$(date -jf "%Y-%m-%d %H:%M" "${BASE_DATE} ${START_TIME}" "+%y%m%d%H")_$(date -jf "%Y-%m-%d %H:%M" "${END_DATE} ${END_TIME}" "+%y%m%d%H")"
    FOLDER_PATH="${ROOT_DIR}/sumoenv/scenarios/${SCENARIO}/${MODE}/${FOLDER_NAME}"
    if [ -d "$FOLDER_PATH" ]; then
      echo "⏭️  Skipping existing NIGHT run: $FOLDER_NAME"
      continue
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
    echo "▶️  NIGHT [$MODE] [$SCENARIO] $START_TIME-$END_TIME on $BASE_DATE (${DURATION} min)"
    set -a
    source "$ENV_PATH"
    set +a
    python "${ROOT_DIR}/main.py"
    "${ROOT_DIR}/clean.sh"
  done

done