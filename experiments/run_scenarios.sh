#!/bin/bash

# Dates for the runs
BASE_DATES=('2021-11-10')
# Scenarios to run
SCENARIOS=('underground_alarm' 'wildcat_strike' 'flash_mob' 'long_rides' 'greedy_drivers' 'budget_passengers' 'boycott_tncs' 'underground_alarm_greedy' 'wildcat_strike_greedy' 'flash_mob_greedy' 'long_rides_greedy' 'greedy_drivers_greedy' 'budget_passengers_greedy' 'boycott_tncs_greedy' 'underground_alarm_long_rides' 'wildcat_strike_long_rides' 'wildcat_strike_budget_passengers' 'wildcat_strike_boycott_tncs' 'flash_mob_long_rides' 'flash_mob_boycott_tncs')
ACTIVE_GUI='no'
VERBOSE='no'
MODE='social_groups'    # Available modes: sumo, multi_agent, social_groups
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
DURATIONS=(480)     # 480 = 08:00–16:00 d
START_MIN=$(time_to_minutes 08:00)

for SCENARIO in "${SCENARIOS[@]}"; do
  for BASE_DATE in "${BASE_DATES[@]}"; do

    for DURATION in "${DURATIONS[@]}"; do
      START_TIME=$(minutes_to_time $START_MIN)
      END_MIN=$((START_MIN + DURATION))
      END_TIME=$(minutes_to_time $END_MIN)
      FOLDER_NAME="$(date -jf "%Y-%m-%d %H:%M" "${BASE_DATE} ${START_TIME}" "+%y%m%d%H")_$(date -jf "%Y-%m-%d %H:%M" "${BASE_DATE} ${END_TIME}" "+%y%m%d%H")"
      FOLDER_PATH="${ROOT_DIR}/sumoenv/scenarios/${SCENARIO}/${MODE}/${FOLDER_NAME}"
      if [ -d "$FOLDER_PATH" ]; then
        echo "⏭️  Skipping existing run: $FOLDER_NAME"
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
      echo "▶️  [$MODE] [$SCENARIO] $START_TIME-$END_TIME on $BASE_DATE (${DURATION} min)"
      set -a
      source "$ENV_PATH"
      set +a
      python "${ROOT_DIR}/main.py"
      "${ROOT_DIR}/clean.sh"
    done
  done
done