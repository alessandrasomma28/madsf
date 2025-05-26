#!/bin/bash

PROJECT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Target directories
DAILY_TRAFFIC_DIR="$PROJECT_PATH/data/sf_traffic/daily_traffic"
MAP_MATCHED_DIR="$PROJECT_PATH/data/sf_traffic/map_matched"
SFMTA_DATASET_DIR="$PROJECT_PATH/data/sf_traffic/sfmta_dataset"
SCENARIOS_DIR="$PROJECT_PATH/sumoenv/scenarios"

# Function to clean all contents of a directory
clean_directory() {
    local dir="$1"
    if [ -d "$dir" ]; then
        echo "🧹 Cleaning directory: $dir"
        rm -rf "$dir"/* "$dir"/.[!.]* "$dir"/..?* 2>/dev/null
    else
        echo "❌ Directory does not exist: $dir"
    fi
}

# Execute cleaning
clean_directory "$DAILY_TRAFFIC_DIR"
clean_directory "$MAP_MATCHED_DIR"
clean_directory "$SFMTA_DATASET_DIR"
clean_directory "$SCENARIOS_DIR"

echo "✅ Directories cleaned"
