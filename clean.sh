#!/bin/bash

PROJECT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Target directories
DAILY_TRAFFIC_DIR="$PROJECT_PATH/data/sf_traffic/daily_traffic"
MAP_MATCHED_DIR="$PROJECT_PATH/data/sf_traffic/map_matched"
SCENARIOS_DIR="$PROJECT_PATH/sumoenv/scenarios"

# Function to clean all contents of a directory (only XML for scenarios)
clean_directory() {
    local dir="$1"
    if [ -d "$dir" ]; then
        if [ "$dir" == "$SCENARIOS_DIR" ]; then
            echo "🧹 Cleaning XML files in: $dir"
            find "$dir" -type f -name "*.xml" -exec rm -f {} +
        else
            echo "🧹 Cleaning directory: $dir"
            rm -rf "$dir"/* "$dir"/.[!.]* "$dir"/..?* 2>/dev/null
        fi
    else
        echo "❌ Directory does not exist: $dir"
    fi
}


# Execute cleaning
clean_directory "$DAILY_TRAFFIC_DIR"
clean_directory "$MAP_MATCHED_DIR"
clean_directory "$SCENARIOS_DIR"

# Clean .env file
ENV_FILE="$PROJECT_PATH/.env"
if [ -f "$ENV_FILE" ]; then
    echo "🧹 Removing .env file: $ENV_FILE"
    rm -f "$ENV_FILE"
fi

echo "✅ Directories and .env file cleaned"