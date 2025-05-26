import pandas as pd

# Load dataset
df = pd.read_csv("/Users/beyzaeken/Desktop/sfdigitalmirror/data/sf_traffic/sfmta_dataset/sf_traffic_data_21010100_21123023.csv")

# Convert timestamp column to datetime
df['vehicle_position_date_time'] = pd.to_datetime(df['vehicle_position_date_time'])

# Extract date (no time)
df['day'] = df['vehicle_position_date_time'].dt.date

# Count entries per day
counts = df['day'].value_counts().sort_index()

# Convert index to datetime (from datetime.date)
counts.index = pd.to_datetime(counts.index)

# Create full date range from actual data range
full_range = pd.date_range(start=df['day'].min(), end=df['day'].max(), freq='D')

# Reindex to include all days; fill missing with 0
complete_counts = counts.reindex(full_range, fill_value=0)

# Filter for days with < 100 entries
filtered_days = complete_counts[complete_counts < 100]

# Create a DataFrame from the filtered result
filtered_df = filtered_days.reset_index()
filtered_df.columns = ['date', 'entry_count']

# Optional: sort by date (already sorted by default)
filtered_df = filtered_df.sort_values('date')

# Save to CSV
filtered_df.to_csv("days_with_less_than_100_traffic_cars.csv", index=False)

print("Exported to 'days_with_less_than_100_traffic_cars.csv'")