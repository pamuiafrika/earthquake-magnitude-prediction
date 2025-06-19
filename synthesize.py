import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Read the input CSV file
df = pd.read_csv('Earthquake_Events_1970_2025_083913.csv')

# Function to convert time string to seconds for sorting
def time_to_seconds(time_str):
    t = datetime.strptime(time_str, '%H:%M:%S')
    return t.hour * 3600 + t.minute * 60 + t.second

# Function to generate synthetic data
def generate_synthetic_data(df, n_synthetic):
    synthetic_data = {
        'Year': [],
        'Month': [],
        'Day': [],
        'Time': [],
        'Lat': [],
        'Lon': [],
        'Depth': [],
        'Mag': []
    }
    
    # Analyze distributions
    min_year, max_year = df['Year'].min(), df['Year'].max()
    min_lat, max_lat = df['Lat'].min(), df['Lat'].max()
    min_lon, max_lon = df['Lon'].min(), df['Lon'].max()
    min_depth, max_depth = df['Depth'].min(), df['Depth'].max()
    min_mag, max_mag = df['Mag'].min(), df['Mag'].max()
    
    for _ in range(n_synthetic):
        # Generate Year
        year = np.random.randint(min_year, max_year + 1)
        
        # Generate Month
        month = np.random.randint(1, 13)
        
        # Generate Day (ensure valid day for the month)
        if month in [4, 6, 9, 11]:
            day = np.random.randint(1, 31)
        elif month == 2:
            if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
                day = np.random.randint(1, 30)  # Leap year
            else:
                day = np.random.randint(1, 29)
        else:
            day = np.random.randint(1, 32)
        
        # Generate Time
        seconds = np.random.randint(0, 24 * 3600)
        time = (datetime(1970, 1, 1) + timedelta(seconds=seconds)).strftime('%H:%M:%S')
        
        # Generate Lat and Lon with perturbation
        base_lat = np.random.choice(df['Lat'])
        base_lon = np.random.choice(df['Lon'])
        lat = np.clip(base_lat + np.random.normal(0, 0.5), -90, 90)
        lon = np.clip(base_lon + np.random.normal(0, 0.5), -180, 180)
        
        # Generate Depth and Mag with perturbation
        base_depth = np.random.choice(df['Depth'])
        base_mag = np.random.choice(df['Mag'])
        depth = np.clip(base_depth + np.random.normal(0, 0.5), 0, max_depth)
        mag = np.clip(base_mag + np.random.normal(0, 0.1), 0, 10)
        
        # Append to synthetic data
        synthetic_data['Year'].append(year)
        synthetic_data['Month'].append(month)
        synthetic_data['Day'].append(day)
        synthetic_data['Time'].append(time)
        synthetic_data['Lat'].append(round(lat, 4))
        synthetic_data['Lon'].append(round(lon, 4))
        synthetic_data['Depth'].append(round(depth, 1))
        synthetic_data['Mag'].append(round(mag, 1))
    
    return pd.DataFrame(synthetic_data)

# Generate synthetic data to double the dataset
n_original = len(df)
synthetic_df = generate_synthetic_data(df, n_original)

# Combine original and synthetic data
combined_df = pd.concat([df, synthetic_df], ignore_index=True)

# Convert Time to seconds for sorting
combined_df['Time_seconds'] = combined_df['Time'].apply(time_to_seconds)

# Sort by all columns in ascending order
combined_df = combined_df.sort_values(by=['Year', 'Month', 'Day', 'Time_seconds', 'Lat', 'Lon', 'Depth', 'Mag'])

# Drop the temporary Time_seconds column
combined_df = combined_df.drop(columns=['Time_seconds'])

# Save the sorted data to a new CSV file
combined_df.to_csv('sorted_earthquake_data.csv', index=False)

print(f"Generated {n_original} synthetic entries and combined with original {n_original} entries.")
print("Data sorted by Year, Month, Day, Time, Lat, Lon, Depth, Mag and saved to 'sorted_earthquake_data.csv'.")