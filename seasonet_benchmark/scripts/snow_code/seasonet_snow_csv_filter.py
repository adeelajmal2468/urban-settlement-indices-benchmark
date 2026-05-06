# import pandas as pd

# # Load the CSV file
# meta_file = r"D:\SeasoNet\meta.csv"  # Change this to your actual file path
# df = pd.read_csv(meta_file)

# # Display column names to check the correct one to filter
# print("Columns in CSV:", df.columns)

# # Assuming the relevant column is named 'season' (update if needed)
# season_column = "Season"  # Change this based on actual column name

# # Filter entries that start with "winter"
# snow_entries = df[df[season_column].str.startswith("Snow", na=False)]

# # Display results
# print("Entries starting with 'snow':")
# print(snow_entries)

# # Save the filtered entries to a new CSV file
# output_file = r"D:\SeasoNet\snow_entries.csv"
# snow_entries.to_csv(output_file, index=False)

# print(f"✅ Filtered data saved to {output_file}")

import pandas as pd

# Load the CSV file
meta_file = r"D:\SeasoNet\meta.csv"  # Update this with your actual file path
df = pd.read_csv(meta_file)

# Display column names to identify the correct "Classes" column
print("Columns in CSV:", df.columns)

# Define relevant column names
season_column = "Season"  # Modify if needed
classes_column = "Classes"  # Modify if needed

# Print unique values in season column for debugging
print("Unique season values:", df[season_column].unique())

# Define the valid class values (1-6, 10, 11)
valid_classes = {1, 2, 3, 4, 5, 6, 10, 11}

# Step 1: Find the index range where "Winter" season starts and ends
snow_start_index = df[df[season_column].str.contains("Snow", case=False, na=False)].index.min()
snow_end_index = df[df[season_column].str.contains("Snow", case=False, na=False)].index.max()

# Select only rows between the first and last occurrence of Winter
df_snow = df.loc[snow_start_index:snow_end_index].copy()  # Make an explicit copy to avoid warnings

# Step 2: Ensure the "Classes" column is treated as strings for flexible matching
df_snow.loc[:, classes_column] = df_snow[classes_column].astype(str)  # Fixes SettingWithCopyWarning

# Function to check if at least one valid class exists in the entry
def contains_valid_class(class_entry):
    class_labels = set(map(int, class_entry.split(",")))  # Convert comma-separated values to a set of integers
    return any(cls in class_labels for cls in valid_classes)

# Step 3: Filter rows where the "Classes" column contains at least one valid class
filtered_df = df_snow[df_snow[classes_column].apply(contains_valid_class)]

# Save the filtered data to a new CSV file
output_file = r"D:\SeasoNet\filtered_snow_meta.csv"
filtered_df.to_csv(output_file, index=False)

print(f"✅ Filtered data saved to {output_file}")
