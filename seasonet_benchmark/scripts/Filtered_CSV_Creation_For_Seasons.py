import pandas as pd

# Load the CSV file
meta_file = r"D:\Work\SeasoNet\meta.csv"  # Update this with your actual file path
df = pd.read_csv(meta_file)

# Display column names to identify the correct "Classes" column
print("Columns in CSV:", df.columns)

# Define relevant column names
season_column = "Season"  # Modify if needed
classes_column = "Classes"  # Modify if needed

# Print unique values in season column for debugging
print("Unique season values:", df[season_column].unique())

# Define urban and non-urban class values
urban_classes = {1, 2, 3, 4, 5, 6, 10, 11}
non_urban_classes = {7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33}

# Step 1: Find the index range where "Winter" season starts and ends
winter_start_index = df[df[season_column].str.contains("Winter", case=False, na=False)].index.min()
winter_end_index = df[df[season_column].str.contains("Winter", case=False, na=False)].index.max()

# Select only rows between the first and last occurrence of Winter
df_winter = df.loc[winter_start_index:winter_end_index].copy()  # Make an explicit copy to avoid warnings

# Step 2: Ensure the "Classes" column is treated as strings for flexible matching
df_winter.loc[:, classes_column] = df_winter[classes_column].astype(str)  # Fixes SettingWithCopyWarning

# Function to check if an entry contains both urban and non-urban classes
def contains_both_urban_and_non_urban(class_entry):
    class_labels = set(map(int, class_entry.split(",")))  # Convert comma-separated values to a set of integers
    has_urban = any(cls in class_labels for cls in urban_classes)
    has_non_urban = any(cls in class_labels for cls in non_urban_classes)
    return has_urban and has_non_urban

# Step 3: Filter rows where the "Classes" column contains both urban and non-urban classes
filtered_df = df_winter[df_winter[classes_column].apply(contains_both_urban_and_non_urban)]

# Save the filtered data to a new CSV file
output_file = r"D:\Work\SeasoNet\filtered_winter_meta_with_both_urban_nonurban.csv"
filtered_df.to_csv(output_file, index=False)

print(f"✅ Filtered data saved to {output_file}")