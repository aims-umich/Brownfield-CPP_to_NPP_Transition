import pandas as pd

# Load the datasets
cpps_fips = pd.read_csv("CPPs_with_FIPS.csv")  # First file
cpp_data = pd.read_csv("CPP_Data.csv")  # Second file

# Extract the required columns
cpps_fips_selected = cpps_fips.iloc[:, :5]  # First 5 columns
cpp_data_selected = cpp_data.iloc[:, 3:]  # Columns from index 3 onwards

# Concatenate the selected columns
cpp_data_complete = pd.concat([cpps_fips_selected, cpp_data_selected], axis=1)

# Save the final dataset
cpp_data_complete.to_csv("CPP_Data_Complete.csv", index=False)

print("Processing complete. File saved as CPP_Data_Complete.csv")
