import pandas as pd

# Load the CSV files
df1 = pd.read_csv("loc_id_vs_obs_rate_top20.csv")  # Top 20 entries
df2 = pd.read_csv("BF_CPP_DATA.csv")  # Full dataset

# Clean the registryid column in df1 by removing brackets and converting to integers
df1['registryid'] = df1['registryid'].astype(str).str.strip("\'[]").astype(str)

# Merge df1 with df2 to get x and y based on registryid
df_merged = df1.merge(df2[['registryid', 'x', 'y']], on='registryid', how='left')

# Save the result to a new CSV file
df_merged.to_csv("top_20_ready_to_map.csv", index=False)

print("File 'top_20_ready_to_map.csv' has been created successfully.")
