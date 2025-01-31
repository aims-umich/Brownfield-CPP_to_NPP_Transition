import pandas as pd

# Load sum_norm_obj_cont.csv and rename its columns
df_obj_cont = pd.read_csv("sum_norm_obj_cont.csv") #, header=None)  There is header in the file
df_obj_cont.columns = [f"obj_cont_{i+1}" for i in range(22)]  # Rename columns to obj_cont_1, ..., obj_cont_22

# Load BF_CPP_DATA.csv
df_bf_cpp = pd.read_csv("BF_CPP_DATA.csv")

# Add obj_cont columns to BF_CPP_DATA.csv, ensuring row order matches
df_bf_cpp = pd.concat([df_bf_cpp, df_obj_cont], axis=1)

# Load loc_id_vs_obs_rate_all.csv
df_all = pd.read_csv("loc_id_vs_obs_rate_all.csv")

# Clean the registryid column in df_all
df_all['registryid'] = df_all['registryid'].astype(str).str.strip("[]'")  # Remove brackets and quotes

# Merge df_all with df_bf_cpp using registryid
df_merged = df_bf_cpp.merge(df_all, on="registryid", how="left")

# Save the final dataset
df_merged.to_csv("BF_CPP_XY_Data.csv", index=False)

print("File 'BF_CPP_XY_Data.csv' has been created successfully.")