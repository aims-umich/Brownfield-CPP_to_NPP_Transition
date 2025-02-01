import pandas as pd

# Load user data
user_df = pd.read_csv('coordinates_with_state_county.csv')

# Load state FIPS codes
state_fips = pd.read_csv('state_fips.txt', delimiter=r'\s{2,}', engine='python', names=['StateFIPS', 'State'], dtype=str)
state_fips['State'] = state_fips['State'].str.upper()

# Load county FIPS codes
county_fips = pd.read_csv('county_fips.txt', delimiter=r'\s{2,}', engine='python', names=['CountyFIPS', 'County'], dtype=str)
county_fips['StateFIPS'] = county_fips['CountyFIPS'].str[:2]  # Extract state FIPS from county FIPS
county_fips['County'] = county_fips['County'].str.upper()

# Merge user data with state FIPS
user_df['State'] = user_df['State'].str.upper()
user_df = user_df.merge(state_fips, on='State', how='left')

# Merge user data with county FIPS
user_df['County'] = user_df['County'].str.upper()
user_df = user_df.merge(county_fips, on=['StateFIPS', 'County'], how='left')


# Drop 'State' and 'County' columns if they exist
user_df = user_df.drop(columns=['State', 'County'], errors='ignore')

# Reorder columns: Swap 'StateFIPS' and 'CountyFIPS'
user_df = user_df[['Identifier', 'Longitude', 'Latitude', 'CountyFIPS', 'StateFIPS']]

# Rename columns
user_df.columns = ['registryid', 'x', 'y', 'county_fips', 'state_fips']

output_file = "CPPs_with_FIPS.csv"

# Save the updated dataset
user_df.to_csv(output_file, index=False)

print(f"Processing complete. File saved as: {output_file}")