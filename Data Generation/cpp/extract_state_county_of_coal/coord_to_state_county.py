import requests
import pandas as pd
import time

# Replace with your actual Geoapify API key
API_KEY = "d2756a36aebf47c5b637661ea358684a"
BASE_URL = "https://api.geoapify.com/v1/geocode/reverse"

csv_path = "CPP_Data.csv"
coordinates = pd.read_csv(csv_path).iloc[:,:3].values

# Data storage
data = []
counter = 1
# Process each coordinate
for identifier, lon, lat in coordinates:
    params = {"lat": lat, "lon": lon, "apiKey": API_KEY}
    response = requests.get(BASE_URL, params=params)

    if response.status_code == 200:
        result = response.json()
        if result.get("features"):
            props = result["features"][0]["properties"]
            state = props.get("state", "N/A")
            county = props.get("county", "N/A")
            data.append([identifier, lon, lat, state, county])
        else:
            data.append([identifier, lon, lat, "N/A", "N/A"])
    else:
        data.append([identifier, lon, lat, "N/A", "N/A"])

    print("Reading",counter,"/ 265 has been completed.")
    counter+=1
    time.sleep(0.5)  # Delay to avoid rate limits

# Convert to DataFrame
df = pd.DataFrame(data, columns=["Identifier", "Longitude", "Latitude", "State", "County"])

# Save to CSV
df.to_csv("coordinates_with_state_county.csv", index=False)

print("Data saved to coordinates_with_state_county.csv")
