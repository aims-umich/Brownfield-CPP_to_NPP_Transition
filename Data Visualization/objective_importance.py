import matplotlib.pyplot as plt
import pandas as pd

# List of column names
column_names = [
    "Number of Nuclear Restrictions in State",
    "State Electricity Price",
    "State Net Electricity Imports",
    "State Nuclear Inclusive Policy",
    "Traditional Regulation In Energy Market",
    "5-Year Average Labor Rate",
    "Social Vulnerability Index",
    "Number of Intersecting Protected Lands",
    "Number of Hazardous Facilities in 5 Miles",
    "No Fault Lines",
    "No Landslide Area",
    "Peak Ground Acceleration Lower Than 0.3g",
    "Not Having A Flood in Previous 100 Years",
    "No Open Water Or Wetland Intersection",
    "Population Sentiment Towards Nuclear Energy",
    "Having A Slope Lower Than 12 Percent",
    "Population Center Distance",
    "Retiring Facility Distance",
    "Existing Nuclear R&D Center in 100 Miles",
    "Electricity Substation Distance",
    "Transportation System Distance",
    "50 Kgpm Streamflow Inside 20 Miles"
]

# Corresponding importance values
importance_values = [
    0.051468744, 0.03390798, 0.030181858, 0.030181858, 0.029396615,
    0.051468744, 0.03390798, 0.051468744, 0.029396615, 0.051468744,
    0.051468744, 0.051468744, 0.029396615, 0.051468744, 0.146289722,
    0.051468744, 0.029396615, 0.032143575, 0.029396615, 0.053788638,
    0.029396615, 0.051468744
]

# Create a DataFrame for easier plotting
df = pd.DataFrame({
    'Feature': column_names,
    'Importance': importance_values
})

# Sort values for better visualization
df = df.sort_values(by='Importance', ascending=False)

# Plotting
plt.figure(figsize=(10, 6))  # Smaller figure size
plt.barh(df['Feature'], df['Importance'], color='skyblue')
plt.xlabel('Objective Importance')
plt.ylabel('Objectives')
plt.gca().invert_yaxis()  # Invert y-axis to have the highest importance at the top

# Set font size for y-axis labels
plt.yticks(fontsize=8)
plt.xticks(fontsize=10)
plt.tight_layout()
plt.savefig("Objective_Importances.png", dpi=300, bbox_inches='tight')
plt.show()
