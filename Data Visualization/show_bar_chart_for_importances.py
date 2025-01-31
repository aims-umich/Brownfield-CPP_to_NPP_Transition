import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd

# Characteristics (x-axis labels)
characteristics = [
    'Number of Nuclear Restrictions in the State', 'State Electricity Price', 'State Net Electricity Imports',
    'State Nuclear Inclusive Policy', 'Population Sentiment of Nuclear Energy', 'Traditional Regulation In Energy Market',
    '5-Year Average Labor Rate', 'Social Vulnerability Index', 'Number of Intersecting Protected Lands',
    'Number of Hazardous Facilities in 5 Miles', 'No Fault Lines', 'No Landslide Area',
    'Peak Ground Acceleration Lower Than 0.3g', 'Not Having A Flood in Previous 100 Years',
    'No Open Water Or Wetland Intersection', 'Slope Lower Than 12 Percent', 'Population Center Distance',
    'Retiring Facility Distance', 'Nuclear R&D Centers in 100 Miles', 'Electricity Substation Distance',
    'Transportation System Distance', '50 Kgpm Streamflow Inside 20 Miles'
]

bf_cpp_xy_df = pd.read_csv('BF_CPP_XY_Data.csv')
# Data (5 rows, one for each ID, and 22 columns for each characteristic)

# Sort dataframe by 'siting_metric' in descending order
bf_cpp_xy_df_sorted = bf_cpp_xy_df.sort_values(by='siting_metric', ascending=False)

# Select top 5 rows and objective contribution columns
data = bf_cpp_xy_df_sorted.iloc[:5, 27:49].to_numpy()


# Color mapping for identifiers (IDs)
color_mapping = {
    'C2914': '#1f77b4',        # Blue
    'C2712': '#ff7f0e',        # Orange
    'C2367': '#2ca02c',        # Green
    'C8042': '#d62728',        # Red
    '110038759572': '#9467bd'  # Purple
}

# Assign each ID to a color
colors = list(color_mapping.values())

# Bar positions (based on characteristics)
bar_width = 0.15
index = np.arange(len(characteristics))

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 7))

# Plot bars for each ID
for i, color in enumerate(colors):
    ax.barh(index + i * bar_width, data[i], bar_width, label=f"ID {list(color_mapping.keys())[i]}", color=color)

# Customize the chart
ax.set_xlabel('Normalized Summed Objective Contribution ($\overrightarrow{\mathrm{SC}}$)', fontname='Times New Roman', fontsize=14)
ax.set_yticks(index + bar_width * 2)
ax.set_yticklabels(characteristics, fontname='Times New Roman')
ax.set_title('Characteristic Values for Different IDs', fontname='Times New Roman')

# Set x and y ticks font to Times New Roman
ax.tick_params(axis='x', labelsize=12, labelcolor='black', labelrotation=0, width=1, direction='in', length=6)
ax.tick_params(axis='y', labelsize=12, labelcolor='black', labelrotation=0, width=1, direction='in', length=6)
ax.xaxis.set_tick_params(labelsize=12, labelcolor='black', labelrotation=0, width=1)
ax.yaxis.set_tick_params(labelsize=12, labelcolor='black', labelrotation=0, width=1)

plt.legend()
# Invert y-axis to keep the order from top to bottom
ax.invert_yaxis()

# Save the figure with 300 DPI
plt.tight_layout()
plt.savefig("characteristics_bar_chart.png", dpi=300)

# Display the plot
plt.show()
