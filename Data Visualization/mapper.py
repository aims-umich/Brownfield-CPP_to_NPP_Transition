import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.lines import Line2D  # Import Line2D for custom legend

# Reversed Data
data = {
    'Registry ID': [
        'C2914', 'C2712', 'C2367', 'C8042', '110038759572', '110015334440', 'C6249', '110038751534',
        '110043241417', 'C136', 'C50611', 'C56068', 'C50951', 'C2936', '110038712453', '110015334333',
        'C6073', 'C7030', 'C6705', 'C6761'
    ],
    'Longitude': [
        -81.46815, -79.0731, -70.7842, -80.0603, -80.32, -122.1, -79.35724, -85.04,
        -79.19, -81.63278, -76.45, -87.8336, -110.3917, -81.254, -78.63, -112.02,
        -88.5553, -96.69503, -87.3328, -105.0212
    ],
    'Latitude': [
        40.520066, 36.4833, 43.0978, 36.2811, 27.42, 37.67, 33.33184, 46.61,
        35.49, 29.733056, 40.6191, 42.8492, 39.5472, 41.7265, 35.76, 40.72,
        30.5322, 31.091925, 37.915, 40.860905
    ],
    'Siting Metric': [
    1.0, 0.7991088893412379, 0.7024962174015452, 0.6650321979625818, 
    0.6168741631187121, 0.6017711093917133, 0.6005502220042526, 0.597281829696846, 
    0.5927438105487253, 0.5868290491214819, 0.5805731315224438, 0.5726682722680971, 
    0.5678012219828754, 0.5673419754185612, 0.5612345494644894, 0.5364642348422098, 
    0.5363179273550641, 0.5352988654161009, 0.5344111271104323, 0.5285220360412378
    ]
}

# Convert to numpy arrays
longitudes = np.array(data['Longitude'])
latitudes = np.array(data['Latitude'])
siting_metrics = np.array(data['Siting Metric'])
registry_ids = np.array(data['Registry ID'])

# Create color map for siting metric
norm = colors.Normalize(vmin=min(siting_metrics), vmax=max(siting_metrics))
cmap = cm.coolwarm

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Create a Basemap instance for the US
m = Basemap(projection='merc', llcrnrlat=24, urcrnrlat=50,
            llcrnrlon=-125, urcrnrlon=-66.5, resolution='i')

# Draw map features with zorder values
m.drawcoastlines(zorder=1)  # Coastlines at the bottom
m.drawcountries(linewidth=2, zorder=2)  # Countries on top of coastlines
m.drawstates(zorder=3)  # States above countries

# Loop through data and plot points with different markers based on Registry ID
sc = None  # Initialize sc for the scatter object
for i in range(len(registry_ids)):
    x, y = m(longitudes[i], latitudes[i])  # Convert longitude and latitude to map projection coordinates
    if registry_ids[i].startswith('C'):
        marker = 's'  # Square marker for Registry IDs starting with 'C'
    else:
        marker = 'o'  # Circle marker for other Registry IDs
    
    # Plot each point with the defined marker
    sc = m.scatter(x, y, c=siting_metrics[i], cmap=cmap, norm=norm, s=80, marker=marker, zorder=4, alpha=0.8)

# Add a colorbar with adjusted size
plt.colorbar(sc, label='Siting Metric', fraction=0.025, pad=0.04)

# Create custom legend with gray filling
legend_elements = [Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, label='CPP'),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Brownfield')]

# Place legend in lower right corner
plt.legend(handles=legend_elements, loc='lower right')

# Set title and adjust layout
plt.title("Locations of Best Performing Sites")
plt.subplots_adjust(right=0.85)  # Adjust the spacing to avoid clipping of the colorbar

# Save and show the plot
plt.savefig('us_map_with_siting_metrics.png', dpi=300, bbox_inches='tight')
plt.show()
