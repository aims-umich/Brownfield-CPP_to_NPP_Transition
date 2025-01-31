import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV into a pandas DataFrame
df = pd.read_csv('preproc_data.csv')
columns_to_delete = [0,1,2,3,4]
df.drop(df.columns[columns_to_delete], axis=1, inplace=True)

df.columns.values[5] = 'traditional_en_market_reg'
df.columns.values[6] = 'state_annual_labor_wage'
df.columns.values[8] = 'hazardous_fac_in_5mi'
df.columns.values[18] = 'nuclear_r&d_within_100mi'
df.columns.values[21] = 'streamflow_50kgpm_20mi'

correlation_matrix = df.corr()
plt.rcParams["font.family"] = "Times New Roman"
plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix, 
    annot=True, 
    cmap='coolwarm', 
    fmt=".2f", 
    linewidths=0.5, 
    annot_kws={"fontname": "Times New Roman", "fontsize": 11}  # Font for in-cell annotations
)

plt.title('Correlation Matrix Heatmap', fontsize=18, fontname='Times New Roman')
#plt.xlabel('Features', fontsize=14, fontname='Times New Roman')
#plt.ylabel('Features', fontsize=14, fontname='Times New Roman')

plt.xticks(fontsize=12, fontname='Times New Roman')
plt.yticks(fontsize=12, fontname='Times New Roman')

plt.show()