import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def write_array(array, file_name):
    with open(file_name, 'w') as f:
        for row in array:
            f.write(' '.join(map(str, row)) + '\n')

file1 = "merged_dataset.csv"
data = pd.read_csv(file1)
col_names = data.columns

def process_pop_column(df):
    df.iloc[:, 21] = np.where(df.iloc[:, 21] < 4, 0, df.iloc[:, 21])
    max_val = df.iloc[:, 21].max()
    if max_val > 4:
        df.iloc[:, 21] = np.where(df.iloc[:, 21] > 4, (df.iloc[:, 21] - 4) / (max_val - 4), 0)
    return df

def process_ret_colum(df):
    max_val = df.iloc[:, 22].max()
    if max_val > 1:
        max_val = 1
    df.iloc[:, 22] = np.where(df.iloc[:, 22] < 1, (max_val - df.iloc[:, 22]) / max_val, 0)
    return df
    
def process_subs_colum(df):
    max_val = df.iloc[:, 24].max()
    df.iloc[:, 24] = (max_val - df.iloc[:, 22]) / max_val
    return df

def process_road_colum(df):
    df.iloc[:, 25] = np.where(df.iloc[:, 25] < 1, 1, df.iloc[:, 25])
    max_val = df.iloc[:, 25].max()
    df.iloc[:, 25] = np.where(df.iloc[:, 25] > 1, (max_val - df.iloc[:, 25]) / (max_val - 1), df.iloc[:, 25])
    return df

def round_and_remove_duplicates(df, decimals=2):
    # df_rounded = df.copy()
    df.iloc[:, 1:3] = df.iloc[:, 1:3].round(decimals)
    df_unique = df.drop_duplicates(subset=df.columns[1:3])
    return df_unique

def negate_columns(df, negative_cols):
    df.iloc[:, negative_cols] = -df.iloc[:, negative_cols]
    return df

def minmax_scale_columns(df):
    scaler = MinMaxScaler()
    df.iloc[:, 5:] = scaler.fit_transform(df.iloc[:, 5:])
    return df

process_pop_column(data)
process_ret_colum(data)
process_subs_colum(data)
process_road_colum(data)
data = round_and_remove_duplicates(data)
data = negate_columns(data, [5, 9, 11, 12, 13])
minmax_scale_columns(data)

data.to_csv('preproc_data.csv', index=False)
dat_arr = data.iloc[:,5:].values
write_array(dat_arr,"brownfield_preprocessed.txt")