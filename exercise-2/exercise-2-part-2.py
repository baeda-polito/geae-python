# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.stats import zscore

import matplotlib.dates as mdates

# Load data
# Replace 'pydata.csv' and 'df_info.csv' with your actual data file paths
pydata = pd.read_csv('data/pydata.csv', parse_dates=['date_time'])
df_info = pd.read_csv('data/df_info.csv')

# Define time windows
spl = ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "23:59"]
spl_times = [datetime.strptime(t, "%H:%M").time() for t in spl]

# Assign periods based on time
def assign_period(row):
    time = row['time']
    if time < spl_times[1]:
        return "Period_1"
    elif time < spl_times[2]:
        return "Period_2"
    elif time < spl_times[3]:
        return "Period_3"
    elif time < spl_times[4]:
        return "Period_4"
    elif time < spl_times[5]:
        return "Period_5"
    else:
        return "Period_6"

# Create 'date' and 'time' columns if not already present
pydata['date'] = pydata['date_time'].dt.date
pydata['time'] = pydata['date_time'].dt.time

# Apply the function to assign periods
pydata['period'] = pydata.apply(assign_period, axis=1)

# Standardize 'energy_h' using z-score
pydata['znorm'] = zscore(pydata['energy_h'])

# Aggregate using mean within each date and period
df_PAA = pydata.groupby(['date', 'period']).agg({'znorm': 'mean'}).reset_index()
df_PAA.rename(columns={'znorm': 'znorm_mean'}, inplace=True)

# Define breakpoints
breakpoints = [-0.84, -0.25, 0.25, 0.84]

# Function to assign symbols based on znorm_mean
def assign_symbol(value):
    if value <= breakpoints[0]:
        return 'A'
    elif value <= breakpoints[1]:
        return 'B'
    elif value <= breakpoints[2]:
        return 'C'
    elif value <= breakpoints[3]:
        return 'D'
    else:
        return 'E'

# Apply the function to assign symbols
df_PAA['symbol'] = df_PAA['znorm_mean'].apply(assign_symbol)

# Merge df_PAA back to pydata
pydata = pd.merge(pydata, df_PAA[['date', 'period', 'symbol']], on=['date', 'period'], how='left')

# Prepare data for plotting
pydata['time_str'] = pydata['time'].astype(str)
pivot_table = pydata.pivot_table(index='date', columns='time_str', values='symbol', aggfunc='first')

# Define custom color mapping
symbol_colors = {
    'A': '#5591ff',
    'B': '#40c8ff',
    'C': '#ffd954',
    'D': '#7fdd56',
    'E': '#ed6f6f'
}

# Create a color map
from matplotlib.colors import ListedColormap

cmap = ListedColormap([symbol_colors[key] for key in sorted(symbol_colors.keys())])

# Map symbols to numeric values for plotting
symbol_to_num = {symbol: idx for idx, symbol in enumerate(sorted(symbol_colors.keys()))}
numeric_data = pivot_table.replace(symbol_to_num)

# Plot the heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(numeric_data, cmap=cmap, cbar_kws={'ticks': list(symbol_to_num.values()), 'label': 'SAX Symbol'}, yticklabels=30)
plt.title('Carpet Plot of Energy Consumption (SAX Symbols)')
plt.xlabel('Time')
plt.ylabel('Date')

# Adjust x-axis labels
plt.xticks(rotation=45)
plt.show()

# Pivot df_PAA to have periods as columns
df_PAA_pivot = df_PAA.pivot(index='date', columns='period', values='symbol').reset_index()

# Ensure periods are ordered correctly
periods = ['Period_1', 'Period_2', 'Period_3', 'Period_4', 'Period_5', 'Period_6']
df_PAA_pivot = df_PAA_pivot[['date'] + periods]

# Concatenate symbols to form words
df_PAA_pivot['word'] = df_PAA_pivot[periods].apply(lambda row: ''.join(row.values.astype(str)), axis=1)

# Calculate word frequencies
word_counts = df_PAA_pivot['word'].value_counts().reset_index()
word_counts.columns = ['word', 'count']

# Merge word counts back to df_PAA_pivot
df_PAA_pivot = pd.merge(df_PAA_pivot, word_counts, on='word')

# Sort words by frequency
df_PAA_pivot['word'] = pd.Categorical(df_PAA_pivot['word'], categories=word_counts['word'], ordered=True)

# Plot frequency of words
plt.figure(figsize=(10, 8))
sns.countplot(y='word', data=df_PAA_pivot, order=word_counts['word'])
plt.axhline(0.05 * df_PAA_pivot['word'].value_counts().max(), color='blue', linestyle='--')
plt.title('Frequency of Daily Words')
plt.xlabel('Count')
plt.ylabel('Word')
plt.show()

# Determine threshold for motifs
threshold = 0.1 * df_PAA_pivot['count'].max()

# Tag motifs and discords
df_PAA_pivot['pattern'] = df_PAA_pivot['count'].apply(lambda x: 'discord' if x < threshold else 'motif')

# Merge back to pydata
pydata = pd.merge(pydata, df_PAA_pivot[['date', 'word', 'pattern']], on='date', how='left')

# Convert 'time' to datetime for plotting
pydata['time_dt'] = pd.to_datetime(pydata['time'].astype(str), format='%H:%M:%S')

# Plot daily load profiles
g = sns.FacetGrid(pydata, col='word', col_wrap=4, height=4, sharey=False)
g.map_dataframe(sns.lineplot, x='time_dt', y='energy_h', hue='pattern', estimator=None, units='date', lw=0.7, alpha=0.7)

# Adjust plot aesthetics
for ax in g.axes.flatten():
    ax.set_xlabel('Hour')
    ax.set_ylabel('Energy Consumption')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=8))
    plt.setp(ax.get_xticklabels(), rotation=45)

plt.tight_layout()
plt.show()
