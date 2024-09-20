# Part 3: Introduction to pandas for Energy Engineers

# 1. Importing pandas
import pandas as pd

# 2. Creating DataFrames
# Creating a DataFrame from a dictionary
data = {
    'Temperature (°C)': [20, 25, 30, 35, 40],
    'Pressure (bar)': [1.0, 1.2, 1.4, 1.6, 1.8],
    'Flow Rate (m³/h)': [100, 150, 200, 250, 300]
}

df = pd.DataFrame(data)
print("Initial DataFrame:")
print(df)


# 4. Exploring the Data
print("\nDataFrame Information:")
print(df.info())

print("\nDescriptive Statistics:")
print(df.describe())

# 5. Selection and Indexing
# Selecting a column
temperatures = df['Temperature (°C)']
print("\nTemperature Column:")
print(temperatures)

# Selecting multiple columns
temp_press = df[['Temperature (°C)', 'Pressure (bar)']]
print("\nTemperatures and Pressures:")
print(temp_press)

# Selecting rows using loc and iloc
first_row = df.loc[0]
print("\nFirst Row of the DataFrame:")
print(first_row)

# Selecting a specific value
specific_value = df.loc[2, 'Pressure (bar)']
print("\nSpecific value at row 2, column 'Pressure (bar)':", specific_value)

# 6. Filtering Data
# Filtering rows where temperature is greater than 30°C
df_high_temp = df[df['Temperature (°C)'] > 30]
print("\nData with temperature greater than 30°C:")
print(df_high_temp)

# 7. Adding New Columns
# Calculating thermal energy Q = m * c * ΔT (simplified for example)
mass = 1000  # kg
specific_heat = 4.18  # kJ/(kg·K)
initial_temperature = 15  # °C
df['Thermal Energy (kJ)'] = mass * specific_heat * (df['Temperature (°C)'] - initial_temperature)
print("\nDataFrame with Calculated Thermal Energy:")
print(df)

# 8. Grouping and Aggregation
# Calculating the average flow rate for each pressure
average_flow_rate = df.groupby('Pressure (bar)')['Flow Rate (m³/h)'].mean()
print("\nAverage Flow Rate by Pressure:")
print(average_flow_rate)

# 9. Merging DataFrames
# Creating another DataFrame
data2 = {
    'Temperature (°C)': [25, 30, 35],
    'Efficiency (%)': [80, 82, 85]
}

df_efficiency = pd.DataFrame(data2)

# Merging DataFrames on the 'Temperature (°C)' column
df_merged = pd.merge(df, df_efficiency, on='Temperature (°C)', how='left')
print("\nMerged DataFrame with Efficiency:")
print(df_merged)

# 10. Data Visualization
import matplotlib.pyplot as plt

# Plotting Temperature vs. Thermal Energy
plt.plot(df['Temperature (°C)'], df['Thermal Energy (kJ)'], marker='o')
plt.title('Temperature vs. Thermal Energy')
plt.xlabel('Temperature (°C)')
plt.ylabel('Thermal Energy (kJ)')
plt.grid(True)
plt.show()

# Bar chart of Efficiency
df_merged.dropna(inplace=True)  # Remove rows with NaN values
plt.bar(df_merged['Temperature (°C)'], df_merged['Efficiency (%)'])
plt.title('Efficiency as a Function of Temperature')
plt.xlabel('Temperature (°C)')
plt.ylabel('Efficiency (%)')
plt.show()
