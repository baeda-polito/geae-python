# Part 2: Introduction to NumPy for Energy Engineers

# 1. Importing NumPy
import numpy as np

# 2. Creating NumPy Arrays
# Creating a 1D array (vector) of temperatures in Celsius
temperatures_celsius = np.array([20, 25, 30, 35, 40])
print("Temperatures in Celsius:", temperatures_celsius)

# Creating a 2D array (matrix) representing a grid of pressure values
pressures = np.array([
    [1.0, 1.2, 1.4],
    [1.1, 1.3, 1.5],
    [1.2, 1.4, 1.6]
])
print("Pressure grid (bar):\n", pressures)

# 3. Array Operations
# Converting temperatures from Celsius to Kelvin
temperatures_kelvin = temperatures_celsius + 273.15
print("Temperatures in Kelvin:", temperatures_kelvin)

# Calculating the average temperature
average_temperature = np.mean(temperatures_celsius)
print("Average temperature (°C):", average_temperature)

# Performing element-wise multiplication
volumes = np.array([1.0, 1.5, 2.0, 2.5, 3.0])  # in cubic meters
# Calculating the product of pressures and volumes (P*V)
pv_product = pressures[0, 0] * volumes  # Using the first pressure value for simplicity
print("P*V product:", pv_product)

# 4. Multidimensional Arrays
# Creating a 3D array representing energy consumption over time and different units
energy_consumption = np.random.rand(2, 3, 4)  # Random values for demonstration
print("Energy consumption data:\n", energy_consumption)

# 5. Indexing and Slicing
# Accessing elements in an array
first_temperature = temperatures_celsius[0]
print("First temperature (°C):", first_temperature)

# Slicing arrays
subset_temperatures = temperatures_celsius[1:4]
print("Subset of temperatures (°C):", subset_temperatures)

# Accessing elements in a 2D array
pressure_value = pressures[1, 2]
print("Pressure at position (1,2) (bar):", pressure_value)

# 6. Mathematical Functions
# Using NumPy mathematical functions

radii = np.array([0.05, 0.1, 0.15])  # in meters
areas = np.pi * radii ** 2  # Calculating areas of circles
print("Areas of circles (m^2):", areas)

# Calculating exponential decay (e.g., for heat loss over time)
time = np.linspace(0, 10, 5)  # 5 time points from 0 to 10 seconds
initial_temperature = 100  # °C
temperature_decay = initial_temperature * np.exp(-0.3 * time)
print("Temperature over time (°C):", temperature_decay)

# 7. Linear Algebra Operations
# Solving a system of linear equations (e.g., network of resistances)
# Ohm's Law: V = I * R
# Suppose we have the following system:
# R * I = V

# Resistance matrix (R)
R = np.array([[5, 2],
              [2, 3]])

# Voltage vector (V)
V = np.array([12, 10])

# Calculating the current vector (I)
I = np.linalg.solve(R, V)
print("Currents (A):", I)

# 8. Statistical Functions
# Generating a dataset of measured temperatures with random noise
measured_temperatures = temperatures_celsius + np.random.normal(0, 0.5, temperatures_celsius.shape)
print("Measured temperatures (°C):", measured_temperatures)

# Calculating standard deviation
std_dev = np.std(measured_temperatures)
print("Standard deviation of measured temperatures (°C):", std_dev)

