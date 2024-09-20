# Part 1: Introduction to Python for Energy Engineers

# 1. Variables and Data Types
# Assigning values to variables
temperature = 25  # integer
pressure = 1.01325  # float (floating-point number)
gas = "Methane"  # string
system_operational = True  # boolean

print("The temperature is:", temperature, "°C")
print("The pressure is:", pressure, "bar")
print("The type of gas is:", gas)
print("Is the system operational?", system_operational)

# 2. Mathematical Operations
# Calculating kinetic energy
mass = 10  # kg
velocity = 5  # m/s
kinetic_energy = 0.5 * mass * velocity ** 2  # ** is the exponentiation operator

print("The kinetic energy is:", kinetic_energy, "J")

# 3. User Input
# Calculating potential energy
height = float(input("Enter the height in meters: "))
gravity = 9.81  # m/s^2
potential_energy = mass * gravity * height

print("The potential energy is:", potential_energy, "J")

# 4. Control Structures
# Determining the state of water based on temperature
water_temperature = float(input("Enter the water temperature in °C: "))

if water_temperature <= 0:
    state = "solid (ice)"
elif water_temperature >= 100:
    state = "gaseous (steam)"
else:
    state = "liquid"

print(f"At {water_temperature} °C, water is in the {state} state.")

# 5. Lists
# A list is an ordered and mutable collection of items
daily_temperatures = [20, 22, 24, 23, 25, 21, 19]  # list of integers
print("Daily temperatures:", daily_temperatures)

# Accessing elements of a list
first_temperature = daily_temperatures[0]  # index 0
print("The first temperature is:", first_temperature, "°C")

# Modifying an element in the list
daily_temperatures[2] = 26
print("Updated list of temperatures:", daily_temperatures)

# Adding an element to the list
daily_temperatures.append(18)
print("List of temperatures after adding a value:", daily_temperatures)

# 6. Dictionaries
# A dictionary is an unordered collection of key-value pairs
gas_properties = {
    "Methane": {"Molar Mass": 16.04, "Boiling Point": -161.5},
    "Ethane": {"Molar Mass": 30.07, "Boiling Point": -88.6},
    "Propane": {"Molar Mass": 44.10, "Boiling Point": -42.1}
}

print("Properties of Methane:", gas_properties["Methane"])

# Adding a new element to the dictionary
gas_properties["Butane"] = {"Molar Mass": 58.12, "Boiling Point": -0.5}
print("Updated gas properties dictionary:", gas_properties)

# 7. Loops
# Using a for loop with lists
print("List of daily temperatures:")
for temp in daily_temperatures:
    print(temp, "°C")

# Using a for loop with dictionaries
print("List of gas properties:")
for gas, properties in gas_properties.items():
    print(f"{gas}: Molar Mass = {properties['Molar Mass']} g/mol, Boiling Point = {properties['Boiling Point']} °C")

# 8. Functions
# Defining a function to calculate specific heat capacity
def specific_heat_capacity(mass, heat_capacity):
    """
    Calculates the specific heat capacity.
    mass: mass of the object in kg
    heat_capacity: heat capacity in J/K
    """
    c = heat_capacity / mass
    return c

object_mass = 5  # kg
heat_capacity = 2500  # J/K

c = specific_heat_capacity(object_mass, heat_capacity)
print("The specific heat capacity of the object is:", c, "J/(kg·K)")

# 9. Importing Modules
# Using the math module for advanced calculations
import math

# Calculating the natural logarithm
number = 10
logarithm = math.log(number)
print("The natural logarithm of", number, "is:", logarithm)
