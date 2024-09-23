# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For linear regression and statistical analysis
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from math import sqrt
import os

if '__file__' in locals():
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
else:
    data_path = os.path.join(os.path.dirname(__name__), 'data')


# 1 - Data Preparation
# Read the CSV file containing pre-retrofit data
# Replace 'data/pre_retrofit.csv' with the correct path to your CSV file
df = pd.read_csv(os.path.join(data_path, 'pre_retrofit.csv'))

# Convert 'DATA_misura' and 'DATA_inizio' to datetime objects
df['DATA_misura'] = pd.to_datetime(df['DATA_misura'], format="%Y-%m-%d")
df['DATA_inizio'] = pd.to_datetime(df['DATA_inizio'], format="%Y-%m-%d")

# Calculate the number of days between measurement dates
df['giorni_letture'] = (df['DATA_misura'] - df['DATA_inizio']).dt.days

# Calculate total hours in the metering period
df['ore_tot'] = df['giorni_letture'] * 24

# Calculate the hours when the heat generator is ON
df['ore_ON'] = df['ore_tot'] - df['giorni_off'] * 24

# Define the conversion factor
cf = 10.94

# Calculate the average thermal power
df['potenza'] = (df['consumo_gasm_smc'] * cf) / df['ore_ON']

# Replace infinite or NaN values with 0
df['potenza'].replace([np.inf, -np.inf], 0, inplace=True)
df['potenza'].fillna(0, inplace=True)

# Filter out records where 'tipo_misura' is 'OFF'
df1 = df[df['tipo_misura'] != 'OFF']

# 2 - Fitting Linear Models for Each Season
def fit_linear_model(data, season):
    """
    Fits a linear model of 'potenza' vs 'T_media' for a given season.
    """
    # Filter data for the specific season
    season_data = data[data['Stagione'] == season]

    # Define independent and dependent variables
    X = season_data['T_media']
    y = season_data['potenza']

    # Add a constant term for the intercept
    X = sm.add_constant(X)

    # Fit the linear regression model
    model = sm.OLS(y, X).fit()

    return model

# List of seasons
seasons = ['2012-2013', '2013-2014', '2014-2015', '2015-2016']

# Dictionary to store models
models = {}

# Fit models for each season
for season in seasons:
    models[season] = fit_linear_model(df1, season)
    print(f"Summary for season {season}:")
    print(models[season].summary())


# 3 - Plotting the Regression Lines
def plot_regression(data, model, season):
    """
    Plots the regression line and data points for a given season.
    """
    # Filter data for the season
    season_data = data[data['Stagione'] == season]

    # Create scatter plot of the data points
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='T_media', y='potenza', hue='tipo_misura', data=season_data, s=50)

    # Generate values for plotting the regression line
    X_plot = np.linspace(season_data['T_media'].min(), season_data['T_media'].max(), 100)
    X_plot_const = sm.add_constant(X_plot)
    y_plot = model.predict(X_plot_const)

    # Plot the regression line
    plt.plot(X_plot, y_plot, color='blue', linestyle='--', linewidth=2)

    # Set plot limits
    plt.xlim(0, 20)
    plt.ylim(0, 70)

    # Set labels and title
    plt.xlabel('Average External Temperature [°C]')
    plt.ylabel('Power [kW]')
    plt.title(f"Season {season}\n"
              f"R² = {model.rsquared:.2f}, "
              f"Intercept = {model.params['const']:.2f}, "
              f"Slope = {model.params['T_media']:.2f}")

    plt.legend()
    plt.show()

for season in seasons:
    plot_regression(df1, models[season], season)

# Create subplots for all seasons
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

for i, season in enumerate(seasons):
    season_data = df1[df1['Stagione'] == season]
    model = models[season]

    sns.scatterplot(x='T_media', y='potenza', hue='tipo_misura', data=season_data, s=50, ax=axes[i])

    X_plot = np.linspace(season_data['T_media'].min(), season_data['T_media'].max(), 100)
    X_plot_const = sm.add_constant(X_plot)
    y_plot = model.predict(X_plot_const)
    axes[i].plot(X_plot, y_plot, color='blue', linestyle='--', linewidth=2)

    axes[i].set_xlim(0, 20)
    axes[i].set_ylim(0, 70)
    axes[i].set_xlabel('Average External Temperature [°C]')
    axes[i].set_ylabel('Power [kW]')
    axes[i].set_title(f"Season {season}\n"
                      f"R² = {model.rsquared:.2f}, "
                      f"Intercept = {model.params['const']:.2f}, "
                      f"Slope = {model.params['T_media']:.2f}")

    axes[i].legend()

plt.tight_layout()
plt.show()


# 4 - Overall Model
# Filter data where 'tipo_misura' is not 'OFF'
df_tot = df[df['tipo_misura'] != 'OFF']

# Independent and dependent variables
X_tot = df_tot['T_media']
y_tot = df_tot['potenza']

# Add a constant term
X_tot = sm.add_constant(X_tot)

# Fit the linear regression model
lm_tot = sm.OLS(y_tot, X_tot).fit()

# Display the summary
print("Summary of the overall model:")
print(lm_tot.summary())

# Plot the overall regression
plt.figure(figsize=(10, 8))
sns.scatterplot(x='T_media', y='potenza', hue='Stagione', data=df_tot, s=50)

# Regression line
X_plot = np.linspace(df_tot['T_media'].min(), df_tot['T_media'].max(), 100)
X_plot_const = sm.add_constant(X_plot)
y_plot = lm_tot.predict(X_plot_const)
plt.plot(X_plot, y_plot, color='blue', linestyle='--', linewidth=2)

# Plot residuals (optional)
df_tot['pred'] = lm_tot.predict(sm.add_constant(df_tot['T_media']))
for _, row in df_tot.iterrows():
    plt.plot([row['T_media'], row['T_media']], [row['potenza'], row['pred']], color='red', linewidth=0.3)

# Set plot limits
plt.xlim(0, 20)
plt.ylim(0, 70)

# Set labels and title
plt.xlabel('Average External Temperature [°C]')
plt.ylabel('Power [kW]')
plt.title(f"Overall Regression\n"
          f"R² = {lm_tot.rsquared:.2f}, "
          f"Intercept = {lm_tot.params['const']:.2f}, "
          f"Slope = {lm_tot.params['T_media']:.2f}")

plt.legend()
plt.show()

# 5 - Model Evaluation

# QQ plot of residuals
sm.qqplot(lm_tot.resid, line='s')
plt.title('QQ Plot of Residuals')
plt.show()

# Histogram of residuals
plt.hist(lm_tot.resid, bins=20)
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# Add predictions to the DataFrame
df1['pred'] = lm_tot.predict(sm.add_constant(df1['T_media']))

# Calculate Mean Absolute Percentage Error (MAPE)
mape = mean_absolute_percentage_error(df1['potenza'], df1['pred']) * 100
print(f"MAPE: {mape:.2f}%")

# Calculate Root Mean Squared Error (RMSE)
rmse = sqrt(mean_squared_error(df1['potenza'], df1['pred']))
print(f"RMSE: {rmse:.2f}")

# Calculate Coefficient of Variation of RMSE (CVRMSE)
cvrmse = (rmse / df1['potenza'].mean()) * 100
print(f"CVRMSE: {cvrmse:.2f}%")

# 6 - Post-Retrofit Analysis
# Read the CSV file containing post-retrofit data
# Replace 'data/post_retrofit.csv' with the correct path to your CSV file
post_retrofit_1 = pd.read_csv(os.path.join(data_path, 'post_retrofit.csv'))

# Convert date columns to datetime objects with the correct format
post_retrofit_1['DATA_misura'] = pd.to_datetime(post_retrofit_1['DATA_misura'], format="%d/%m/%Y")
post_retrofit_1['DATA_inizio'] = pd.to_datetime(post_retrofit_1['DATA_inizio'], format="%d/%m/%Y")

# Calculate the number of days between measurements
post_retrofit_1['giorni_letture'] = (post_retrofit_1['DATA_misura'] - post_retrofit_1['DATA_inizio']).dt.days

# Calculate total hours
post_retrofit_1['ore_tot'] = post_retrofit_1['giorni_letture'] * 24

# Calculate hours when the generator is ON
post_retrofit_1['ore_ON'] = post_retrofit_1['ore_tot'] - post_retrofit_1['giorni_off'] * 24

# Calculate the average thermal power
post_retrofit_1['potenza'] = (post_retrofit_1['consumo_gasm_smc'] * cf) / post_retrofit_1['ore_ON']

# Visualization of the Post-Retrofit Data
# Plot pre-retrofit data
plt.figure(figsize=(10, 8))
ax = sns.scatterplot(x='T_media', y='potenza', hue='Stagione', data=df1, s=50)

# Plot post-retrofit data using plt.scatter
plt.scatter(post_retrofit_1['T_media'], post_retrofit_1['potenza'], s=50, color='black', label='Post-retrofit')

# Regression line from the overall model
X_plot = np.linspace(
    min(df_tot['T_media'].min(), post_retrofit_1['T_media'].min()),
    max(df_tot['T_media'].max(), post_retrofit_1['T_media'].max()),
    100
)
X_plot_const = sm.add_constant(X_plot)
y_plot = lm_tot.predict(X_plot_const)
plt.plot(X_plot, y_plot, color='blue', linestyle='--', linewidth=2, label='Regression Line')

# Set plot limits
plt.xlim(0, 20)
plt.ylim(0, 70)

# Set labels and title
plt.xlabel('Average External Temperature [°C]')
plt.ylabel('Power [kW]')
plt.title(
    f"Overall Regression with Post-Retrofit Data\n"
    f"R² = {lm_tot.rsquared:.2f}, "
    f"Intercept = {lm_tot.params['const']:.2f}, "
    f"Slope = {lm_tot.params['T_media']:.2f}"
)

plt.legend()

plt.show()


# Predict average thermal power using the baseline model
post_retrofit_1['pred'] = lm_tot.predict(sm.add_constant(post_retrofit_1['T_media']))

# Calculate the estimated gas consumption
post_retrofit_1['est_cons'] = post_retrofit_1['pred'] * post_retrofit_1['ore_ON'] / cf

# Calculate energy saving percentage
actual_consumption = post_retrofit_1['consumo_gasm_smc'].sum()
estimated_consumption = post_retrofit_1['est_cons'].sum()

energy_saving = ((estimated_consumption - actual_consumption) /
                 estimated_consumption) * 100

print(f"Energy Saving: {energy_saving:.2f}%")




