# -*- coding: utf-8 -*-
"""
Created on Sat May  3 19:51:27 2025

@author: npiro
"""

#%% 
## Prepare workspace
# Import packages
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import pandas as pd 

#%% Prepare data
# Read file and consolidate all sheets
raw_data = pd.read_excel("Syracuse-Revenue-Combined_Piro.xlsx", sheet_name=None)
raw_data = pd.concat(raw_data.values(), ignore_index=True)

# Consolidate sheets and select columns to keep
keep_columns = ["CALENDAR_YEAR", "MUNICIPAL_CODE", "ACCOUNT_CODE", "LEVEL_2_CATEGORY", "AMOUNT", "SNAPSHOT_DATE"]
filtered_data = raw_data[keep_columns]

# Sort to only include sales tax data
st_raw = filtered_data.query("LEVEL_2_CATEGORY == 'SALES TAX DISTRIBUTION' or LEVEL_2_CATEGORY == 'Sales Tax Distribution'")

# Group separate accounts by year and build data frame for analysis
st_yearly = st_raw.groupby('CALENDAR_YEAR')["AMOUNT"].sum().reset_index()

#%%
## Plot data to visualize trends
# Construct basic plot showing sales tax revenue over time
st_yearly.info()
st_yearly.plot()

#%%
## Test to see if the data is stationary
# Run a Dickey Fuller test
dfuller = adfuller(st_yearly["AMOUNT"])

# Display results
print(f"ADF Statistic: {dfuller[0]}")
print(f"p-value: {dfuller[1]}")
print("Critical Values:")
for key, value in dfuller[4].items():
    print(f"{key}: {value}")
   
# Data is not stationary (p-value is .996), so need integrated (i) model is needed ARIMA (p, q, d)

#%%
## Apply differencing to determine d
st_yearly["Differenced_AMOUNT"] = st_yearly["AMOUNT"].diff().dropna()
st_yearly = st_yearly.dropna(subset=["Differenced_AMOUNT"])

# Run Dickey-Fuller again to confirm stationarity
dfuller_diff = adfuller(st_yearly["Differenced_AMOUNT"])
print("After Differencing:")
print(f"ADF Statistic: {dfuller_diff[0]}")
print(f"p-value: {dfuller_diff[1]}")
print("Critical Values:")
for key, value in dfuller_diff[4].items():
    print(f"{key}: {value}")

# P-value is .016 after one differencing, so d=1

#%%
## Run partial autocorrelation funcation to determine Aurogregressive order (p)

plt.figure(figsize=(8, 5))
plot_pacf(st_yearly["Differenced_AMOUNT"].dropna(), lags=14)
plt.title("Partial Autocorrelation Function (PACF)")
plt.xlabel("Lag")
plt.ylabel("PACF")
plt.show()

# no lags exceed the blue shadow, so p = 0

#%%
## Run autocorrelation function to determine MA order (q)

plt.figure(figsize=(8, 5))
plot_acf(st_yearly["Differenced_AMOUNT"].dropna(), lags=14)
plt.title("Autocorrelation Function (ACF)")
plt.xlabel("Lag")
plt.ylabel("ACF")
plt.show()

# no lags exceed the blue shadow, so q = 0
# Recommended ARIMA order is (0, 1, 0)

#%%
## Run Auto ARIMA to double check manual selection
# Fit auto_arima to find the best p, d, q
auto_model = auto_arima(st_yearly["AMOUNT"], seasonal=False, stepwise=True, trace=True)

# Display the best ARIMA model's summary
print(auto_model.summary())

# Construct recommended order
rec_order = auto_model.order
print(f"Ideal ARIMA order: {rec_order}")
# Best fit model is ARIMA(0, 1, 0), so manual selection above works

#%%

# Construct ARIMA (0, 1, 0)
arima = ARIMA(st_yearly["AMOUNT"], order=(0, 1, 0))
arima_fit = arima.fit()
print(arima_fit.summary())

#%% 
## Forecast two year in the future
# Perform forecast with ARIMA
forecast_two_years = arima_fit.forecast(steps=2)

# Extend the index for the forecasted years
future_years = [st_yearly["CALENDAR_YEAR"].iloc[-1] + i for i in range(1, 3)]

# Plot data
plt.figure(figsize=(10, 6))
plt.plot(st_yearly["CALENDAR_YEAR"], st_yearly["AMOUNT"], marker='o', linestyle='-', label="Historical Data")
plt.scatter(future_years, forecast_two_years, color='red', label="Forecast (Next 2 Years)", marker='X', s=100)

plt.title("Sales Tax Revenue Forecast for Next 2 Years")
plt.xlabel("Year")
plt.ylabel("Amount ($)")
plt.legend()
plt.grid()

plt.show()