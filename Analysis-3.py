# -*- coding: utf-8 -*-
"""
Created on Sun May  4 20:22:14 2025

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
keep_columns = ["CALENDAR_YEAR", "MUNICIPAL_CODE", "ACCOUNT_CODE", "ACCOUNT_CODE_NARRATIVE", "AMOUNT", "SNAPSHOT_DATE"]
filtered_data = raw_data[keep_columns]

# Sort to only include permit fee data
park_raw = filtered_data.query("ACCOUNT_CODE_NARRATIVE == 'Parking Meter Fees Non-Taxable' or ACCOUNT_CODE_NARRATIVE == 'Parking Meter Fees (Non Taxable)'")

# Group separate accounts by year and build data frame for analysis
park_yearly = park_raw.groupby('CALENDAR_YEAR')["AMOUNT"].sum().reset_index()

#%%
## Plot data to visualize trends
# Construct basic plot showing permit fee revenue over time
park_yearly.info()
park_yearly.plot()

#%%
## Test to see if the data is stationary
# Run a Dickey Fuller test
dfuller = adfuller(park_yearly["AMOUNT"])

# Display results
print(f"ADF Statistic: {dfuller[0]}")
print(f"p-value: {dfuller[1]}")
print("Critical Values:")
for key, value in dfuller[4].items():
    print(f"{key}: {value}")
   
# Data is not stationary (p-value is 4.55), so need integrated (i) model is needed ARIMA (p, d, q)

#%%
## Apply differencing to determine d
park_yearly["Differenced_AMOUNT"] = park_yearly["AMOUNT"].diff().dropna()
park_yearly = park_yearly.dropna(subset=["Differenced_AMOUNT"])

# Run Dickey-Fuller again to confirm stationarity
dfuller_diff = adfuller(park_yearly["Differenced_AMOUNT"])
print("After Differencing:")
print(f"ADF Statistic: {dfuller_diff[0]}")
print(f"p-value: {dfuller_diff[1]}")
print("Critical Values:")
for key, value in dfuller_diff[4].items():
    print(f"{key}: {value}")
    
# p-value is .92, so must difference again
    
# Apply second order differencing
park_yearly["Differenced_AMOUNT_2"] = park_yearly["Differenced_AMOUNT"].diff().dropna()
park_yearly = park_yearly.dropna(subset=["Differenced_AMOUNT_2"])

# Run Dickey-Fuller again to confirm stationarity
dfuller_diff_2 = adfuller(park_yearly["Differenced_AMOUNT_2"])
print("After Second Differencing:")
print(f"ADF Statistic: {dfuller_diff_2[0]}")
print(f"p-value: {dfuller_diff_2[1]}")
print("Critical Values:")
for key, value in dfuller_diff_2[4].items():
    print(f"{key}: {value}")

# P-value is .048 after one differencing, so d=2

#%%
## Run partial autocorrelation funcation to determine Auroregressive order (p)

plt.figure(figsize=(8, 5))
plot_pacf(park_yearly["Differenced_AMOUNT_2"].dropna(), lags=8)
plt.title("Partial Autocorrelation Function (PACF)")
plt.xlabel("Lag")
plt.ylabel("PACF")
plt.show()

# 1 lag exceed the blue shadow, so p = 1

#%%
## Run autocorrelation function to determine MA order (q)

plt.figure(figsize=(8, 5))
plot_acf(park_yearly["Differenced_AMOUNT_2"].dropna(), lags=8)
plt.title("Autocorrelation Function (ACF)")
plt.xlabel("Lag")
plt.ylabel("ACF")
plt.show()

# No lags exceed the blue shadow, so q = 0
# Recommended ARIMA order is (1, 2, 0)
#%%
## Run Auto ARIMA to double check manual selection
# Fit auto_arima to find the best p, d, q
auto_model = auto_arima(park_yearly["AMOUNT"], seasonal=False, stepwise=True, trace=True)

# Display the best ARIMA model's summary
print(auto_model.summary())

# Construct recommended order
rec_order = auto_model.order
print(f"Ideal ARIMA order: {rec_order}")
# Best fit model is ARIMA(0, 1, 0), which is different than manual

#%%
## Manually construct recommended ARIMA
# Construct ARIMA (1, 2, 0)
arima = ARIMA(park_yearly["AMOUNT"], order=(1, 2, 0))
arima_fit = arima.fit()
print(arima_fit.summary())

#%%
## Forecast two year in the future
# Perform forecast with ARIMA
forecast_two_years = arima_fit.forecast(steps=2)

# Extend the index for the forecasted years
future_years = [park_yearly["CALENDAR_YEAR"].iloc[-1] + i for i in range(1, 3)]

# Plot data
plt.figure(figsize=(10, 6))
plt.plot(park_yearly["CALENDAR_YEAR"], park_yearly["AMOUNT"], marker='o', linestyle='-', label="Historical Data")
plt.scatter(future_years, forecast_two_years, color='red', label="Forecast (Next 2 Years)", marker='X', s=100)

plt.title("Parking Meter Fee (Non-Taxable) Revenue Forecast for Next 2 Years")
plt.xlabel("Year")
plt.ylabel("Amount (Millions of $)")
plt.legend()
plt.grid()

plt.show()