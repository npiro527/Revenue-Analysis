# -*- coding: utf-8 -*-
"""
Created on Sat May  3 20:35:48 2025

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
perm_raw = filtered_data.query("ACCOUNT_CODE_NARRATIVE == 'Building And Alteration Permits' or ACCOUNT_CODE_NARRATIVE == 'Building and Alteration Permits'")

# Group separate accounts by year and build data frame for analysis
perm_yearly = perm_raw.groupby('CALENDAR_YEAR')["AMOUNT"].sum().reset_index()

#%%
## Plot data to visualize trends
# Construct basic plot showing permit fee revenue over time
perm_yearly.info()
perm_yearly.plot()

#%%
## Test to see if the data is stationary
# Run a Dickey Fuller test
dfuller = adfuller(perm_yearly["AMOUNT"])

# Display results
print(f"ADF Statistic: {dfuller[0]}")
print(f"p-value: {dfuller[1]}")
print("Critical Values:")
for key, value in dfuller[4].items():
    print(f"{key}: {value}")
   
# Data is not stationary (p-value is .313), so need integrated (i) model is needed ARIMA (p, q, d)

#%%
## Apply differencing to determine d
perm_yearly["Differenced_AMOUNT"] = perm_yearly["AMOUNT"].diff().dropna()
perm_yearly = perm_yearly.dropna(subset=["Differenced_AMOUNT"])

# Run Dickey-Fuller again to confirm stationarity
dfuller_diff = adfuller(perm_yearly["Differenced_AMOUNT"])
print("After Differencing:")
print(f"ADF Statistic: {dfuller_diff[0]}")
print(f"p-value: {dfuller_diff[1]}")
print("Critical Values:")
for key, value in dfuller_diff[4].items():
    print(f"{key}: {value}")

# P-value is .00002 after one differencing, so d=1

#%%
## Run partial autocorrelation funcation to determine Aurogregressive order (p)

plt.figure(figsize=(8, 5))
plot_pacf(perm_yearly["Differenced_AMOUNT"].dropna(), lags=8)
plt.title("Partial Autocorrelation Function (PACF)")
plt.xlabel("Lag")
plt.ylabel("PACF")
plt.show()

# no lags exceed the blue shadow, so p = 0

#%%
## Run autocorrelation function to determine MA order (q)

plt.figure(figsize=(8, 5))
plot_acf(perm_yearly["Differenced_AMOUNT"].dropna(), lags=8)
plt.title("Autocorrelation Function (ACF)")
plt.xlabel("Lag")
plt.ylabel("ACF")
plt.show()

# no lags exceed the blue shadow, so q = 0
# Recommended ARIMA order is (0, 1, 0)
#%%
## Run Auto ARIMA to double check manual selection
# Fit auto_arima to find the best p, d, q
auto_model = auto_arima(perm_yearly["AMOUNT"], seasonal=False, stepwise=True, trace=True)

# Display the best ARIMA model's summary
print(auto_model.summary())

# Construct recommended order
rec_order = auto_model.order
print(f"Ideal ARIMA order: {rec_order}")
# Best fit model is ARIMA(0, 1, 0), so manual selection above works