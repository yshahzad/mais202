import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

# Load the dataset
file_path = "Data/Survival Analysis/Survival Analysis_arXiv_scrape.csv"
df = pd.read_csv(file_path)

# Convert 'dates' to datetime and extract 'Year'
df["Year"] = pd.to_datetime(df["dates"], errors="coerce").dt.year

# Group by 'Year' and count the number of publications per year
df = df.groupby("Year").size().reset_index(name="Publication_Count")



# Ensure 'Year' is in datetime format and set it as index
df["Year"] = pd.to_datetime(df["Year"], format="%Y")
df.set_index("Year", inplace=True)

# Plot the original data
plt.figure(figsize=(10, 5))
plt.plot(df, marker="o", linestyle="-", label="Actual Publications")
plt.xlabel("Year")
plt.ylabel("Number of Publications")
plt.title("Publication Trends Over Time")
plt.legend()
plt.show()

#take the log of each publication count (see special case of Box-Cox transformation)
# https://otexts.com/fpp2/transformations.html
df["Publication_Count"] = np.log(df["Publication_Count"])

# Automatically determine the best ARIMA parameters
best_model = auto_arima(df["Publication_Count"], seasonal=False, trace=True, stepwise=True)
print(best_model.summary())

# Train ARIMA model with the best (p, d, q)
p, d, q = best_model.order
model = ARIMA(df["Publication_Count"], order=(p, d, q))
model_fit = model.fit()

# Forecast the next 10 years
forecast_years = pd.date_range(start=df.index[-1] + pd.DateOffset(years=1), periods=10, freq="Y")
forecast = model_fit.forecast(steps=10)

# Convert forecast to DataFrame
forecast_df = pd.DataFrame({"Year": forecast_years, "Predicted_Publications": np.exp(forecast.values)}) # convert back the log trasnformation
forecast_df.set_index("Year", inplace=True)

#Reconvert into the real number of publication count
df["Publication_Count"] = np.exp(df["Publication_Count"])




# Plot results
plt.figure(figsize=(10, 5))
plt.plot(df, marker="o", linestyle="-", label="Actual Publications")
plt.plot(forecast_df, marker="o", linestyle="--", color="red", label="Forecast")
plt.xlabel("Year")
plt.ylabel("Number of Publications")
plt.title("ARIMA Forecast for Future Publications")
plt.legend()
plt.show()

# Display forecasted values
# Display forecasted values
print("\nARIMA Forecast Results:\n", forecast_df)

# Save forecast to a CSV file (optional)
forecast_df.to_csv("ARIMA_Forecast_Results.csv")
print("\nForecast saved as 'ARIMA_Forecast_Results.csv'.")
