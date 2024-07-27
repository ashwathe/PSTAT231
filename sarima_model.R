# Install and load necessary packages
install.packages("forecast")
install.packages("xts")
library(forecast)
library(lubridate)

# Load data
load("data/data_clean.RData")

dates <- data_clean$Date

data <- data.matrix(data_clean[,-1])
# Standardize data --> center around mean for each column 
mean <- apply(data, 2, mean)
std <- apply(data, 2, sd)
data <- scale(data, center = mean, scale = std)

# Normalize, create func. --> make between 0 and 1 for activation function 
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
max <- apply(data, 2, max)
min <- apply(data, 2, min)

# Normalize data & get rid of adjusted close 
data_normalized <- apply(data, 2, normalize)

#Combine all datasets
df <- as.data.frame(data_normalized)
df <- cbind(Date = dates, df)

# Convert 'date' to Date class and set it as the time index
ts_data <- ts(df$Close, frequency = 1, start = c(year(df$Date[1]), month(df$Date[1])))

# Number of days for the rolling window
window_size <- 6

# Initialize an empty dataframe to store forecasts
forecasts <- data.frame(Date = as.Date(character()), Actual = numeric(), ARIMA = numeric(), SARIMA = numeric())

# Perform rolling window forecast
for (i in (window_size + 1):length(ts_data)) {
  # Extract the current window
  current_window <- ts_data[(i - window_size):(i - 1)]
  
  # SARIMA Model (Seasonal)
  sarima_model <- auto.arima(current_window, seasonal = TRUE)
  
  # Forecast the next day
 
  sarima_forecast <- forecast(sarima_model, h = 1)
  
  # Store the results in the forecasts dataframe
  forecasts <- rbind(forecasts, data.frame(Date = time(ts_data)[i], 
                                           Actual = ts_data[i],
                                           ARIMA = arima_forecast$mean[1],  # Extract the first forecast value
                                           SARIMA = sarima_forecast$mean[1]))  # Extract the first forecast value
  }


#Visualize forecasts for SARIMA
plot(ts_data, type = "l", col = "blue", lwd = 2, main = "SARIMA Model Forecast", xlab = "Date", ylab = "Closing Prices")
lines(forecasts$Date, forecasts$Actual, col = "black", lwd = 2, lty = 2, type = "b", pch = 16)
lines(forecasts$Date, forecasts$SARIMA, col = "green", lwd = 2, type = "b", pch = 16)
legend("topright", legend = c("Actual", "SARIMA Forecast"), col = c("black","green"), lwd = 2, pch = 16)

actual_values <- forecasts$Actual
sarima_forecast_values <- forecasts$SARIMA

# Remove missing values, if any
actual_values <- actual_values[!is.na(sarima_forecast_values)]
sarima_forecast_values <- sarima_forecast_values[!is.na(sarima_forecast_values)]

# Calculate squared errors
squared_errors <- (sarima_forecast_values - actual_values)^2

# Calculate mean squared error
mse <- mean(squared_errors)

# Calculate root mean squared error
rmse <- sqrt(mse)

# Print RMSE value
cat("RMSE for SARIMA model:", rmse, "\n")


