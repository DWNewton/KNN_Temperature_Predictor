import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the weather data from the CSV file
file_path = "spotsylvania_refined_weather_data.csv"  # Update this path as needed
weather_data = pd.read_csv(file_path)

# Convert 'date' and 'time' columns into a single datetime column for temporal features
weather_data["datetime"] = pd.to_datetime(
    weather_data["date"] + " " + weather_data["time"]
)

# Drop unnecessary columns for prediction
weather_data = weather_data.drop(columns=["date", "time"])

# Sort by datetime to maintain temporal order
weather_data = weather_data.sort_values(by="datetime")

# Add aggregated features for the previous 3 days
weather_data["prev_3day_avg_temp"] = (
    weather_data["temp_f"].rolling(window=3).mean().shift(1)
)
weather_data["prev_3day_avg_cloud"] = (
    weather_data["cloud"].rolling(window=3).mean().shift(1)
)
weather_data["prev_3day_avg_wind"] = (
    weather_data["wind_mph"].rolling(window=3).mean().shift(1)
)
weather_data["prev_3day_total_precip"] = (
    weather_data["precip_in"].rolling(window=3).sum().shift(1)
)

# Drop rows with NaN values (due to rolling window)
weather_data = weather_data.dropna()

# Extract features and target
features = weather_data[
    [
        "prev_3day_avg_temp",
        "prev_3day_avg_cloud",
        "prev_3day_avg_wind",
        "prev_3day_total_precip",
    ]
]
target = weather_data["temp_f"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# Perform cross-validation to find the optimal k
param_grid = {"n_neighbors": range(1, 21)}  # Test k values from 1 to 20
grid = GridSearchCV(
    KNeighborsRegressor(), param_grid, scoring="neg_mean_squared_error", cv=5
)
grid.fit(X_train, y_train)

# Extract the best k value and refit the model
best_k = grid.best_params_["n_neighbors"]
print(f"Optimal k: {best_k}")
knn = KNeighborsRegressor(n_neighbors=best_k)
knn.fit(X_train, y_train)

# Evaluate the model using RMSE
y_pred = knn.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Model RMSE with optimal k ({best_k}): {rmse:.3f}")

# Prompt the user for a date
input_date = input("Enter a date (YYYY-MM-DD) to predict the temperature: ")
try:
    input_date = pd.to_datetime(input_date)

    # Ensure there is sufficient data for the previous 3 days
    recent_days = weather_data[weather_data["datetime"] < input_date]
    if len(recent_days) >= 3:
        input_features = (
            recent_days[["temp_f", "cloud", "wind_mph", "precip_in"]]
            .tail(3)
            .agg(
                {
                    "temp_f": "mean",
                    "cloud": "mean",
                    "wind_mph": "mean",
                    "precip_in": "sum",
                }
            )
            .values
        )
        prediction = knn.predict([input_features])
        print(
            f"Predicted Temperature for {input_date.strftime('%Y-%m-%d')}: {prediction[0]:.2f}°F"
        )
    else:
        print("Not enough recent data available for prediction.")
except ValueError:
    print("Invalid date format. Please enter a valid date in YYYY-MM-DD format.")

