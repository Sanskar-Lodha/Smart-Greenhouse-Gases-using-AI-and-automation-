import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Load and clean data
weather = pd.read_csv("3994114.csv", index_col="DATE")
weather = weather.ffill()

# Drop columns with too many nulls
null_pct = weather.apply(pd.isnull).sum() / weather.shape[0]
valid_columns = weather.columns[null_pct < 0.05]
weather = weather[valid_columns].copy()
weather.columns = weather.columns.str.lower()

# Convert index to datetime
weather.index = pd.to_datetime(weather.index)

# Create shifted target columns (1-day-ahead forecast)
forecast_targets = ["tmax", "tmin", "prcp", "snow"]
for target in forecast_targets:
    weather[f"target_{target}"] = weather.shift(-1)[target]

# Feature engineering
def pct_diff(old, new):
    return (new - old) / old

def compute_rolling(weather, horizon, col):
    label = f"rolling_{horizon}_{col}"
    weather[label] = weather[col].rolling(horizon).mean()
    weather[f"{label}_pct"] = pct_diff(weather[label], weather[col])
    return weather

rolling_horizon = [3, 14]
for horizon in rolling_horizon:
    for col in ["tmax", "tmin", "prcp", "snow"]:
        if col in weather.columns:
            weather = compute_rolling(weather, horizon, col)

def expand_mean(df):
    return df.expanding(1).mean()

for col in ["tmax", "tmin", "prcp", "snow"]:
    if col in weather.columns:
        weather[f"month_avg_{col}"] = weather[col].groupby(weather.index.month, group_keys=False).apply(expand_mean)
        weather[f"day_avg_{col}"] = weather[col].groupby(weather.index.day_of_year, group_keys=False).apply(expand_mean)

# Fill any remaining nulls
weather = weather.fillna(0)

# Backtest function
def backtest(weather, model, predictors, target_col, start=3650, step=90):
    all_predictions = []
    for i in range(start, weather.shape[0], step):
        train = weather.iloc[:i]
        test = weather.iloc[i:(i+step)]

        model.fit(train[predictors], train[target_col])
        preds = model.predict(test[predictors])

        preds = pd.Series(preds, index=test.index)
        combined = pd.concat([test[target_col], preds], axis=1)
        combined.columns = ["actual", "prediction"]
        combined["diff"] = (combined["prediction"] - combined["actual"]).abs()
        all_predictions.append(combined)
    return pd.concat(all_predictions)

# Create predictor set (excluding targets and irrelevant columns)
excluded = ["station", "name"] + [f"target_{t}" for t in forecast_targets]
predictors = weather.columns[~weather.columns.isin(excluded)]

# Model and results for each target
results = {}
model = Ridge(alpha=0.1)

for target in forecast_targets:
    target_col = f"target_{target}"
    predictions = backtest(weather, model, predictors, target_col)
    mae = mean_absolute_error(predictions["actual"], predictions["prediction"])
    print(f"\n🧪 Forecasting 1-day-ahead '{target.upper()}':")
    print(f"MAE: {mae:.3f}")
    print(f"Average absolute diff: {predictions['diff'].mean():.3f}\n")
    results[target] = predictions

    # Optional: Plot actual vs predicted
    predictions[["actual", "prediction"]].plot(title=f"1-Day Ahead Forecast for {target.upper()}")
    plt.show()
