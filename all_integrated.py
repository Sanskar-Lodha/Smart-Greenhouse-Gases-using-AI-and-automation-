import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


weather = pd.read_csv("3994114.csv", index_col="DATE")
weather.columns = weather.columns.str.lower()
null_pct = weather.isnull().mean()
weather = weather[null_pct[null_pct < 0.05].index]
weather = weather.ffill()
weather.index = pd.to_datetime(weather.index)

forecast_targets = ['tmin', 'tmax', 'prcp']
model = Ridge(alpha=0.1)


def pct_diff(old, new):
    return (new - old) / old

def compute_rolling(df, horizon, col):
    label = f"rolling_{horizon}_{col}"
    df[label] = df[col].rolling(horizon).mean()
    df[f"{label}_pct"] = pct_diff(df[label], df[col])
    return df

def expand_mean(df):
    return df.expanding(1).mean()

rolling_horizons = [3, 14]
for h in rolling_horizons:
    for col in forecast_targets:
        weather = compute_rolling(weather, h, col)

weather = weather.fillna(0)

for col in forecast_targets:
    weather[f"month_avg_{col}"] = weather[col].groupby(weather.index.month, group_keys=False).apply(expand_mean)
    weather[f"day_avg_{col}"] = weather[col].groupby(weather.index.day_of_year, group_keys=False).apply(expand_mean)


predictors = weather.columns[~weather.columns.isin(['name', 'station'])]

# tmax
weather['target'] = weather.shift(-1)['tmax']
model.fit(weather.iloc[:-1][predictors], weather.iloc[:-1]['target'])
tmax_pred = model.predict(weather.iloc[-1:][predictors])[0]

# tmin
weather['target'] = weather.shift(-1)['tmin']
model.fit(weather.iloc[:-1][predictors], weather.iloc[:-1]['target'])
tmin_pred = model.predict(weather.iloc[-1:][predictors])[0]

# prcp
weather['target'] = weather.shift(-1)['prcp']
model.fit(weather.iloc[:-1][predictors], weather.iloc[:-1]['target'])
prcp_pred = model.predict(weather.iloc[-1:][predictors])[0]

# Humidity (estimated as midpoint between tmin and tmax)
humidity_pred = (tmin_pred + tmax_pred) / 2

# --- Display Forecast ---
# --- Display Forecast ---
print(f"\n🌾 Forecast for tomorrow:")
print(f"Temperature (min): {tmin_pred:.2f} °F")
print(f"Temperature (max): {tmax_pred:.2f} °F")
print(f"Rainfall: {prcp_pred:.2f} mm")
print(f"Humidity (est.): {humidity_pred:.2f} %")


# --- Crop prediction based on weather forecast ---
# Load crop dataset
crop_df = pd.read_csv("Crop_production.csv")  # Replace with correct path

# Check correct column name
if 'Crop' not in crop_df.columns:
    raise ValueError("Column 'Crop' is missing from the dataset!")

# Use only numeric columns
numeric_cols = crop_df.select_dtypes(include='number').columns.tolist()

if not numeric_cols:
    raise ValueError("No numeric columns found for training the crop model!")

# Drop missing values
crop_df = crop_df.dropna(subset=numeric_cols + ['Crop'])

# Encode crop labels
le = LabelEncoder()
crop_df['Crop'] = le.fit_transform(crop_df['Crop'])

# Train crop recommendation model
X = crop_df[numeric_cols]
y = crop_df['Crop']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

crop_model = RandomForestClassifier(n_estimators=100, random_state=42)
crop_model.fit(X_train, y_train)

# Build input for crop prediction
avg_temp = (tmin_pred + tmax_pred) / 2
input_data = {
    'temperature': [avg_temp],
    'humidity': [humidity_pred],
    'rainfall': [prcp_pred]
}

input_df = pd.DataFrame(input_data)

# Match training columns
input_df = input_df.reindex(columns=X.columns, fill_value=0)

# Predict crop
predicted_crop = crop_model.predict(input_df)
predicted_crop_name = le.inverse_transform(predicted_crop)

# --- Final output ---
print(f"\n✅ Recommended Crop to Grow: {predicted_crop_name[0]}")
