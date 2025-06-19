import pandas as pd
import os
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ====== Load the data ======
file_path = os.path.join(os.getcwd(), '002crop_yield.csv')
data = pd.read_csv(file_path)

# ====== Encode categorical columns ======
region_encoder = LabelEncoder()
soil_encoder = LabelEncoder()
crop_encoder = LabelEncoder()
weather_encoder = LabelEncoder()

data['Region'] = region_encoder.fit_transform(data['Region'])
data['Soil_Type'] = soil_encoder.fit_transform(data['Soil_Type'])
data['Crop'] = crop_encoder.fit_transform(data['Crop'])
data['Weather_Condition'] = weather_encoder.fit_transform(data['Weather_Condition'])

# ====== Create new features ======
data['Rainfall_Fertilizer_Sum'] = data['Rainfall_mm'] + data['Fertilizer_Used']
data['Temp_Fertilizer_Rainfall_Interaction'] = (
    data['Temperature_Celsius'] * data['Rainfall_Fertilizer_Sum']
)
data['Rainfall_Temperature_Fertilizer_Ratio'] = (
    (data['Rainfall_mm'] + 1) / (data['Temperature_Celsius'] + 1) / (data['Fertilizer_Used'] + 1)
)

# ====== Split the data ======
X = data.drop('Yield_tons_per_hectare', axis=1)
y = data['Yield_tons_per_hectare']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ====== Train the model ======
model = XGBRegressor(objective='reg:squarederror', n_estimators=50, learning_rate=0.1, max_depth=5)
model.fit(X_train, y_train)

# ====== Evaluate the model ======
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R² Score: {r2}")

# ====== Save the model and encoders ======
joblib.dump(model, 'xgboost_model.pkl')
joblib.dump(region_encoder, 'region_encoder.pkl')
joblib.dump(soil_encoder, 'soil_encoder.pkl')
joblib.dump(crop_encoder, 'crop_encoder.pkl')
joblib.dump(weather_encoder, 'weather_encoder.pkl')

print("Model and encoders saved successfully.")
