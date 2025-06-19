import joblib
import pandas as pd

# Load model and encoders
model = joblib.load('xgboost_model.pkl')

soil_encoder = joblib.load('soil_encoder.pkl')
# print("Available soil types:", soil_encoder.classes_)

region_encoder = joblib.load('region_encoder.pkl')
# print("Available regions:", region_encoder.classes_)

crop_encoder = joblib.load('crop_encoder.pkl')
# print("Available crops:", crop_encoder.classes_)

weather_encoder = joblib.load('weather_encoder.pkl')
# print("Available weather conditions:", weather_encoder.classes_)

# Fake input
region = region_encoder.transform(['West'])[0]
soil = soil_encoder.transform(['Sandy'])[0]
crop = crop_encoder.transform(['Cotton'])[0]
weather = weather_encoder.transform(['Cloudy'])[0]

rainfall = 200.0
temp = 28.0
fertilizer = 1.0
irrigation = 1.0
days_to_harvest = 120


# Features
rainfall_fertilizer_sum = rainfall + fertilizer
temp_fertilizer_rainfall_interaction = temp * rainfall_fertilizer_sum
rainfall_temp_fertilizer_ratio = (rainfall + 1) / (temp + 1) / (fertilizer + 1)


# DataFrame
input_data = pd.DataFrame([[
    region, soil, crop, rainfall, temp, fertilizer, irrigation,
    weather, days_to_harvest, rainfall_fertilizer_sum,
    temp_fertilizer_rainfall_interaction, rainfall_temp_fertilizer_ratio
]], columns=[
    'Region', 'Soil_Type', 'Crop', 'Rainfall_mm', 'Temperature_Celsius',
    'Fertilizer_Used', 'Irrigation_Used', 'Weather_Condition',
    'Days_to_Harvest', 'Rainfall_Fertilizer_Sum',
    'Temp_Fertilizer_Rainfall_Interaction',
    'Rainfall_Temperature_Fertilizer_Ratio'
])

# Prediction
prediction = model.predict(input_data)[0]
print(f"Predicted yield: {prediction:.2f} tons/hectare")

# Finding the best crop
best_yield = 0
best_crop_name = None
for c in crop_encoder.classes_:
    test_data = input_data.copy()
    test_data['Crop'] = crop_encoder.transform([c])[0]
    test_prediction = model.predict(test_data)[0]
    if test_prediction > best_yield:
        best_yield = test_prediction
        best_crop_name = c

print(f"Best crop: {best_crop_name} with estimated yield: {best_yield:.2f} tons/hectare")
