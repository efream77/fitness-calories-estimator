import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load dataset
df = pd.read_csv('model/calories_burned_dataset.csv')

# Encode categorical columns
gender_encoder = LabelEncoder()
activity_encoder = LabelEncoder()

df['Gender'] = gender_encoder.fit_transform(df['Gender'])
df['Activity'] = activity_encoder.fit_transform(df['Activity'])

# Features and target
X = df[['Age', 'Gender', 'Height','Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Activity']]
y = df['Calories_Burned']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f"Random Forest MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# Save the model + encoders
joblib.dump(model, 'model/calories_model.pkl')
joblib.dump({
    "gender": gender_encoder,
    "activity": activity_encoder
}, 'model/label_encoders.pkl')

print("Model and encoders saved successfully!")
