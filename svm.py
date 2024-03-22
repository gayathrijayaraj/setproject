import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

# Load gold price data (replace 'your_data.csv' with your actual file path)
data = pd.read_csv('/workspaces/setproject/set1.csv')

# Define features and target variable (replace 'Close' with your target column)
features = ['Open', 'High', 'Low']  # Adjust features as needed
target = 'Price'

# Scale features (optional but recommended)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[features])

# Split data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(scaled_data, data[target], test_size=0.2)

# Create and train the SVC model
svc_model = SVC()
svc_model.fit(X_train, y_train)

# Make predictions on test data
predictions = svc_model.predict(X_test)

# Evaluate model performance (optional)
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse:.2f}")

# You can also use other evaluation metrics like R-squared

