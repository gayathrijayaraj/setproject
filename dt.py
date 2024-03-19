import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Load gold price data (replace 'your_data.csv' with your actual file path)
df = pd.read_csv('/workspaces/hello-world/set1.csv')
df['Date'] = pd.to_datetime(df['Date']).dt.day

# Select relevant features (consider adding more based on your data)
features = df['Date', 'Volume']  # Example features
target = df['Price']
df['Change'] = df['Price'].diff()  # Calculate price change
df['Up/Down'] = (df['Change'] > 0).astype(int)

# Data preprocessing (handle missing values, normalization if needed)
# ... (Add data cleaning steps as required)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Create the decision tree model with hyperparameter tuning (consider GridSearchCV for optimization)
model = DecisionTreeRegressor(max_depth=5, min_samples_split=2)  # Initial parameters

# Train the model
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate model performance (use metrics like R-squared, Mean Squared Error)
from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R-squared: {r2:.4f}")
print(f"Mean Squared Error: {mse:.2f}")

# Feature importance analysis (optional)
importances = model.feature_importances_
feature_names = features

for feature, importance in zip(feature_names, importances):
    print(f"{feature}: {importance:.4f}")
