import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import joblib  # For saving and loading models
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load datasets from local files
train = pd.read_csv(r"C:\predictive_maintenance\train_FD001.txt", sep=r'\s+', header=None)
test = pd.read_csv(r"C:\predictive_maintenance\test_FD001.txt", sep=r'\s+', header=None)
rul = pd.read_csv(r"C:\predictive_maintenance\rUL_FD001.txt", sep=r'\s+', header=None)

# Step 2: Assign column names to the datasets
columns = ['engine_id', 'time_in_cycles'] + [f'sensor{i}' for i in range(1, 22)] + ["sensor22", "sensor23", "sensor24"]
train.columns = columns
test.columns = columns

# Display the first 5 rows of each dataset to see if the data is loaded correctly
print("Train Dataset:")
print(train.head())
print("\nTest Dataset:")
print(test.head())
print("\nRUL Dataset:")
print(rul.head())

# Step 3: Check for missing values in the training data and handle them by forward filling
print("\nMissing values in Train Dataset:")
print(train.isnull().sum())
train.ffill(inplace=True)  # Forward fill to handle missing values

# Step 4: Remove constant features (columns with zero variance)
std_dev = train.iloc[:, 2:].std()  # Calculate standard deviation for each column, excluding engine_id and time_in_cycles
constant_features = std_dev[std_dev == 0].index
train.drop(columns=constant_features, inplace=True)
test.drop(columns=constant_features, inplace=True)

# Ensure all sensor columns are float64 before scaling
print("\nData types before casting:")
print(train.dtypes)

# Step 5: Explicitly cast sensor columns to float64 before scaling
train.iloc[:, 2:] = train.iloc[:, 2:].astype('float64')
test.iloc[:, 2:] = test.iloc[:, 2:].astype('float64')

# Confirm the data types after casting to ensure they are correct
print("\nData types after casting:")
print(train.dtypes)

# Step 6: Scale the sensor data to the range [0, 1] using MinMaxScaler
scaler = MinMaxScaler()
train_scaled = train.copy()
test_scaled = test.copy()
train_scaled.iloc[:, 2:] = scaler.fit_transform(train.iloc[:, 2:])
test_scaled.iloc[:, 2:] = scaler.transform(test.iloc[:, 2:])

# Step 7: Remove duplicate rows in the training data to avoid redundancy
train.drop_duplicates(inplace=True)

# Step 8: Calculate the Remaining Useful Life (RUL) for each entry in the training data
max_cycle = train.groupby('engine_id')['time_in_cycles'].max()  # Get the maximum cycle for each engine
train['RUL'] = train.apply(lambda row: max_cycle[row['engine_id']] - row['time_in_cycles'], axis=1)

# Separate the features (X) and target (y) for the training data
X_train = train_scaled.iloc[:, 2:-1]  # Exclude engine_id, time_in_cycles, and RUL
y_train = train['RUL']

# Step 9: Load the RUL file and prepare the test set
# Get the last cycle for each engine in the test set and add the RUL
test_last_cycle = test.groupby('engine_id').last().reset_index()
test_last_cycle['RUL'] = rul[0]
X_test = test_last_cycle.iloc[:, 2:-1]  # Features for test set
y_test = test_last_cycle['RUL']  # Target for test set

# Splitting Data

# Split the training data into training and validation sets
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print(f"Training Set Shape: {X_train_split.shape}")
print(f"Validation Set Shape: {X_val.shape}")

# Training a baseline model

# Starting with Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_split, y_train_split)

# Predict RUL on the validation set
y_pred = rf_model.predict(X_val)

# Evaluate the model performance using RMSE
mse = mean_squared_error(y_val, y_pred)
rmse = mse ** 0.5
print(f"Validation RMSE: {rmse}")

# Saving the Random Forest model using joblib
joblib.dump(rf_model, 'predictive_model.pkl')

# Experimenting with models - Gradient Boosting
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train_split, y_train_split)
gb_y_pred = gb_model.predict(X_val)

# Visualizations (Optional - these parts are useful but not necessary for the core functionality)

# Plot the distribution of RUL in the training data
plt.figure(figsize=(10, 6))
sns.histplot(train['RUL'], kde=True, bins=30, color='blue')
plt.title("Distribution of RUL in Training Data")
plt.xlabel("RUL")
plt.ylabel("Frequency")
plt.show()

# Plot sensor trends for a specific engine (e.g., engine_id = 1) to see how sensor readings change over time
engine_1_data = train[train['engine_id'] == 1]
plt.figure(figsize=(12, 8))
for sensor in train.columns[2:-1]:  # Loop over sensor columns
    plt.plot(engine_1_data['time_in_cycles'], engine_1_data[sensor], label=sensor)
plt.title("Sensor Trends for Engine 1")
plt.xlabel("Time in Cycles")
plt.ylabel("Sensor Readings")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Plot a correlation heatmap to see the relationships between sensor readings
plt.figure(figsize=(12, 8))
correlation_matrix = train.iloc[:, 2:].corr()  # Correlation among sensor data and RUL
sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", fmt='.2f')
plt.title("Correlation Heatmap of Features")
plt.show()

# Plot predicted vs actual RUL to see how well the model is performing
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_val, y=y_pred)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], '--r', linewidth=2)
plt.title("Predicted vs Actual RUL")
plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")
plt.show()

# Visualizing feature importance to understand which sensors are most important for the model
importances = rf_model.feature_importances_
features = X_train.columns

plt.figure(figsize=(12, 8))
sns.barplot(x=importances, y=features, color="blue")
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()
