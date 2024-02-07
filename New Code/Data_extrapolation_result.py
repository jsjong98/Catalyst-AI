# %%
import pandas as pd

# Load the provided datasets
train_df = pd.read_csv(r'C:\Users\OJH\OneDrive\SKKU\연구자료\Catalyst AI_rev\New Database\Na_W_Mn_train.csv')
test_df = pd.read_csv(r'C:\Users\OJH\OneDrive\SKKU\연구자료\Catalyst AI_rev\New Database\Na_W_Mn_test.csv')

# Display the first few rows of each dataset for an overview
train_df.head(), test_df.head()
# %%
# Dropping the 'Unnamed: 0' column as it's not needed for the analysis
train_df.drop(columns='Unnamed: 0', inplace=True)
test_df.drop(columns='Unnamed: 0', inplace=True)

# Extracting the independent variables (X) and the dependent variable (y)
X_train = train_df[['Temperature', 'pCH4_per_pO2', 'Contact time']]
y_train = train_df['Y(C2)_predicted']
X_test = test_df[['Temperature', 'pCH4_per_pO2', 'Contact time']]
y_test = test_df['Y(C2)_predicted']

# Determining the range of each variable in the training data
train_range = {
    'Temperature': (X_train['Temperature'].min(), X_train['Temperature'].max()),
    'pCH4_per_pO2': (X_train['pCH4_per_pO2'].min(), X_train['pCH4_per_pO2'].max()),
    'Contact time': (X_train['Contact time'].min(), X_train['Contact time'].max())
}
# Function to determine the extrapolation strength for a given observation
def determine_extrapolation_strength(row, train_range):
    strength = 0
    for col in train_range:
        if row[col] < train_range[col][0] or row[col] > train_range[col][1]:
            strength += 1
    return strength

# Applying the function to the test data
test_df['Extrapolation Strength'] = test_df.apply(lambda row: determine_extrapolation_strength(row, train_range), axis=1)

# Displaying the first few rows of the modified test data
test_df.head()
# %%
# Function to identify which variables are contributing to the extrapolation strength for each observation
def identify_extrapolating_variables(row, train_range):
    extrapolating_vars = []
    for col in train_range:
        if row[col] < train_range[col][0] or row[col] > train_range[col][1]:
            extrapolating_vars.append(col)
    return extrapolating_vars

# Applying the function to the test data
test_df['Extrapolating Variables'] = test_df.apply(lambda row: identify_extrapolating_variables(row, train_range), axis=1)

# Filtering the test data for extrapolation strengths 1 and 2
extrapolation_1_df = test_df[test_df['Extrapolation Strength'] == 1]
extrapolation_2_df = test_df[test_df['Extrapolation Strength'] == 2]

# Identifying the unique combinations of extrapolating variables for extrapolation strengths 1 and 2
unique_extrapolating_vars_1 = extrapolation_1_df['Extrapolating Variables'].value_counts()
unique_extrapolating_vars_2 = extrapolation_2_df['Extrapolating Variables'].value_counts()

unique_extrapolating_vars_1, unique_extrapolating_vars_2
# %%
from sklearn.preprocessing import StandardScaler

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

# %%
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

best_order_3 = 7386
best_order_2 = 7778
best_order_1 = 7523

X_train_original = X_train_scaled.copy()
y_train_original = y_train_scaled.copy()
X_test_original = X_test_scaled.copy()
y_test_original = y_test_scaled.copy()

selected_data_point_X = X_test_original[best_order_3]
selected_data_point_y = y_test_original[best_order_3]

X_train_expanded = np.vstack([X_train_original, selected_data_point_X])
y_train_expanded = np.vstack([y_train_original, selected_data_point_y])

X_test_reduced = np.delete(X_test_original, best_order_3, axis=0)
y_test_reduced = np.delete(y_test_original, best_order_3, axis=0)

best_params_3 = {'n_estimators': 527, 'max_depth': 2, 'min_child_weight': 6.35670516445723, 'gamma': 0.0319207029438828, 'subsample': 0.666207936133433, 'colsample_bytree': 0.732451312222895, 'learning_rate': 0.0591186824330719}
best_params_2 = {'n_estimators': 936, 'max_depth': 2, 'min_child_weight': 4.8137748444094, 'gamma': 0.0108624992226107, 'subsample': 0.595477457861828, 'colsample_bytree': 0.82919448840144, 'learning_rate': 0.0367718266424177}
best_params_1 = {'n_estimators': 893, 'max_depth': 50, 'min_child_weight': 6.690636709, 'gamma': 0.015915062, 'subsample': 0.681697512, 'colsample_bytree': 0.535367949, 'learning_rate': 0.060951398}

model = XGBRegressor(n_estimators=int(best_params_3['n_estimators']), 
                     max_depth=int(best_params_3['max_depth']), 
                     min_child_weight=best_params_3['min_child_weight'],
                     gamma=best_params_3['gamma'],
                     subsample=best_params_3['subsample'],
                     colsample_bytree=best_params_3['colsample_bytree'],
                     learning_rate=best_params_3['learning_rate'], 
                     random_state=42)

model.fit(X_train_expanded, y_train_expanded.ravel())

predictions = model.predict(X_test_reduced).flatten()
rmse = np.sqrt(mean_squared_error(y_test_reduced, predictions))
mae = mean_absolute_error(y_test_reduced, predictions)
r2 = r2_score(y_test_reduced, predictions)
# %%
y_test_reduced_original = scaler_y.inverse_transform(y_test_reduced)
predictions_original = scaler_y.inverse_transform(predictions.reshape(-1, 1))

X_test_reduced_original = scaler_X.inverse_transform(X_test_reduced)
pd.DataFrame(X_test_reduced_original, columns=['Temperature', 'pCH4_per_pO2', 'Contact time'])
# %%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Output the performance metrics
print(f"RMSE: {rmse}, MAE: {mae}, R2: {r2}")

Temperature = X_test_reduced_original[:, 0]
pCH4_per_pO2 = X_test_reduced_original[:, 1]
Contact_time = X_test_reduced_original[:, 2]

residuals = predictions_original - y_test_reduced_original

fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax1.scatter(Temperature, pCH4_per_pO2, predictions_original, color='red', label='Predicted', alpha=0.6)
ax1.set_title('Predicted Y(C2)')
ax1.set_xlabel('Temperature')
ax1.set_ylabel('pCH4_per_pO2')
ax1.set_zlabel('Predicted Y(C2)')
ax1.legend()

ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.scatter(Temperature, pCH4_per_pO2, y_test_reduced_original, color='blue', label='Actual', alpha=0.6)
ax2.set_title('Actual Y(C2)')
ax2.set_xlabel('Temperature')
ax2.set_ylabel('pCH4_per_pO2')
ax2.set_zlabel('Actual Y(C2)')
ax2.legend()

ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax3.scatter(Temperature, pCH4_per_pO2, residuals, color='green', label='Residuals', alpha=0.6)
ax3.set_title('Residuals')
ax3.set_xlabel('Temperature')
ax3.set_ylabel('pCH4_per_pO2')
ax3.set_zlabel('Residuals')
ax3.legend()

plt.tight_layout()

plt.show()
# %%
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax1.scatter(Temperature, Contact_time, predictions_original, color='red', label='Predicted', alpha=0.6)
ax1.set_title('Predicted Y(C2)')
ax1.set_xlabel('Temperature')
ax1.set_ylabel('Contact time')
ax1.set_zlabel('Predicted Y(C2)')
ax1.legend()

ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.scatter(Temperature, Contact_time, y_test_reduced_original, color='blue', label='Actual', alpha=0.6)
ax2.set_title('Actual Y(C2)')
ax2.set_xlabel('Temperature')
ax2.set_ylabel('Contact time')
ax2.set_zlabel('Actual Y(C2)')
ax2.legend()

ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax3.scatter(Temperature, Contact_time, residuals, color='green', label='Residuals', alpha=0.6)
ax3.set_title('Residuals')
ax3.set_xlabel('Temperature')
ax3.set_ylabel('Contact time')
ax3.set_zlabel('Residuals')
ax3.legend()

plt.tight_layout()

plt.show()
# %%
