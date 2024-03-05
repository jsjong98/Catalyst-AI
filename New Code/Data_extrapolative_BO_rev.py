# %%
import pandas as pd

# Load the provided datasets
train_df = pd.read_csv(r'C:\Users\OJH\OneDrive\SKKU\연구자료\Catalyst AI_rev\New Database\Na_W_Mn_Cat_train_rev.csv')
test_df = pd.read_csv(r'C:\Users\OJH\OneDrive\SKKU\연구자료\Catalyst AI_rev\New Database\Na_W_Mn_Cat_test_rev.csv')

train_df_copy = train_df.copy()
test_df_copy = test_df.copy()
test_df_copy = test_df_copy.drop([0])
test_df_copy = test_df_copy.drop([53])
test_df_copy = test_df_copy.reset_index(drop=True)

# Display the first few rows of each dataset for an overview
train_df.head(), test_df.head()
# %%
# Extracting the independent variables (X) and the dependent variable (y)
X_train = train_df[['Temperature', 'pCH4_per_pO2', 'Contact time']]
y_train = train_df['Y(C2)_predicted']
X_test = test_df[['Temperature', 'pCH4_per_pO2', 'Contact time']]
y_test = test_df['Y(C2)_predicted']

X_train_copy = train_df_copy[['Temperature', 'pCH4_per_pO2', 'Contact time']]
y_train_copy = train_df_copy['Y(C2)_predicted']
X_test_copy = test_df_copy[['Temperature', 'pCH4_per_pO2', 'Contact time']]
y_test_copy = test_df_copy['Y(C2)_predicted']

# Determining the range of each variable in the training data
train_range = {
    'Temperature': (X_train_copy['Temperature'].min(), X_train_copy['Temperature'].max()),
    'pCH4_per_pO2': (X_train_copy['pCH4_per_pO2'].min(), X_train_copy['pCH4_per_pO2'].max()),
    'Contact time': (X_train_copy['Contact time'].min(), X_train_copy['Contact time'].max())
}
# Function to determine the extrapolation strength for a given observation
def determine_extrapolation_strength(row, train_range):
    strength = 0
    for col in train_range:
        if row[col] < train_range[col][0] or row[col] > train_range[col][1]:
            strength += 1
    return strength

# Applying the function to the test data
test_df_copy['Extrapolation Strength'] = test_df_copy.apply(lambda row: determine_extrapolation_strength(row, train_range), axis=1)

# Displaying the first few rows of the modified test data
test_df_copy.head()
# %%
extrapolation_1_hyperparameters = pd.read_csv(r'C:\Users\OJH\OneDrive\SKKU\연구자료\Catalyst AI_rev\Extrapolative_stength_BO_rev\extrapolation_strength_1_BO_rev.csv').drop(['Unnamed: 0','Data Order', 'Temperature', 'pCH4_per_pO2', 'Contact time', 'R2', 'RMSE', 'MAE'], axis=1)
extrapolation_2_hyperparameters = pd.read_csv(r'C:\Users\OJH\OneDrive\SKKU\연구자료\Catalyst AI_rev\Extrapolative_stength_BO_rev\extrapolation_strength_2_BO_rev.csv').drop(['Unnamed: 0','Data Order', 'Temperature', 'pCH4_per_pO2', 'Contact time', 'R2', 'RMSE', 'MAE'], axis=1)
extrapolation_3_hyperparameters = pd.read_csv(r'C:\Users\OJH\OneDrive\SKKU\연구자료\Catalyst AI_rev\Extrapolative_stength_BO_rev\extrapolative_strength_3_BO_rev.csv').drop(['Unnamed: 0','Data Order', 'Temperature', 'pCH4_per_pO2', 'Contact time', 'R2', 'RMSE', 'MAE'], axis=1)
# %%
Hard_extrapolation = pd.read_csv(r'C:\Users\OJH\OneDrive\SKKU\연구자료\Catalyst AI_rev\New Database\Hard_Extrapolation.csv')
Easy_extrapolation = pd.read_csv(r'C:\Users\OJH\OneDrive\SKKU\연구자료\Catalyst AI_rev\New Database\Easy_Extrapolation.csv')
# %%
extrapolation_1_hyperparameters=extrapolation_1_hyperparameters.drop([0])
extrapolation_1_hyperparameters=extrapolation_1_hyperparameters.drop([28])
# %%
extrapolation_1_hyperparameters
# %%
extrapolation_1_hyperparameters = extrapolation_1_hyperparameters.reset_index(drop=True)
extrapolation_1_hyperparameters
# %%
X_hard_extrapolation = Hard_extrapolation[['Temperature', 'pCH4_per_pO2', 'Contact time']].iloc[0]
y_hard_extrapolation = Hard_extrapolation[['Y(C2)_predicted']].iloc[0]

X_easy_extrapolation = Easy_extrapolation[['Temperature', 'pCH4_per_pO2', 'Contact time']].iloc[18]
y_easy_extrapolation = Easy_extrapolation[['Y(C2)_predicted']].iloc[18]

X_hard_extrapolation
X_easy_extrapolation
# %%
from sklearn.preprocessing import StandardScaler

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
X_hard_extrapolation_scaled = scaler_X.transform(X_hard_extrapolation.values.reshape(1, -1))
X_easy_extrapolation_scaled = scaler_X.transform(X_easy_extrapolation.values.reshape(1, -1))

y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))
y_hard_extrapolation_scaled = scaler_y.transform(y_hard_extrapolation.values.reshape(-1, 1))
y_easy_extrapolation_scaled = scaler_y.transform(y_easy_extrapolation.values.reshape(-1, 1))
# %%
X_train_scaled = pd.DataFrame(X_train_scaled, columns=['Temperature', 'pCH4_per_pO2', 'Contact time'])
X_test_scaled = pd.DataFrame(X_test_scaled, columns=['Temperature', 'pCH4_per_pO2', 'Contact time'])
X_hard_extrapolation_scaled = pd.DataFrame(X_hard_extrapolation_scaled, columns=['Temperature', 'pCH4_per_pO2', 'Contact time'])
X_easy_extrapolation_scaled = pd.DataFrame(X_easy_extrapolation_scaled, columns=['Temperature', 'pCH4_per_pO2', 'Contact time'])

y_train_scaled = pd.DataFrame(y_train_scaled, columns=['Y(C2)_predicted'])
y_test_scaled = pd.DataFrame(y_test_scaled, columns=['Y(C2)_predicted'])
y_hard_extrapolation_scaled = pd.DataFrame(y_hard_extrapolation_scaled, columns=['Y(C2)_predicted'])
y_easy_extrapolation_scaled = pd.DataFrame(y_easy_extrapolation_scaled, columns=['Y(C2)_predicted'])

X_test_scaled = X_test_scaled.drop([0])
X_test_scaled = X_test_scaled.drop([53])
X_test_scaled = X_test_scaled.reset_index(drop=True)

y_test_scaled = y_test_scaled.drop([0])
y_test_scaled = y_test_scaled.drop([53])
y_test_scaled = y_test_scaled.reset_index(drop=True)
# %%
print(X_test_scaled)
print(y_test_scaled)
# %%
X_train_scaled = X_train_scaled.to_numpy()
y_train_scaled = y_train_scaled.to_numpy()
X_test_scaled = X_test_scaled.to_numpy()
y_test_scaled = y_test_scaled.to_numpy()
X_hard_extrapolation_scaled = X_hard_extrapolation_scaled.to_numpy()
y_hard_extrapolation_scaled = y_hard_extrapolation_scaled.to_numpy()
X_easy_extrapolation_scaled = X_easy_extrapolation_scaled.to_numpy()
y_easy_extrapolation_scaled = y_easy_extrapolation_scaled.to_numpy()
# %%
from xgboost import XGBRegressor
import optuna
import os
import wandb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
import numpy as np
from dotenv import load_dotenv
import warnings

# 경고메세지 끄기
warnings.filterwarnings(action='ignore')

load_dotenv()
WANDB_AUTH_KEY = os.getenv('215ac9421ba42be98f8eb3d3fd77df35ca74aff4')
wandb.login(key=WANDB_AUTH_KEY)

wandb.init(entity='jonghwan_oh', project='Model prediction')
# %%
best_value = float('-inf')

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
import numpy as np

def cv_score_xgb(n_estimators, max_depth, min_child_weight, gamma, subsample, colsample_bytree, learning_rate):
    global best_value
    model = XGBRegressor(n_estimators=int(n_estimators), 
                         max_depth=int(max_depth), 
                         min_child_weight=min_child_weight,
                         gamma=gamma,
                         subsample=subsample,
                         colsample_bytree=colsample_bytree,
                         learning_rate=learning_rate,
                         tree_method='gpu_hist',
                         gpu_id=0, 
                         random_state=42)
    
    scores = cross_val_score(model, X_train_expanded, y_train_expanded, cv=5, scoring='neg_mean_squared_error')
    current_value = np.mean(scores)

    # Perform predictions with the model
    predictions_hard_extrapolation = model.predict(X_hard_extrapolation_scaled)
    predictions_easy_extrapolation = model.predict(X_easy_extrapolation_scaled)

    # Reshape the predictions to make them 2D arrays for the scaler
    predictions_hard_extrapolation_reshaped = predictions_hard_extrapolation.reshape(-1, 1)
    predictions_easy_extrapolation_reshaped = predictions_easy_extrapolation.reshape(-1, 1)

    scaled_hard_predictions_original = scaler_y.inverse_transform(predictions_hard_extrapolation_reshaped)
    scaled_easy_predictions_original = scaler_y.inverse_transform(predictions_easy_extrapolation_reshaped)

    # Flatten the scaled predictions to ensure they are 1-dimensional
    scaled_hard_predictions_flattened = scaled_hard_predictions_original.flatten()
    scaled_easy_predictions_flattened = scaled_easy_predictions_original.flatten()

    # Calculate the absolute errors
    # Assuming y_hard_extrapolation and y_easy_extrapolation are Pandas Series, we directly use the values without .values attribute
    hard_extrapolative_error = np.abs(y_hard_extrapolation.values - scaled_hard_predictions_flattened.values).values
    easy_extrapolative_error = np.abs(y_easy_extrapolation.values - scaled_easy_predictions_flattened.values).values

    hard_extrapolative_error_value = hard_extrapolative_error.item()
    easy_extrapolative_error_value = easy_extrapolative_error.item()

    wandb.log({
        'Temperature': selected_data_point_X[0].tolist(),
        'pCH4_per_pO2': selected_data_point_X[1].tolist(),
        'Contact time': selected_data_point_X[2].tolist(),
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'Hard extrapolative error': hard_extrapolative_error_value,
        'Easy extrapolative error': easy_extrapolative_error_value,  # Convert to list if necessary
        'n_estimators': int(extrapolation_2_hyperparameters['n_estimators'][index]),
        'max_depth': int(extrapolation_2_hyperparameters['max_depth'][index]),
        'min_child_weight': extrapolation_2_hyperparameters['min_child_weight'][index].tolist(),  # If needed
        'gamma': extrapolation_2_hyperparameters['gamma'][index].tolist(),  # If needed
        'subsample': extrapolation_2_hyperparameters['subsample'][index].tolist(),  # If needed
        'colsample_bytree': extrapolation_2_hyperparameters['colsample_bytree'][index].tolist(),  # If needed
        'learning_rate': extrapolation_2_hyperparameters['learning_rate'][index].tolist(),  # If needed
    })

    return current_value, hard_extrapolative_error, easy_extrapolative_error
# %%
X_test_scaled
# %%
extrapolation_2_indices = test_df_copy[test_df_copy['Extrapolation Strength'] == 2].index
X_test_extrapolation_2 = X_test_scaled[extrapolation_2_indices]
X_test_extrapolation_2 = pd.DataFrame(X_test_extrapolation_2, columns=['Temperature', 'pCH4_per_pO2', 'Contact time'])
X_test_extrapolation_2
# %%
len(y_test_scaled)
# %%
# Initialize best metrics and best point
best_rmse = float('inf')
best_mae = float('inf')
best_r2 = float('-inf')
best_point_rmse = None
best_point_mae = None
best_point_r2 = None

logs_df = pd.DataFrame(columns=['Data Order', 'Temperature', 'pCH4_per_pO2', 'Contact time',
                                'R2', 'RMSE', 'MAE', 'Hard extrapolative error', 'Easy extrapolative error',
                                'n_estimators', 'max_depth', 'min_child_weight',
                                'gamma', 'subsample', 'colsample_bytree', 'learning_rate'])

# 외삽 강도가 2인 데이터 포인트 선택
extrapolation_2_indices = test_df_copy[test_df_copy['Extrapolation Strength'] == 2].index
X_test_extrapolation_2 = X_test_scaled[extrapolation_2_indices]
y_test_extrapolation_2 = y_test_scaled[extrapolation_2_indices]
X_test_extrapolation_2 = pd.DataFrame(X_test_extrapolation_2, columns=['Temperature', 'pCH4_per_pO2', 'Contact time'])
y_test_extrapolation_2 = pd.DataFrame(y_test_extrapolation_2, columns=['Y(C2)_predicted'])
X_test_extrapolation_2 = X_test_extrapolation_2.reset_index(drop=True)
y_test_extrapolation_2 = y_test_extrapolation_2.reset_index(drop=True)

X_train_original = X_train_scaled.copy()
y_train_original = y_train_scaled.copy()
X_test_original = X_test_scaled.copy()
y_test_original = y_test_scaled.copy()
X_test_extrapolation_2_original = X_test_extrapolation_2.copy()
y_test_extrapolation_2_original = y_test_extrapolation_2.copy()

sample_ranking_rmse = []
sample_ranking_mae = []
sample_ranking_r2 = []

# 반복적으로 외삽 강도가 2인 데이터 포인트를 학습 데이터에 추가하고 테스트 데이터에서 제거
for index in X_test_extrapolation_2_original.index:

    print(index)

    selected_data_point_X = X_test_original[index]
    selected_data_point_y = y_test_original[index]

    X_train_expanded = np.vstack([X_train_original, selected_data_point_X])
    y_train_expanded = np.vstack([y_train_original, selected_data_point_y])

    X_test_reduced = np.delete(X_test_original, index, axis=0)
    y_test_reduced = np.delete(y_test_original, index, axis=0)

    model = XGBRegressor(n_estimators=int(extrapolation_2_hyperparameters['n_estimators'][index]), 
                         max_depth=int(extrapolation_2_hyperparameters['max_depth'][index]), 
                         min_child_weight=extrapolation_2_hyperparameters['min_child_weight'][index],
                         gamma=extrapolation_2_hyperparameters['gamma'][index],
                         subsample=extrapolation_2_hyperparameters['subsample'][index],
                         colsample_bytree=extrapolation_2_hyperparameters['colsample_bytree'][index],
                         learning_rate=extrapolation_2_hyperparameters['learning_rate'][index],
                         tree_method='gpu_hist',
                         gpu_id=0, 
                         random_state=42)
    
    model.fit(X_train_expanded, y_train_expanded.ravel())

    predictions = model.predict(X_test_reduced).flatten()
    rmse = np.sqrt(mean_squared_error(y_test_reduced, predictions))
    mae = mean_absolute_error(y_test_reduced, predictions)
    r2 = r2_score(y_test_reduced, predictions)

    # Perform predictions with the model
    predictions_hard_extrapolation = model.predict(X_hard_extrapolation_scaled)
    predictions_easy_extrapolation = model.predict(X_easy_extrapolation_scaled)

    # Reshape the predictions to make them 2D arrays for the scaler
    predictions_hard_extrapolation_reshaped = predictions_hard_extrapolation.reshape(-1, 1)
    predictions_easy_extrapolation_reshaped = predictions_easy_extrapolation.reshape(-1, 1)

    scaled_hard_predictions_original = scaler_y.inverse_transform(predictions_hard_extrapolation_reshaped)
    scaled_easy_predictions_original = scaler_y.inverse_transform(predictions_easy_extrapolation_reshaped)

    # Flatten the scaled predictions to ensure they are 1-dimensional
    scaled_hard_predictions_flattened = scaled_hard_predictions_original.flatten()
    scaled_easy_predictions_flattened = scaled_easy_predictions_original.flatten()

    # Calculate the absolute errors
    # Assuming y_hard_extrapolation and y_easy_extrapolation are Pandas Series, we directly use the values without .values attribute
    hard_extrapolative_error = np.abs(y_hard_extrapolation - scaled_hard_predictions_flattened).values
    easy_extrapolative_error = np.abs(y_easy_extrapolation - scaled_easy_predictions_flattened).values

    hard_extrapolative_error_value = hard_extrapolative_error.item()
    easy_extrapolative_error_value = easy_extrapolative_error.item()

    new_row = pd.DataFrame({'Data Order': [index],
                            'Temperature': [selected_data_point_X[0]],
                            'pCH4_per_pO2': [selected_data_point_X[1]],
                            'Contact time': [selected_data_point_X[2]],
                            'R2': [r2],
                            'RMSE': [rmse],
                            'MAE': [mae],
                            'Hard extrapolative error': [hard_extrapolative_error],
                            'Easy extrapolative error': [easy_extrapolative_error], 
                            'n_estimators': [int(extrapolation_2_hyperparameters['n_estimators'][index])],
                            'max_depth': [int(extrapolation_2_hyperparameters['max_depth'][index])],
                            'min_child_weight': [extrapolation_2_hyperparameters['min_child_weight'][index]],
                            'gamma': [extrapolation_2_hyperparameters['gamma'][index]],
                            'subsample': [extrapolation_2_hyperparameters['subsample'][index]],
                            'colsample_bytree': [extrapolation_2_hyperparameters['colsample_bytree'][index]],
                            'learning_rate': [extrapolation_2_hyperparameters['learning_rate'][index]]})
    logs_df = pd.concat([logs_df, new_row], ignore_index=True)
    
    wandb.log({
        'Temperature': selected_data_point_X[0].tolist(),
        'pCH4_per_pO2': selected_data_point_X[1].tolist(),
        'Contact time': selected_data_point_X[2].tolist(),
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'Hard extrapolative error': hard_extrapolative_error_value,
        'Easy extrapolative error': easy_extrapolative_error_value,  # Convert to list if necessary
        'n_estimators': int(extrapolation_2_hyperparameters['n_estimators'][index]),
        'max_depth': int(extrapolation_2_hyperparameters['max_depth'][index]),
        'min_child_weight': extrapolation_2_hyperparameters['min_child_weight'][index].tolist(),  # If needed
        'gamma': extrapolation_2_hyperparameters['gamma'][index].tolist(),  # If needed
        'subsample': extrapolation_2_hyperparameters['subsample'][index].tolist(),  # If needed
        'colsample_bytree': extrapolation_2_hyperparameters['colsample_bytree'][index].tolist(),  # If needed
        'learning_rate': extrapolation_2_hyperparameters['learning_rate'][index].tolist(),  # If needed
    })
    
    sample_ranking_rmse.append((index, rmse))
    sample_ranking_mae.append((index, mae))
    sample_ranking_r2.append((index, r2))

    # Check if thist point is the best
    if rmse < best_rmse:
        best_point_rmse = index
        best_rmse = rmse

    if mae < best_mae:
        best_point_mae = index
        best_mae = mae

    if r2 > best_r2:
        best_point_r2 = index
        best_r2 = r2
 
# Sort the sample rankings
sample_ranking_rmse = sorted(sample_ranking_rmse, key=lambda x: x[1])
sample_ranking_mae = sorted(sample_ranking_mae, key=lambda x: x[1])
sample_ranking_r2 = sorted(sample_ranking_r2, key=lambda x: x[1], reverse=True)

# Display the sample rankings based on RMSE and R2
print("Sample ranking based on RMSE:", sample_ranking_rmse)
print("Sample ranking based on MAE:", sample_ranking_mae)
print("Sample ranking based on R2:", sample_ranking_r2)
print(f"Best point based on RMSE: {best_point_rmse}")
print(f"Best point based on MAE: {best_point_mae}")
print(f"Best point based on R2: {best_point_r2}")

print(logs_df)
# %%
logs_df.to_csv('extrapolation_strength_2_BO.csv',index=False)
# %%
print(X_hard_extrapolation)
# %%
print(X_easy_extrapolation)
# %%
