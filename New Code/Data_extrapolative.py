# %%
import pandas as pd

# Load the provided datasets
train_df = pd.read_csv(r'C:\Users\OJH\OneDrive\SKKU\연구자료\Catalyst_AI_rev\New Database\Na_W_Mn_train.csv')
test_df = pd.read_csv(r'C:\Users\OJH\OneDrive\SKKU\연구자료\Catalyst_AI_rev\New Database\Na_W_Mn_test.csv')

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
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
import numpy as np

def cv_score_xgb(n_estimators, max_depth, min_child_weight, gamma, subsample, colsample_bytree, learning_rate):
    model = XGBRegressor(n_estimators=int(n_estimators), 
                         max_depth=int(max_depth), 
                         min_child_weight=min_child_weight,
                         gamma=gamma,
                         subsample=subsample,
                         colsample_bytree=colsample_bytree,
                         learning_rate=learning_rate, 
                         random_state=42)
    
    scores = cross_val_score(model, X_train_expanded, y_train_expanded, cv=5, scoring='neg_mean_squared_error')
    return np.mean(scores)
# %%
# Initialize best metrics and best point
best_rmse = float('inf')
best_mae = float('inf')
best_r2 = float('-inf')
best_point_rmse = None
best_point_mae = None
best_point_r2 = None

logs_df = pd.DataFrame(columns=['Data Order', 'Temperature', 'pCH4_per_pO2', 'Contact time',
                                'R2', 'RMSE', 'MAE', 'n_estimators', 'max_depth', 'min_child_weight',
                                'gamma', 'subsample', 'colsample_bytree', 'learning_rate'])

# 외삽 강도가 1인 데이터 포인트 선택
extrapolation_3_indices = test_df[test_df['Extrapolation Strength'] == 3].index

X_train_original = X_train_scaled.copy()
y_train_original = y_train_scaled.copy()
X_test_original = X_test_scaled.copy()
y_test_original = y_test_scaled.copy()

sample_ranking_rmse = []
sample_ranking_mae = []
sample_ranking_r2 = []

# 반복적으로 외삽 강도가 1인 데이터 포인트를 학습 데이터에 추가하고 테스트 데이터에서 제거
for index in extrapolation_3_indices:

    selected_data_point_X = X_test_original[index]
    selected_data_point_y = y_test_original[index]

    # 선택한 데이터 포인트를 학습 데이터에 추가
    X_train_expanded = np.vstack([X_train_original, selected_data_point_X])
    y_train_expanded = np.vstack([y_train_original, selected_data_point_y])

    # 동일한 데이터 포인트를 테스트 데이터에서 제거
    X_test_reduced = np.delete(X_test_original, index, axis=0)
    y_test_reduced = np.delete(y_test_original, index, axis=0)

    pbounds = {'n_estimators': (10, 1000), 'max_depth': (1, 50), 'min_child_weight': (1, 10), 'gamma': (0, 10), 'subsample': (0.5, 1),'colsample_bytree': (0.5, 1),'learning_rate': (0.0001, 0.1)}
    optimizer = BayesianOptimization(f=cv_score_xgb, pbounds=pbounds, random_state=42)
    optimizer.maximize(init_points=10, n_iter=100)

    best_params = optimizer.max['params']

    model = XGBRegressor(n_estimators=int(best_params['n_estimators']), 
                         max_depth=int(best_params['max_depth']), 
                         min_child_weight=best_params['min_child_weight'],
                         gamma=best_params['gamma'],
                         subsample=best_params['subsample'],
                         colsample_bytree=best_params['colsample_bytree'],
                         learning_rate=best_params['learning_rate'], 
                         random_state=42)
    
    selected_data_point_X_original = scaler_X.inverse_transform(selected_data_point_X)
    pd.DataFrame(selected_data_point_X_original.reshape(1, -1), columns=['Temperature', 'pCH4_per_pO2', 'Contact time'])

    model.fit(X_train_expanded, y_train_expanded.ravel())

    predictions = model.predict(X_test_reduced).flatten()
    rmse = np.sqrt(mean_squared_error(y_test_reduced, predictions))
    mae = mean_absolute_error(y_test_reduced, predictions)
    r2 = r2_score(y_test_reduced, predictions)

    new_row = pd.DataFrame({'Data Order': [index],
                            'Temperature': [selected_data_point_X[0]],
                            'pCH4_per_pO2': [selected_data_point_X[1]],
                            'Contact time': [selected_data_point_X[2]],
                            'R2': [r2],
                            'RMSE': [rmse],
                            'MAE': [mae],
                            'n_estimators': [int(best_params['n_estimators'])],
                            'max_depth': [int(best_params['max_depth'])],
                            'min_child_weight': [best_params['min_child_weight']],
                            'gamma': [best_params['gamma']],
                            'subsample': [best_params['subsample']],
                            'colsample_bytree': [best_params['colsample_bytree']],
                            'learning_rate': [best_params['learning_rate']]})
    logs_df = pd.concat([logs_df, new_row], ignore_index=True)

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
logs_df
# %%
# Display the sample rankings based on RMSE and R2
print("Sample ranking based on RMSE:", sample_ranking_rmse)
print("Sample ranking based on MAE:", sample_ranking_mae)
print("Sample ranking based on R2:", sample_ranking_r2)
print(f"Best point based on RMSE: {best_point_rmse}")
print(f"Best point based on MAE: {best_point_mae}")
print(f"Best point based on R2: {best_point_r2}")
# %%
logs_df.to_csv('extrapolation_strength_3_results.csv')
# %%
best_rmse = float('inf')
best_mae = float('inf')
best_r2 = float('-inf')
best_point_rmse_2 = None
best_point_mae_2 = None
best_point_r2_2 = None

logs_df_2 = pd.DataFrame(columns=['Data Order', 'Temperature', 'pCH4_per_pO2', 'Contact time',
                                'R2', 'RMSE', 'MAE', 'n_estimators', 'max_depth', 'min_child_weight',
                                'gamma', 'subsample', 'colsample_bytree', 'learning_rate'])

# 외삽 강도가 1인 데이터 포인트 선택
extrapolation_2_indices = test_df[test_df['Extrapolation Strength'] == 2].index

X_train_original = X_train_scaled.copy()
y_train_original = y_train_scaled.copy()
X_test_original = X_test_scaled.copy()
y_test_original = y_test_scaled.copy()

sample_ranking_rmse = []
sample_ranking_mae = []
sample_ranking_r2 = []

# 반복적으로 외삽 강도가 1인 데이터 포인트를 학습 데이터에 추가하고 테스트 데이터에서 제거
for index in extrapolation_2_indices:

    selected_data_point_X = X_test_original[index]
    selected_data_point_y = y_test_original[index]

    # 선택한 데이터 포인트를 학습 데이터에 추가
    X_train_expanded = np.vstack([X_train_original, selected_data_point_X])
    y_train_expanded = np.vstack([y_train_original, selected_data_point_y])

    # 동일한 데이터 포인트를 테스트 데이터에서 제거
    X_test_reduced = np.delete(X_test_original, index, axis=0)
    y_test_reduced = np.delete(y_test_original, index, axis=0)

    pbounds = {'n_estimators': (10, 1000), 'max_depth': (1, 50), 'min_child_weight': (1, 10), 'gamma': (0, 10), 'subsample': (0.5, 1),'colsample_bytree': (0.5, 1),'learning_rate': (0.0001, 0.1)}
    optimizer = BayesianOptimization(f=cv_score_xgb, pbounds=pbounds, random_state=42)
    optimizer.maximize(init_points=10, n_iter=100)

    best_params = optimizer.max['params']

    model = XGBRegressor(n_estimators=int(best_params['n_estimators']), 
                         max_depth=int(best_params['max_depth']), 
                         min_child_weight=best_params['min_child_weight'],
                         gamma=best_params['gamma'],
                         subsample=best_params['subsample'],
                         colsample_bytree=best_params['colsample_bytree'],
                         learning_rate=best_params['learning_rate'], 
                         random_state=42)
    
    model.fit(X_train_expanded, y_train_expanded.ravel())

    predictions = model.predict(X_test_reduced).flatten()
    rmse = np.sqrt(mean_squared_error(y_test_reduced, predictions))
    mae = mean_absolute_error(y_test_reduced, predictions)
    r2 = r2_score(y_test_reduced, predictions)

    new_row_2 = pd.DataFrame({'Data Order': [index],
                            'Temperature': [selected_data_point_X[0]],
                            'pCH4_per_pO2': [selected_data_point_X[1]],
                            'Contact time': [selected_data_point_X[2]],
                            'R2': [r2],
                            'RMSE': [rmse],
                            'MAE': [mae],
                            'n_estimators': [int(best_params['n_estimators'])],
                            'max_depth': [int(best_params['max_depth'])],
                            'min_child_weight': [best_params['min_child_weight']],
                            'gamma': [best_params['gamma']],
                            'subsample': [best_params['subsample']],
                            'colsample_bytree': [best_params['colsample_bytree']],
                            'learning_rate': [best_params['learning_rate']]})
    logs_df_2 = pd.concat([logs_df_2, new_row_2], ignore_index=True)

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
sample_ranking_rmse_2 = sorted(sample_ranking_rmse, key=lambda x: x[1])
sample_ranking_mae_2 = sorted(sample_ranking_mae, key=lambda x: x[1])
sample_ranking_r2_2 = sorted(sample_ranking_r2, key=lambda x: x[1], reverse=True)

# Display the sample rankings based on RMSE and R2
print("Sample ranking based on RMSE:", sample_ranking_rmse_2)
print("Sample ranking based on MAE:", sample_ranking_mae_2)
print("Sample ranking based on R2:", sample_ranking_r2_2)
print(f"Best point based on RMSE: {best_point_rmse_2}")
print(f"Best point based on MAE: {best_point_mae_2}")
print(f"Best point based on R2: {best_point_r2_2}")

print(logs_df)
# %%
logs_df_2
# %%
logs_df_2.to_csv('extrapolation_strength_2_results.csv')
# %%
