# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r'C:\Users\OJH\OneDrive\SKKU\연구자료\Catalyst AI\New Database\Na_W_Mn_Cat_preprocess_operation.csv')

data.head()
# %%
data.columns = [col.replace('(', '').replace(')', '').replace('/', '_per_').replace(',','') for col in data.columns]

column_names = data.columns.tolist()
column_names
# %%
# Na 2.1 mol%, W 1.05 mol%, Mn 2.25 mol%, Si 94.6 mol% (Na) --> Na2WO4/SiO2 catalyst (OCM)
import pandas as pd

revised_data = pd.read_csv(r'C:\Users\OJH\OneDrive\SKKU\연구자료\Catalyst AI\New Database\Na_W_Mn_Cat_preprocess_top5.csv')

revised_data.drop(['Preparation'], axis=1, inplace=True)

revised_data
# %%
revised_data.columns = [col.replace('(', '').replace(')', '').replace('/', '_per_').replace(',','') for col in revised_data.columns]

column_names = revised_data.columns.tolist()
column_names

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
revised_data = scaler.fit_transform(revised_data)

revised_data = pd.DataFrame(revised_data, columns=column_names)

from sklearn.model_selection import train_test_split

input_columns = [col for col in column_names if col not in ['YC2']]
output_column = 'YC2'

X = revised_data[input_columns]
y = revised_data[output_column]

# Splitting the dataset into training (90%), and test (10%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# %%
from sklearn.model_selection import cross_val_score
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import warnings
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score

best_params = {}

best_params = {}

def cv_score_rfr(n_estimators, max_depth, min_samples_split, min_samples_leaf):
    model = RandomForestRegressor(n_estimators=int(n_estimators), 
                                  max_depth=int(max_depth), 
                                  min_samples_split=int(min_samples_split), 
                                  min_samples_leaf=int(min_samples_leaf), 
                                  random_state=42
    )
    
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    return np.mean(scores)

pbounds = {
    'n_estimators': (10, 1000),
    'max_depth': (1, 50),
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 10)
}

optimizer = BayesianOptimization(f=cv_score_rfr, pbounds=pbounds, random_state=42)
optimizer.maximize(init_points=50, n_iter=500)
best_params['RF'] = optimizer.max['params']

# Function to optimize of the hyperparameters of the XGBoost
def cv_score_xgb(n_estimators, max_depth, min_child_weight, gamma, subsample, colsample_bytree, learning_rate):
    model = XGBRegressor(n_estimators=int(n_estimators), 
                         max_depth=int(max_depth), 
                         min_child_weight=min_child_weight, 
                         gamma=gamma, 
                         subsample=subsample, 
                         colsample_bytree=colsample_bytree, 
                         learning_rate=learning_rate, 
                         random_state=42
    )
    
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    return np.mean(scores)

pbounds = {
    'n_estimators': (10, 1000),
    'max_depth': (1, 50),
    'min_child_weight': (1, 10),
    'gamma': (0, 10),
    'subsample': (0.5, 1),
    'colsample_bytree': (0.5, 1),
    'learning_rate': (0.0001, 0.1)
}

optimizer = BayesianOptimization(f=cv_score_xgb, pbounds=pbounds, random_state=42)
optimizer.maximize(init_points=50, n_iter=500)
best_params['XGB'] = optimizer.max['params']

def cv_score_lgbm(n_estimators, max_depth, min_child_weight, min_child_samples, subsample, colsample_bytree, learning_rate):
    model = LGBMRegressor(n_estimators=int(n_estimators),
                            max_depth=int(max_depth),
                            min_child_weight=min_child_weight,
                            min_child_samples=int(min_child_samples),
                            subsample=subsample,
                            colsample_bytree=colsample_bytree,
                            learning_rate=learning_rate,
                            random_state=42
        )
    
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    return np.mean(scores)

pbounds = {
    'n_estimators': (10, 1000),
    'max_depth': (1, 50),
    'min_child_weight': (1, 10),
    'min_child_samples': (1, 10),
    'subsample': (0.5, 1),
    'colsample_bytree': (0.5, 1),
    'learning_rate': (0.0001, 0.1)
}

optimizer = BayesianOptimization(f=cv_score_lgbm, pbounds=pbounds, random_state=42)
optimizer.maximize(init_points=50, n_iter=500)
best_params['LGBM'] = optimizer.max['params']

from sklearn.ensemble import ExtraTreesRegressor
def cv_score_etr(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, max_leaf_nodes, min_impurity_decrease):
    model = ExtraTreesRegressor(n_estimators=int(n_estimators), 
                                  max_depth=int(max_depth), 
                                  min_samples_split=int(min_samples_split), 
                                  min_samples_leaf=int(min_samples_leaf), 
                                  max_features=max_features, 
                                  max_leaf_nodes=int(max_leaf_nodes), 
                                  min_impurity_decrease=min_impurity_decrease, 
                                  random_state=42
    )
    
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    return np.mean(scores)

pbounds ={
    'n_estimators': (10, 1000),
    'max_depth': (1, 50),
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 10),
    'max_features': (0.5, 1),
    'max_leaf_nodes': (2, 10),
    'min_impurity_decrease': (0, 0.1)
}

optimizer = BayesianOptimization(f=cv_score_etr, pbounds=pbounds, random_state=42)
optimizer.maximize(init_points=50, n_iter=500)
best_params['ETR'] = optimizer.max['params']

from catboost import CatBoostRegressor

def cv_score_cbr(iterations, learning_rate, depth, l2_leaf_reg):
    model = CatBoostRegressor(iterations=int(iterations), 
                              learning_rate=learning_rate, 
                              depth=int(depth), 
                              l2_leaf_reg=l2_leaf_reg, 
                              random_state=42
    )
    
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    return np.mean(scores)

pbounds = {"iterations": (1, 100), 
           "learning_rate": (0.001, 0.1), 
           "depth": (1, 10), 
           "l2_leaf_reg": (1, 10)}

optimizer = BayesianOptimization(f=cv_score_cbr, pbounds=pbounds, random_state=42)
optimizer.maximize(init_points=50, n_iter=500)
best_params['CBR'] = optimizer.max['params']
# %%
from sklearn.metrics import mean_squared_error, r2_score

validation_rmse = {}
validation_r2 = {}

from sklearn.metrics import mean_squared_error, r2_score

validation_rmse = {}
validation_r2 = {}

# Random Forest
best_rfr_params = best_params['RF']
best_rfr_params['n_estimators'] = int(round(best_rfr_params['n_estimators']))
best_rfr_params['max_depth'] = int(round(best_rfr_params['max_depth']))
best_rfr_params['min_samples_split'] = int(round(best_rfr_params['min_samples_split']))
best_rfr_params['min_samples_leaf'] = int(round(best_rfr_params['min_samples_leaf']))
rfr_model = RandomForestRegressor(**best_rfr_params)
rfr_model.fit(X_train, y_train)
rfr_predictions = rfr_model.predict(X_test)
rfr_rmse = np.sqrt(mean_squared_error(y_test, rfr_predictions))
rfr_r2 = r2_score(y_test, rfr_predictions)
validation_rmse['RFR'] = rfr_rmse
validation_r2['RFR'] = rfr_r2

# XGBoost
best_xgb_params = best_params['XGB']
best_xgb_params['n_estimators'] = int(round(best_xgb_params['n_estimators']))
best_xgb_params['max_depth'] = int(round(best_xgb_params['max_depth']))
xgb_model = XGBRegressor(**best_xgb_params)
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))
xgb_r2 = r2_score(y_test, xgb_predictions)
validation_rmse['XGB'] = xgb_rmse
validation_r2['XGB'] = xgb_r2

# # LightGBM
best_lgbm_params = best_params['LGBM']
best_lgbm_params['n_estimators'] = int(round(best_lgbm_params['n_estimators']))
best_lgbm_params['max_depth'] = int(round(best_lgbm_params['max_depth']))
best_lgbm_params['min_child_samples'] = int(round(best_lgbm_params['min_child_samples']))
lgbm_model = LGBMRegressor(**best_lgbm_params)
lgbm_model.fit(X_train, y_train)
lgbm_predictions = lgbm_model.predict(X_test)
lgbm_rmse = np.sqrt(mean_squared_error(y_test, lgbm_predictions))
lgbm_r2 = r2_score(y_test, lgbm_predictions)
validation_rmse['LGBM'] = lgbm_rmse
validation_r2['LGBM'] = lgbm_r2

# Extra Trees Regressor
best_etr_params = best_params['ETR']
best_etr_params['n_estimators'] = int(round(best_etr_params['n_estimators']))
best_etr_params['max_depth'] = int(round(best_etr_params['max_depth']))
best_etr_params['min_samples_split'] = int(round(best_etr_params['min_samples_split']))
best_etr_params['min_samples_leaf'] = int(round(best_etr_params['min_samples_leaf']))
best_etr_params['max_leaf_nodes'] = int(round(best_etr_params['max_leaf_nodes']))
etr_model = ExtraTreesRegressor(**best_etr_params)
etr_model.fit(X_train, y_train)
etr_predictions = etr_model.predict(X_test)
etr_rmse = np.sqrt(mean_squared_error(y_test, etr_predictions))
etr_r2 = r2_score(y_test, etr_predictions)
validation_rmse['ETR'] = etr_rmse
validation_r2['ETR'] = etr_r2

# CatBoost
best_cb_params = best_params['CBR']
best_cb_params['iterations'] = int(round(best_cb_params['iterations']))
best_cb_params['depth'] = int(round(best_cb_params['depth']))
cb_model = CatBoostRegressor(**best_cb_params, verbose=False)
cb_model.fit(X_train, y_train)
cb_predictions = cb_model.predict(X_test)
cb_rmse = np.sqrt(mean_squared_error(y_test, cb_predictions))
cb_r2 = r2_score(y_test, cb_predictions)
validation_rmse['CB'] = cb_rmse
validation_r2['CB'] = cb_r2
# %%
import matplotlib.pyplot as plt

print([rfr_r2, xgb_r2, lgbm_r2, etr_r2, cb_r2])

mse_values = [rfr_r2, xgb_r2, lgbm_r2, etr_r2, cb_r2]
model_names = ['RandomForest', 'XGBoost', 'LightGBM', 'ExtraTrees', 'CatBoost']


# Plotting the MSE values for each model
plt.figure(figsize=(12, 6))
plt.barh(model_names, mse_values, color='green')
plt.xlabel('R2-scores')
plt.ylabel('Models')
plt.title('R2-scores of Different Models on y_test')
plt.grid(axis='x')

# Annotate each bar with the specific MSE value
for i, v in enumerate(mse_values):
    plt.text(v, i, " {:.2f}".format(v), va='center', color='black')

plt.show()
# %%
from sklearn.metrics import mean_squared_error, r2_score

validation_rmse_train = {}
validation_r2_train = {}

# Random Forest
best_rfr_params = best_params['RF']
best_rfr_params['n_estimators'] = int(round(best_rfr_params['n_estimators']))
best_rfr_params['max_depth'] = int(round(best_rfr_params['max_depth']))
best_rfr_params['min_samples_split'] = int(round(best_rfr_params['min_samples_split']))
best_rfr_params['min_samples_leaf'] = int(round(best_rfr_params['min_samples_leaf']))
rfr_model = RandomForestRegressor(**best_rfr_params)
rfr_model.fit(X_train, y_train)
rfr_predictions_train = rfr_model.predict(X_train)
rfr_rmse_train = np.sqrt(mean_squared_error(y_train, rfr_predictions_train))
rfr_r2_train = r2_score(y_train, rfr_predictions_train)
validation_rmse_train['RFR'] = rfr_rmse_train
validation_r2_train['RFR'] = rfr_r2_train

# XGBoost
best_xgb_params = best_params['XGB']
best_xgb_params['n_estimators'] = int(round(best_xgb_params['n_estimators']))
best_xgb_params['max_depth'] = int(round(best_xgb_params['max_depth']))
xgb_model = XGBRegressor(**best_xgb_params)
xgb_model.fit(X_train, y_train)
xgb_predictions_train = xgb_model.predict(X_train)
xgb_rmse_train = np.sqrt(mean_squared_error(y_train, xgb_predictions_train))
xgb_r2_train = r2_score(y_train, xgb_predictions_train)
validation_rmse_train['XGB'] = xgb_rmse_train
validation_r2_train['XGB'] = xgb_r2_train

# # LightGBM
best_lgbm_params = best_params['LGBM']
best_lgbm_params['n_estimators'] = int(round(best_lgbm_params['n_estimators']))
best_lgbm_params['max_depth'] = int(round(best_lgbm_params['max_depth']))
best_lgbm_params['min_child_samples'] = int(round(best_lgbm_params['min_child_samples']))
lgbm_model = LGBMRegressor(**best_lgbm_params)
lgbm_model.fit(X_train, y_train)
lgbm_predictions_train = lgbm_model.predict(X_train)
lgbm_rmse_train = np.sqrt(mean_squared_error(y_train, lgbm_predictions_train))
lgbm_r2_train = r2_score(y_train, lgbm_predictions_train)
validation_rmse_train['LGBM'] = lgbm_rmse_train
validation_r2_train['LGBM'] = lgbm_r2_train

# Extra Trees Regressor
best_etr_params = best_params['ETR']
best_etr_params['n_estimators'] = int(round(best_etr_params['n_estimators']))
best_etr_params['max_depth'] = int(round(best_etr_params['max_depth']))
best_etr_params['min_samples_split'] = int(round(best_etr_params['min_samples_split']))
best_etr_params['min_samples_leaf'] = int(round(best_etr_params['min_samples_leaf']))
best_etr_params['max_leaf_nodes'] = int(round(best_etr_params['max_leaf_nodes']))
etr_model = ExtraTreesRegressor(**best_etr_params)
etr_model.fit(X_train, y_train)
etr_predictions_train = etr_model.predict(X_train)
etr_rmse_train = np.sqrt(mean_squared_error(y_train, etr_predictions_train))
etr_r2_train = r2_score(y_train, etr_predictions_train)
validation_rmse_train['ETR'] = etr_rmse_train
validation_r2_train['ETR'] = etr_r2_train

# CatBoost
best_cb_params = best_params['CBR']
best_cb_params['iterations'] = int(round(best_cb_params['iterations']))
best_cb_params['depth'] = int(round(best_cb_params['depth']))
cb_model = CatBoostRegressor(**best_cb_params, verbose=False)
cb_model.fit(X_train, y_train)
cb_predictions_train = cb_model.predict(X_train)
cb_rmse_train = np.sqrt(mean_squared_error(y_train, cb_predictions_train))
cb_r2_train = r2_score(y_train, cb_predictions_train)
validation_rmse_train['CB'] = cb_rmse_train
validation_r2_train['CB'] = cb_r2_train
# %%
import matplotlib.pyplot as plt

print([rfr_r2_train, xgb_r2_train, lgbm_r2_train, etr_r2_train, cb_r2_train])

mse_values = [rfr_r2_train, xgb_r2_train, lgbm_r2_train, etr_r2_train, cb_r2_train]
model_names = ['RandomForest', 'XGBoost', 'LightGBM', 'ExtraTrees', 'CatBoost']


# Plotting the MSE values for each model
plt.figure(figsize=(12, 6))
plt.barh(model_names, mse_values, color='blue')
plt.xlabel('R2-scores')
plt.ylabel('Models')
plt.title('R2-scores of Different Models on y_train')
plt.grid(axis='x')

# Annotate each bar with the specific MSE value
for i, v in enumerate(mse_values):
    plt.text(v, i, " {:.2f}".format(v), va='center', color='black')

plt.show()
# %%
revised_data = pd.read_csv(r'C:\Users\OJH\OneDrive\SKKU\연구자료\Catalyst AI\New Database\Na_W_Mn_Cat_preprocess_top5.csv')

revised_data.drop(['Preparation'], axis=1, inplace=True)

revised_data.columns = [col.replace('(', '').replace(')', '').replace('/', '_per_').replace(',','') for col in revised_data.columns]

revised_data

# %%
# Interpolate data
num_points = 10
columns_to_interpolate = ['Temperature', 'pCH4_per_pO2', 'p total', 'Contact time']
interpolated_data = pd.DataFrame()

for col in columns_to_interpolate:
    min_val, max_val = revised_data[col].min(), revised_data[col].max()
    interpolated_data[col] = np.linspace(min_val, max_val, num_points)

# Fill other columns with mean values (you can choose a different strategy if needed)
for col in revised_data.columns:
    if col not in columns_to_interpolate + ['YC2']:
        interpolated_data[col] = revised_data[col].mean()

# Prepare features and target variable
X = revised_data.drop('YC2', axis=1)
y = revised_data['YC2']

# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

lgbm_model.fit(X_train, y_train)

# Predict on interpolated data
interpolated_data_scaled = scaler.transform(interpolated_data)
interpolated_data['YC2'] = lgbm_model.predict(interpolated_data_scaled)

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(interpolated_data['Temperature'], interpolated_data['YC2'], label='Predicted Y(C2)')
plt.xlabel('Temperature')
plt.ylabel('Predicted Y(C2)')
plt.title('Predicted Y(C2) for Interpolated Data')
plt.legend()
plt.show()

print(interpolated_data)
# %%
revised_data
# %%
interpolated_data
# %%
from itertools import product

# Generate 10 linearly spaced values for each of the four features
num_points = 10
features_to_interpolate = ['Temperature', 'pCH4_per_pO2', 'p total', 'Contact time']
interpolated_values = {feature: np.linspace(revised_data[feature].min(), revised_data[feature].max(), num_points) for feature in features_to_interpolate}

# Generate all combinations (Cartesian product)
all_combinations = list(product(*(interpolated_values[feature] for feature in features_to_interpolate)))

# Create a DataFrame from these combinations
interpolated_data = pd.DataFrame(all_combinations, columns=features_to_interpolate)

# Initialize 'YC2' to 0
interpolated_data['YC2'] = 0

# Concatenate the columns for scaling
data_for_scaling = pd.concat([revised_data[features_to_interpolate + ['YC2']], interpolated_data])

# Apply MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data_for_scaling)

# Create a DataFrame of the scaled data
scaled_data_df = pd.DataFrame(scaled_data, columns=features_to_interpolate + ['YC2'])

scaled_data_df.head()
# %%
scaled_data_df.drop('YC2', axis=1, inplace=True)
scaled_data_df
# %%
# Predict 'Y(C2)' for the interpolated data
scaled_data_df['Y(C2)_predicted'] = lgbm_model.predict(scaled_data_df)

scaled_data_df
# %%
# Post-process predictions to ensure they are non-negative
scaled_data_df['Y(C2)_predicted'] = scaled_data_df['Y(C2)_predicted'].clip(lower=0)
# %%
scaled_data_real = scaler.inverse_transform(scaled_data_df)

scaled_data_real = pd.DataFrame(scaled_data_real, columns=features_to_interpolate + ['Y(C2)_predicted'])
# %%
scaled_data_real
# %%
scaled_data_real.describe()
# %%
# Visualization
plt.figure(figsize=(10, 6))
plt.plot(scaled_data_real['Temperature'], scaled_data_real['Y(C2)_predicted'], label='Predicted Y(C2)')
plt.xlabel('Temperature')
plt.ylabel('Predicted Y(C2)')
plt.title('Predicted Y(C2) for Interpolated Data')
plt.legend()
plt.show()

print(interpolated_data)
# %%
scaled_data_real
# %%
scaled_data_real.to_csv(r'C:\Users\OJH\OneDrive\SKKU\연구자료\Catalyst AI\New Database\Na_W_Mn_Cat_preprocess_interpolation.csv')
# %%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Separating actual data and interpolated data
actual_data = scaled_data_real.iloc[:120]  # Actual data (first 120 rows)
interpolation_data = scaled_data_real.iloc[120:]  # Interpolated data (remaining rows)

# Define pairs of features for the plots
feature_pairs = [
    ('Temperature', 'p total'),
    ('Temperature', 'pCH4_per_pO2'),
    ('Temperature', 'Contact time'),
    ('pCH4_per_pO2', 'p total'),
    ('pCH4_per_pO2', 'Contact time'),
    ('p total', 'Contact time')
]

# Create 3D plots for each pair of features
fig = plt.figure(figsize=(18, 12))

for i, (feature1, feature2) in enumerate(feature_pairs, 1):
    ax = fig.add_subplot(2, 3, i, projection='3d')
    
    # Plot interpolated data as a surface
    ax.plot_trisurf(interpolation_data[feature1], interpolation_data[feature2], interpolation_data['Y(C2)_predicted'], alpha=0.5)

    # Plot actual data as red dots
    ax.scatter(actual_data[feature1], actual_data[feature2], actual_data['Y(C2)_predicted'], color='red')

    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)
    ax.set_zlabel('Predicted Y(C2)')
    ax.set_title(f'{feature1} vs {feature2}')

plt.tight_layout()
plt.show()

# %%
