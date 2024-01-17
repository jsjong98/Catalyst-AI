# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_set_path = r'C:\Users\OJH\OneDrive\SKKU\연구자료\Catalyst AI\New Database\Na_W_Mn_train.csv'
train_set = pd.read_csv(train_set_path)

train_set.drop(['Unnamed: 0'], axis=1, inplace=True)

train_set
# %%
test_set_path = r'C:\Users\OJH\OneDrive\SKKU\연구자료\Catalyst AI\New Database\Na_W_Mn_test.csv'
test_set = pd.read_csv(test_set_path)

test_set.drop(['Unnamed: 0'], axis=1, inplace=True)

test_set
# %%
from sklearn.preprocessing import MinMaxScaler

input_columns = ['Temperature', 'pCH4_per_pO2', 'Contact time']
output_column = ['Y(C2)_predicted']

X_train = train_set[input_columns]
y_train = train_set[output_column]
X_test = test_set[input_columns]
y_test = test_set[output_column]
# %%
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_train_scaled = y_train_scaled.ravel()

y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))
y_test_scaled = y_test_scaled.ravel()
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
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

best_params = {}

def cv_score_rfr(n_estimators, max_depth, min_samples_split, min_samples_leaf):
    model = RandomForestRegressor(n_estimators=int(n_estimators), 
                                  max_depth=int(max_depth), 
                                  min_samples_split=int(min_samples_split), 
                                  min_samples_leaf=int(min_samples_leaf), 
                                  random_state=42
    )
    
    scores = cross_val_score(model, X_train_scaled, y_train_scaled, cv=5, scoring='neg_mean_squared_error')
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
    
    scores = cross_val_score(model, X_train_scaled, y_train_scaled, cv=5, scoring='neg_mean_squared_error')
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
    
    scores = cross_val_score(model, X_train_scaled, y_train_scaled, cv=5, scoring='neg_mean_squared_error')
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
    
    scores = cross_val_score(model, X_train_scaled, y_train_scaled, cv=5, scoring='neg_mean_squared_error')
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
    
    scores = cross_val_score(model, X_train_scaled, y_train_scaled, cv=5, scoring='neg_mean_squared_error')
    return np.mean(scores)

pbounds = {"iterations": (1, 100), 
           "learning_rate": (0.001, 0.1), 
           "depth": (1, 10), 
           "l2_leaf_reg": (1, 10)}

optimizer = BayesianOptimization(f=cv_score_cbr, pbounds=pbounds, random_state=42)
optimizer.maximize(init_points=50, n_iter=500)
best_params['CBR'] = optimizer.max['params']

# %%
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow import keras
from scikeras.wrappers import KerasRegressor


def create_model(learning_rate, hidden_units_1, hidden_units_2, hidden_units_3, hidden_units_4, hidden_units_5):
    model = Sequential()
    model.add(Dense(units=int(hidden_units_1), activation='relu', input_dim=3))
    model.add(Dense(units=int(hidden_units_2), activation='relu'))
    model.add(Dense(units=int(hidden_units_3), activation='relu'))
    model.add(Dense(units=int(hidden_units_4), activation='relu'))
    model.add(Dense(units=int(hidden_units_5), activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
    return model

def cv_score_mlp(learning_rate, hidden_units_1, hidden_units_2, hidden_units_3, hidden_units_4, hidden_units_5):
    model = KerasRegressor(model=create_model, learning_rate=learning_rate,
                           hidden_units_1=hidden_units_1, hidden_units_2=hidden_units_2,
                           hidden_units_3=hidden_units_3, hidden_units_4=hidden_units_4,
                           hidden_units_5=hidden_units_5, epochs=100, batch_size=10, verbose=0)
    scores = cross_val_score(model, X_train_scaled, y_train_scaled, cv=5, scoring='neg_mean_squared_error')
    return np.mean(scores)

pbounds = {'learning_rate': (0.0001, 0.1),
           'hidden_units_1': (1, 100),
           'hidden_units_2': (1, 100),
           'hidden_units_3': (1, 100),
           'hidden_units_4': (1, 100),
           'hidden_units_5': (1, 100)}

optimizer = BayesianOptimization(f=cv_score_mlp, pbounds=pbounds, random_state=42)
optimizer.maximize(init_points=50, n_iter=500)

best_params['MLP'] = optimizer.max['params']
# %%
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold

validation_rmse = {}
validation_r2 = {}

# Random Forest
best_rfr_params = best_params['RF']
best_rfr_params['n_estimators'] = int(round(best_rfr_params['n_estimators']))
best_rfr_params['max_depth'] = int(round(best_rfr_params['max_depth']))
best_rfr_params['min_samples_split'] = int(round(best_rfr_params['min_samples_split']))
best_rfr_params['min_samples_leaf'] = int(round(best_rfr_params['min_samples_leaf']))
rfr_model = RandomForestRegressor(**best_rfr_params)
rfr_model.fit(X_train_scaled, y_train_scaled)
rfr_predictions = rfr_model.predict(X_test_scaled)
rfr_rmse = np.sqrt(mean_squared_error(y_test_scaled, rfr_predictions))
rfr_r2 = r2_score(y_test_scaled, rfr_predictions)
validation_rmse['RFR'] = rfr_rmse
validation_r2['RFR'] = rfr_r2

# XGBoost
best_xgb_params = best_params['XGB']
best_xgb_params['n_estimators'] = int(round(best_xgb_params['n_estimators']))
best_xgb_params['max_depth'] = int(round(best_xgb_params['max_depth']))
xgb_model = XGBRegressor(**best_xgb_params)
xgb_model.fit(X_train_scaled, y_train_scaled)
xgb_predictions = xgb_model.predict(X_test_scaled)
xgb_rmse = np.sqrt(mean_squared_error(y_test_scaled, xgb_predictions))
xgb_r2 = r2_score(y_test_scaled, xgb_predictions)
validation_rmse['XGB'] = xgb_rmse
validation_r2['XGB'] = xgb_r2

# # LightGBM
best_lgbm_params = best_params['LGBM']
best_lgbm_params['n_estimators'] = int(round(best_lgbm_params['n_estimators']))
best_lgbm_params['max_depth'] = int(round(best_lgbm_params['max_depth']))
best_lgbm_params['min_child_samples'] = int(round(best_lgbm_params['min_child_samples']))
lgbm_model = LGBMRegressor(**best_lgbm_params)
lgbm_model.fit(X_train_scaled, y_train_scaled)
lgbm_predictions = lgbm_model.predict(X_test_scaled)
lgbm_rmse = np.sqrt(mean_squared_error(y_test_scaled, lgbm_predictions))
lgbm_r2 = r2_score(y_test_scaled, lgbm_predictions)
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
etr_model.fit(X_train_scaled, y_train_scaled)
etr_predictions = etr_model.predict(X_test_scaled)
etr_rmse = np.sqrt(mean_squared_error(y_test_scaled, etr_predictions))
etr_r2 = r2_score(y_test_scaled, etr_predictions)
validation_rmse['ETR'] = etr_rmse
validation_r2['ETR'] = etr_r2

# CatBoost
best_cb_params = best_params['CBR']
best_cb_params['iterations'] = int(round(best_cb_params['iterations']))
best_cb_params['depth'] = int(round(best_cb_params['depth']))
cb_model = CatBoostRegressor(**best_cb_params, verbose=False)
cb_model.fit(X_train_scaled, y_train_scaled)
cb_predictions = cb_model.predict(X_test_scaled)
cb_rmse = np.sqrt(mean_squared_error(y_test_scaled, cb_predictions))
cb_r2 = r2_score(y_test_scaled, cb_predictions)
validation_rmse['CB'] = cb_rmse
validation_r2['CB'] = cb_r2

# MLP
best_mlp_params = best_params['MLP']
best_mlp_params['hidden_units_1'] = int(round(best_mlp_params['hidden_units_1']))
best_mlp_params['hidden_units_2'] = int(round(best_mlp_params['hidden_units_2']))
best_mlp_params['hidden_units_3'] = int(round(best_mlp_params['hidden_units_3']))
best_mlp_params['hidden_units_4'] = int(round(best_mlp_params['hidden_units_4']))
best_mlp_params['hidden_units_5'] = int(round(best_mlp_params['hidden_units_5']))

mlp_model = Sequential()
mlp_model.add(Dense(units=best_mlp_params['hidden_units_1'], activation='relu', input_dim=3))
mlp_model.add(Dense(units=best_mlp_params['hidden_units_2'], activation='relu'))
mlp_model.add(Dense(units=best_mlp_params['hidden_units_3'], activation='relu'))
mlp_model.add(Dense(units=best_mlp_params['hidden_units_4'], activation='relu'))
mlp_model.add(Dense(units=best_mlp_params['hidden_units_5'], activation='relu'))
mlp_model.add(Dense(units=1, activation='sigmoid'))
mlp_model.compile(loss='mse', optimizer=Adam(learning_rate=best_mlp_params['learning_rate']))

mlp_predictions = mlp_model.predict(X_test_scaled)
mlp_rmse = np.sqrt(mean_squared_error(y_test_scaled, mlp_predictions))
mlp_r2 = r2_score(y_test_scaled, mlp_predictions)

validation_rmse['MLP'] = mlp_rmse
validation_r2['MLP'] = mlp_r2
# %%
import matplotlib.pyplot as plt

print([rfr_r2, xgb_r2, lgbm_r2, etr_r2, cb_r2, mlp_r2])

mse_values = [rfr_r2, xgb_r2, lgbm_r2, etr_r2, cb_r2, mlp_r2]
model_names = ['RandomForest', 'XGBoost', 'LightGBM', 'ExtraTrees', 'CatBoost', 'MLP']

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
