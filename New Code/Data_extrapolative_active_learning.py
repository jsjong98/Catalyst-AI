# %%
import pandas as pd

# Load the provided datasets
train_df = pd.read_csv(r'C:\Users\OJH\OneDrive\SKKU\연구자료\Catalyst AI_rev\New Database\Na_W_Mn_Cat_train_rev.csv')
test_df = pd.read_csv(r'C:\Users\OJH\OneDrive\SKKU\연구자료\Catalyst AI_rev\New Database\Na_W_Mn_Cat_test_rev.csv')

# Display the first few rows of each dataset for an overview
train_df.head(), test_df.head()
# %%
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
test_df[test_df['Extrapolation Strength'] == 3].to_csv('extrapolative_3_rev.csv')
# %%
import matplotlib.pyplot as plt

data = pd.read_csv('extrapolative_3_rev.csv')

# Assuming 'Temperature', 'pCH4_per_pO2', and 'Contact time' are your variables
# and 'MAE', 'RMSE', 'R2' are the metrics you want to plot

fig = plt.figure(figsize=(12, 8))

# For MAE
ax = fig.add_subplot(131, projection='3d') # 1 row, 3 cols, 1st subplot
ax.scatter(data['Temperature'], data['pCH4_per_pO2'], data['Y(C2)_predicted'], c='r', marker='o')
ax.set_xlabel('Temperature')
ax.set_ylabel('pCH4_per_pO2')
ax.set_zlabel('Y(C2)')
ax.title.set_text('3D Plot for Y(C2) (Temp & Feed ratio)')

# For RMSE
ax = fig.add_subplot(132, projection='3d') # 1 row, 3 cols, 2nd subplot
ax.scatter(data['Temperature'], data['Contact time'], data['Y(C2)_predicted'], c='b', marker='o')
ax.set_xlabel('Temperature')
ax.set_ylabel('pCH4_per_pO2')
ax.set_zlabel('Y(C2)')
ax.title.set_text('3D Plot for Y(C2) (Temp & Contact time)')

# For R2
ax = fig.add_subplot(133, projection='3d') # 1 row, 3 cols, 3rd subplot
ax.scatter(data['Contact time'], data['pCH4_per_pO2'], data['Y(C2)_predicted'], c='g', marker='o')
ax.set_xlabel('Temperature')
ax.set_ylabel('pCH4_per_pO2')
ax.set_zlabel('Y(C2)')
ax.title.set_text('3D Plot for Y(C2) (Contact time & Feed ratio)')

plt.tight_layout()
plt.show()
# %%
X_train = pd.concat([X_train, X_test.iloc[[12]]])
y_train = pd.concat([y_train, y_test.iloc[[12]]])
X_test = X_test.drop(12)
y_test = y_test.drop(12)

# %%
from sklearn.model_selection import cross_val_score
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
from bayes_opt import BayesianOptimization
warnings.filterwarnings('ignore')

best_params = {}

def cv_score_KNN(n_neighbors):
    model = KNeighborsRegressor(n_neighbors=int(n_neighbors))
    scores = cross_val_score(model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
    return np.mean(scores)

pbounds = {
    'n_neighbors': (1, 100)
}

optimizer = BayesianOptimization(f=cv_score_KNN, pbounds=pbounds, random_state=42)
optimizer.maximize(init_points=5, n_iter=50)
best_params['KNN'] = optimizer.max['params']

def cv_score_SVR (C, epsilon, gamma):
    model = SVR(C=C, epsilon=epsilon, gamma=gamma)
    scores = cross_val_score(model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
    return np.mean(scores)

pbounds = {
    'C': (0.1, 100),
    'epsilon': (0.01, 1),
    'gamma': (0.0001, 1)
}

optimizer = BayesianOptimization(f=cv_score_SVR, pbounds=pbounds, random_state=42)
optimizer.maximize(init_points=50, n_iter=500)
best_params['SVR'] = optimizer.max['params']

def cv_score_GPR (alpha):
    model = GaussianProcessRegressor(alpha=alpha)
    scores = cross_val_score(model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
    return np.mean(scores)

pbounds = {
    'alpha': (1e-10, 1e-1)
}

optimizer = BayesianOptimization(f=cv_score_GPR, pbounds=pbounds, random_state=42)
optimizer.maximize(init_points=10, n_iter=100)
best_params['GPR'] = optimizer.max['params']

def cv_score_DTR(max_depth, min_samples_split, min_samples_leaf):
    model = DecisionTreeRegressor(max_depth=int(max_depth), 
                                  min_samples_split=int(min_samples_split), 
                                  min_samples_leaf=int(min_samples_leaf), 
                                  random_state=42
    )
    
    scores = cross_val_score(model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
    return np.mean(scores)

pbounds = {
    'max_depth': (1, 50),
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 10)
}

optimizer = BayesianOptimization(f=cv_score_DTR, pbounds=pbounds, random_state=42)
optimizer.maximize(init_points=50, n_iter=500)
best_params['DTR'] = optimizer.max['params']

def cv_score_rfr(n_estimators, max_depth, min_samples_split, min_samples_leaf):
    model = RandomForestRegressor(n_estimators=int(n_estimators), 
                                  max_depth=int(max_depth), 
                                  min_samples_split=int(min_samples_split), 
                                  min_samples_leaf=int(min_samples_leaf), 
                                  random_state=42
    )
    
    scores = cross_val_score(model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
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
    
    scores = cross_val_score(model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
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
    
    scores = cross_val_score(model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
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
    
    scores = cross_val_score(model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
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
    
    scores = cross_val_score(model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
    return np.mean(scores)

pbounds = {"iterations": (1, 100), 
           "learning_rate": (0.001, 0.1), 
           "depth": (1, 10), 
           "l2_leaf_reg": (1, 10)}

optimizer = BayesianOptimization(f=cv_score_cbr, pbounds=pbounds, random_state=42)
optimizer.maximize(init_points=50, n_iter=500)
best_params['CBR'] = optimizer.max['params']

def cv_score_mlp(hidden_layer_sizes, alpha, batch_size, learning_rate_init):
    model = MLPRegressor(hidden_layer_sizes=(int(hidden_layer_sizes)), 
                         activation='relu',
                         solver='adam',
                         alpha=alpha,
                         batch_size=(int(batch_size)),
                         learning_rate='constant',
                         learning_rate_init=learning_rate_init,
                         max_iter=1000,
                         random_state=42
    )
    
    scores = cross_val_score(model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
    return np.mean(scores)

pbounds = {
    'hidden_layer_sizes': (10, 100),
    'alpha': (0.0001, 0.1),
    'batch_size': (1, 400),
    'learning_rate_init': (0.0001, 0.1),
}

optimizer = BayesianOptimization(f=cv_score_mlp, pbounds=pbounds, random_state=42)
optimizer.maximize(init_points=50, n_iter=500)
best_params['MLP'] = optimizer.max['params']
# %%
from sklearn.metrics import r2_score

validation_rmse = {}
validation_r2 = {}

# K-Nearest Neighbors
knn_predictions_list = []
best_knn_params = best_params['KNN']
best_knn_params['n_neighbors'] = int(round(best_knn_params['n_neighbors']))
knn_model = KNeighborsRegressor(**best_knn_params)
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)
knn_predictions_list.append(knn_predictions)
knn_rmse = np.sqrt(mean_squared_error(y_test, knn_predictions))
knn_r2 = r2_score(y_test, knn_predictions)
validation_rmse['KNN'] = knn_rmse
validation_r2['KNN'] = knn_r2

# Support Vector Regressor
svr_predictions_list = []
best_svr_params = best_params['SVR']
svr_model = SVR(**best_svr_params)
svr_model.fit(X_train, y_train)
svr_predictions = svr_model.predict(X_test)
svr_predictions_list.append(svr_predictions)
svr_rmse = np.sqrt(mean_squared_error(y_test, svr_predictions))
svr_r2 = r2_score(y_test, svr_predictions)
validation_rmse['SVR'] = svr_rmse
validation_r2['SVR'] = svr_r2

# Gaussian Process Regressor
gpr_predictions_list = []
best_gpr_params = best_params['GPR']
gpr_model = GaussianProcessRegressor(**best_gpr_params)
gpr_model.fit(X_train, y_train)
gpr_predictions = gpr_model.predict(X_test)
gpr_predictions_list.append(gpr_predictions)
gpr_rmse = np.sqrt(mean_squared_error(y_test, gpr_predictions))
gpr_r2 = r2_score(y_test, gpr_predictions)
validation_rmse['GPR'] = gpr_rmse
validation_r2['GPR'] = gpr_r2

# Decision Tree Regressor
dtr_predictions_list = []
best_dtr_params = best_params['DTR']
best_dtr_params['max_depth'] = int(round(best_dtr_params['max_depth']))
best_dtr_params['min_samples_split'] = int(round(best_dtr_params['min_samples_split']))
best_dtr_params['min_samples_leaf'] = int(round(best_dtr_params['min_samples_leaf']))
dtr_model = DecisionTreeRegressor(**best_dtr_params)
dtr_model.fit(X_train, y_train)
dtr_predictions = dtr_model.predict(X_test)
dtr_predictions_list.append(dtr_predictions)
dtr_rmse = np.sqrt(mean_squared_error(y_test, dtr_predictions))
dtr_r2 = r2_score(y_test, dtr_predictions)
validation_rmse['DTR'] = dtr_rmse
validation_r2['DTR'] = dtr_r2

# Random Forest
rfr_predictions_list = []
best_rfr_params = best_params['RF']
best_rfr_params['n_estimators'] = int(round(best_rfr_params['n_estimators']))
best_rfr_params['max_depth'] = int(round(best_rfr_params['max_depth']))
best_rfr_params['min_samples_split'] = int(round(best_rfr_params['min_samples_split']))
best_rfr_params['min_samples_leaf'] = int(round(best_rfr_params['min_samples_leaf']))
rfr_model = RandomForestRegressor(**best_rfr_params)
rfr_model.fit(X_train, y_train)
rfr_predictions = rfr_model.predict(X_test)
rfr_predictions_list.append(rfr_predictions)
rfr_rmse = np.sqrt(mean_squared_error(y_test, rfr_predictions))
rfr_r2 = r2_score(y_test, rfr_predictions)
validation_rmse['RFR'] = rfr_rmse
validation_r2['RFR'] = rfr_r2

# XGBoost
xgb_predictions_list = []
best_xgb_params = best_params['XGB']
best_xgb_params['n_estimators'] = int(round(best_xgb_params['n_estimators']))
best_xgb_params['max_depth'] = int(round(best_xgb_params['max_depth']))
xgb_model = XGBRegressor(**best_xgb_params)
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)
xgb_predictions_list.append(xgb_predictions)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))
xgb_r2 = r2_score(y_test, xgb_predictions)
validation_rmse['XGB'] = xgb_rmse
validation_r2['XGB'] = xgb_r2

# # LightGBM
lgbm_predictions_list = []
best_lgbm_params = best_params['LGBM']
best_lgbm_params['n_estimators'] = int(round(best_lgbm_params['n_estimators']))
best_lgbm_params['max_depth'] = int(round(best_lgbm_params['max_depth']))
best_lgbm_params['min_child_samples'] = int(round(best_lgbm_params['min_child_samples']))
lgbm_model = LGBMRegressor(**best_lgbm_params)
lgbm_model.fit(X_train, y_train)
lgbm_predictions = lgbm_model.predict(X_test)
lgbm_predictions_list.append(lgbm_predictions)
lgbm_rmse = np.sqrt(mean_squared_error(y_test, lgbm_predictions))
lgbm_r2 = r2_score(y_test, lgbm_predictions)
validation_rmse['LGBM'] = lgbm_rmse
validation_r2['LGBM'] = lgbm_r2

# Extra Trees Regressor
etr_predictions_list = []
best_etr_params = best_params['ETR']
best_etr_params['n_estimators'] = int(round(best_etr_params['n_estimators']))
best_etr_params['max_depth'] = int(round(best_etr_params['max_depth']))
best_etr_params['min_samples_split'] = int(round(best_etr_params['min_samples_split']))
best_etr_params['min_samples_leaf'] = int(round(best_etr_params['min_samples_leaf']))
best_etr_params['max_leaf_nodes'] = int(round(best_etr_params['max_leaf_nodes']))
etr_model = ExtraTreesRegressor(**best_etr_params)
etr_model.fit(X_train, y_train)
etr_predictions = etr_model.predict(X_test)
etr_predictions_list.append(etr_predictions)
etr_rmse = np.sqrt(mean_squared_error(y_test, etr_predictions))
etr_r2 = r2_score(y_test, etr_predictions)
validation_rmse['ETR'] = etr_rmse
validation_r2['ETR'] = etr_r2

# CatBoost
cb_predictions_list = []
best_cb_params = best_params['CBR']
best_cb_params['iterations'] = int(round(best_cb_params['iterations']))
best_cb_params['depth'] = int(round(best_cb_params['depth']))
cb_model = CatBoostRegressor(**best_cb_params, verbose=False)
cb_model.fit(X_train, y_train)
cb_predictions = cb_model.predict(X_test)
cb_predictions_list.append(cb_predictions)
cb_rmse = np.sqrt(mean_squared_error(y_test, cb_predictions))
cb_r2 = r2_score(y_test, cb_predictions)
validation_rmse['CB'] = cb_rmse
validation_r2['CB'] = cb_r2

# Multi-layer Perceptron
mlp_predictions_list = []
best_mlp_params = best_params['MLP']
best_mlp_params['hidden_layer_sizes'] = int(round(best_mlp_params['hidden_layer_sizes']))
best_mlp_params['batch_size'] = int(round(best_mlp_params['batch_size']))
mlp_model = MLPRegressor(hidden_layer_sizes=(best_mlp_params['hidden_layer_sizes']), 
                         activation='relu',
                         solver='adam',
                         alpha=best_mlp_params['alpha'],
                         batch_size=(best_mlp_params['batch_size']),
                         learning_rate='constant',
                         learning_rate_init=best_mlp_params['learning_rate_init'],
                         max_iter=1000,
                         random_state=42
    )
mlp_model.fit(X_train, y_train)
mlp_predictions = mlp_model.predict(X_test)
mlp_predictions_list.append(mlp_predictions)
mlp_rmse = np.sqrt(mean_squared_error(y_test, mlp_predictions))
mlp_r2 = r2_score(y_test, mlp_predictions)
validation_rmse['MLP'] = mlp_rmse
validation_r2['MLP'] = mlp_r2
# %%
import matplotlib.pyplot as plt

print([knn_r2, svr_r2, gpr_r2, dtr_r2, rfr_r2, xgb_r2, lgbm_r2, etr_r2, cb_r2, mlp_r2])

mse_values = [knn_r2, svr_r2, gpr_r2, dtr_r2, rfr_r2, xgb_r2, lgbm_r2, etr_r2, cb_r2, mlp_r2]
model_names = ['K-NN', 'SVR', 'GPR', 'DTR', 'RFR', 'XGB', 'LGBM', 'ETR', 'CB', 'MLP']


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
knn_predictions_df = pd.DataFrame(knn_predictions_list).T
svr_predictions_df = pd.DataFrame(svr_predictions_list).T
gpr_predictions_df = pd.DataFrame(gpr_predictions_list).T
dtr_predictions_df = pd.DataFrame(dtr_predictions_list).T
rfr_predictions_df = pd.DataFrame(rfr_predictions_list).T
xgb_predictions_df = pd.DataFrame(xgb_predictions_list).T
lgbm_predictions_df = pd.DataFrame(lgbm_predictions_list).T
etr_predictions_df = pd.DataFrame(etr_predictions_list).T
cb_predictions_df = pd.DataFrame(cb_predictions_list).T
mlp_predictions_df = pd.DataFrame(mlp_predictions_list).T
# %%
predictions_df = pd.concat([knn_predictions_df, svr_predictions_df, gpr_predictions_df, dtr_predictions_df, rfr_predictions_df, xgb_predictions_df, lgbm_predictions_df, etr_predictions_df, cb_predictions_df, mlp_predictions_df], axis=1)
predictions_df.columns = ['K-NN', 'SVR', 'GPR', 'DTR', 'RFR', 'XGB', 'LGBM', 'ETR', 'CB', 'MLP']
predictions_df
# %%
predictions_df.to_csv('predictions_2.csv')
# %%
