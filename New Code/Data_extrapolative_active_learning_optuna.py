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
X_test.iloc[[12]]
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
from sklearn.ensemble import ExtraTreesRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from optuna import create_study, visualization,pruners, samplers
import warnings
import optuna
warnings.filterwarnings('ignore')

best_params = {}

def objective_knn(trial):
    n_neighbors = trial.suggest_int("n_neighbors", 1, 100)

    knn_model = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn_score = cross_val_score(knn_model, X_train, y_train, cv=10, scoring='neg_mean_squared_error').mean()
    return knn_score

study = optuna.create_study(sampler = samplers.QMCSampler(), direction = "maximize")
study.optimize(objective_knn, n_trials = 55)

trial = study.best_trial
print("Best Score of K-NN:", trial.value)
print("Best Params:")
for key, value in trial.params.items():
    print("  {}: {}".format(key, value))

best_params['K-NN'] = trial.params
# %%
def objective_svr(trial):
    C = trial.suggest_float("C", 0.1, 100)
    epsilon = trial.suggest_float("epsilon", 0.01, 1)
    gamma = trial.suggest_float("gamma", 0.0001, 1)

    svr_model = SVR(C=C, epsilon=epsilon, gamma=gamma)
    svr_score = cross_val_score(svr_model, X_train, y_train, cv=10, scoring='neg_mean_squared_error').mean()
    return svr_score

study = optuna.create_study(sampler = samplers.QMCSampler(), direction = "maximize")
study.optimize(objective_svr, n_trials = 550)

trial = study.best_trial
print("Best Score of SVR:", trial.value)
print("Best Params:")
for key, value in trial.params.items():
    print("  {}: {}".format(key, value))

best_params['SVR'] = trial.params
# %%
def objective_gpr(trial):
    alpha = trial.suggest_float("alpha", 1e-10, 1e-1)

    gpr_model = GaussianProcessRegressor(alpha=alpha)
    gpr_score = cross_val_score(gpr_model, X_train, y_train, cv=10, scoring='neg_mean_squared_error').mean()
    return gpr_score

study = optuna.create_study(sampler = samplers.QMCSampler(), direction = "maximize")
study.optimize(objective_gpr, n_trials = 550)

trial = study.best_trial
print("Best Score of GPR:", trial.value)
print("Best Params:")
for key, value in trial.params.items():
    print("  {}: {}".format(key, value))

best_params['GPR'] = trial.params
# %%
def objective_dtr(trial):
    max_depth = trial.suggest_int("max_depth", 1, 50)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

    dtr_model = DecisionTreeRegressor(max_depth=max_depth,
                                      min_samples_split=min_samples_split,
                                      min_samples_leaf=min_samples_leaf,
                                      random_state=42)
    dtr_score = cross_val_score(dtr_model, X_train, y_train, cv=10, scoring='neg_mean_squared_error').mean()
    return dtr_score

study = optuna.create_study(sampler = samplers.QMCSampler(), direction = "maximize")
study.optimize(objective_dtr, n_trials = 550)

trial = study.best_trial
print("Best Score of DTR:", trial.value)
print("Best Params:")
for key, value in trial.params.items():
    print("  {}: {}".format(key, value))

best_params['DTR'] = trial.params
# %%
def objective_rfr(trial):
    n_estimators = trial.suggest_int("n_estimators", 10, 1000)
    max_depth = trial.suggest_int("max_depth", 1, 50)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

    rfr_model = RandomForestRegressor(n_estimators=n_estimators,
                                      max_depth=max_depth,
                                      min_samples_split=min_samples_split,
                                      min_samples_leaf=min_samples_leaf,
                                      random_state=42)
    rfr_score = cross_val_score(rfr_model, X_train, y_train, cv=10, scoring='neg_mean_squared_error').mean()
    return rfr_score

study = optuna.create_study(sampler = samplers.QMCSampler(), direction = "maximize")
study.optimize(objective_rfr, n_trials = 550)

trial = study.best_trial
print("Best Score of RFR:", trial.value)
print("Best Params:")
for key, value in trial.params.items():
    print("  {}: {}".format(key, value))

best_params['RFR'] = trial.params
# %%
def objective_xgb(trial):
    n_estimators = trial.suggest_int("n_estimators", 10, 1000)
    max_depth = trial.suggest_int("max_depth", 1, 50)
    min_child_weight = trial.suggest_int("min_child_weight", 1, 10)
    gamma = trial.suggest_float("gamma", 0, 10)
    subsample = trial.suggest_float("subsample", 0.5, 1)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1)
    learning_rate = trial.suggest_float("learning_rate", 0.0001, 1)

    xgb_model = XGBRegressor(n_estimators=n_estimators,
                             max_depth=max_depth,
                             min_child_weight=min_child_weight,
                             gamma=gamma,
                             subsample=subsample,
                             colsample_bytree=colsample_bytree,
                             learning_rate=learning_rate,
                             random_state=42)
    xgb_score = cross_val_score(xgb_model, X_train, y_train, cv=10, scoring='neg_mean_squared_error').mean()
    return xgb_score

study = optuna.create_study(sampler = samplers.QMCSampler(), direction = "maximize")
study.optimize(objective_xgb, n_trials = 550)

trial = study.best_trial
print("Best Score of XGB:", trial.value)
print("Best Params:")
for key, value in trial.params.items():
    print("  {}: {}".format(key, value))

best_params['XGB'] = trial.params
# %%
warnings.filterwarnings('ignore')

def objective_lgbm(trial):
    n_estimators = trial.suggest_int("n_estimators", 10, 1000)
    max_depth = trial.suggest_int("max_depth", 1, 50)
    min_child_weight = trial.suggest_int("min_child_weight", 1, 10)
    min_child_samples = trial.suggest_int("min_child_weight", 1, 10)
    subsample = trial.suggest_float("subsample", 0.5, 1)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1)
    learning_rate = trial.suggest_float("learning_rate", 0.0001, 1)

    lgbm_model = LGBMRegressor(n_estimators=n_estimators,
                               max_depth=max_depth,
                               min_child_weight=min_child_weight,
                               min_child_samples=min_child_samples,
                               subsample=subsample,
                               colsample_bytree=colsample_bytree,
                               learning_rate=learning_rate,
                               verbose_eval=False,
                               random_state=42)
    
    lgbm_score = cross_val_score(lgbm_model, X_train, y_train, cv=10, scoring='neg_mean_squared_error').mean()
    return lgbm_score

study = optuna.create_study(sampler = samplers.QMCSampler(), direction = "maximize")
study.optimize(objective_lgbm, n_trials = 550)

trial = study.best_trial
print("Best Score of LGBM:", trial.value)
print("Best Params:")
for key, value in trial.params.items():
    print("  {}: {}".format(key, value))

best_params['LGBM'] = trial.params
# %%
from sklearn.ensemble import ExtraTreesRegressor

def objective_etr(trial):
    n_estimators = trial.suggest_int("n_estimators", 10, 1000)
    max_depth = trial.suggest_int("max_depth", 1, 50)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    max_features = trial.suggest_float("max_features", 0.5, 1)
    max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 2, 10)
    min_impurity_decrease = trial.suggest_float("min_impurity_decrease", 0, 0.1)

    etr_model = ExtraTreesRegressor(n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf,
                                    max_features=max_features,
                                    max_leaf_nodes=max_leaf_nodes,
                                    min_impurity_decrease=min_impurity_decrease,
                                    random_state=42)
    
    etr_score = cross_val_score(etr_model, X_train, y_train, cv=10, scoring='neg_mean_squared_error').mean()
    return etr_score

study = optuna.create_study(sampler = samplers.QMCSampler(), direction = "maximize")
study.optimize(objective_etr, n_trials = 550)

trial = study.best_trial
print("Best Score of ETR:", trial.value)
print("Best Params:")
for key, value in trial.params.items():
    print("  {}: {}".format(key, value))

best_params['ETR'] = trial.params
# %%
from catboost import CatBoostRegressor

def objective_cbr(trial):
    iterations = trial.suggest_int("iterations", 1, 100)
    learning_rate = trial.suggest_float("learning_rate", 0.001, 1)
    depth = trial.suggest_int("depth", 1, 10)
    l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1, 10)

    cbr_model = CatBoostRegressor(iterations=iterations,
                                  learning_rate=learning_rate,
                                  depth=depth,
                                  l2_leaf_reg=l2_leaf_reg,
                                  verbose=False,
                                  random_state=42)
    
    cbr_score = cross_val_score(cbr_model, X_train, y_train, cv=10, scoring='neg_mean_squared_error').mean()

    return cbr_score

study = optuna.create_study(sampler = samplers.QMCSampler(), direction = "maximize")
study.optimize(objective_cbr, n_trials = 550)

trial = study.best_trial
print("Best Score of CBR:", trial.value)
print("Best Params:")
for key, value in trial.params.items():
    print("  {}: {}".format(key, value))

best_params['CBR'] = trial.params
# %%
def objective_mlp(trial):
    hidden_layer_sizes = trial.suggest_int("hidden_layer_sizes", 10, 100)
    alpha = trial.suggest_float("alpha", 0.0001, 1)
    batch_size = trial.suggest_int("batch_size", 1, 400)
    learning_rate_init = trial.suggest_float("learning_rate_init", 0.0001, 1)

    mlp_model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                             activation='relu',
                             solver='adam',
                             alpha=alpha,
                             batch_size=batch_size,
                             learning_rate='constant',
                             learning_rate_init=learning_rate_init,
                             max_iter=1000,
                             random_state=42)
    
    mlp_score = cross_val_score(mlp_model, X_train, y_train, cv=10, scoring='neg_mean_squared_error').mean()

    return mlp_score

study = optuna.create_study(sampler = samplers.QMCSampler(), direction = "maximize")
study.optimize(objective_mlp, n_trials = 550)

trial = study.best_trial
print("Best Score of MLP:", trial.value)
print("Best Params:")
for key, value in trial.params.items():
    print("  {}: {}".format(key, value))

best_params['MLP'] = trial.params
# %%
from sklearn.metrics import r2_score

validation_rmse = {}
validation_r2 = {}

# K-Nearest Neighbors
knn_predictions_list = []
best_knn_params = best_params['K-NN']
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
best_rfr_params = best_params['RFR']
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
predictions_df.to_csv('predictions_QMC.csv')
# %%