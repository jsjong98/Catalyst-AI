# %%
# Na 2.1 mol%, W 1.05 mol%, Mn 2.25 mol%, Si 94.6 mol% (Na) --> Na2WO4/SiO2 catalyst (OCM)
import pandas as pd
import numpy as np

revised_data = pd.read_csv(r'C:\Users\OJH\OneDrive\SKKU\연구자료\Catalyst AI\New Database\Na_W_Mn_Cat_rev.csv')

revised_data.drop(['Preparation'], axis=1, inplace=True)

revised_data
# %%
from scipy import stats

revised_data.columns = [col.replace('(', '').replace(')', '').replace('/', '_per_').replace(',','') for col in revised_data.columns]

column_names = revised_data.columns.tolist()

# Detect outliers based on Z-score
z_scores = np.abs(stats.zscore(revised_data.select_dtypes(include=np.number)))
outlier_threshold = 5
outliers = (z_scores > outlier_threshold).any(axis=1)

# Remove outliers
revised_data = revised_data[~outliers]

print(outliers)
print(revised_data)
# %%
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
revised_data = scaler.fit_transform(revised_data)

revised_data = pd.DataFrame(revised_data, columns=column_names)

from sklearn.model_selection import train_test_split

input_columns = [col for col in column_names if col not in ['YC2']]
output_column = 'YC2'

X = revised_data[input_columns]
y = revised_data[output_column]
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
    
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
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
    
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
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
    
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
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
    
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
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
    
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
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
rfr_model.fit(X, y)

# XGBoost
best_xgb_params = best_params['XGB']
best_xgb_params['n_estimators'] = int(round(best_xgb_params['n_estimators']))
best_xgb_params['max_depth'] = int(round(best_xgb_params['max_depth']))
xgb_model = XGBRegressor(**best_xgb_params)
xgb_model.fit(X, y)

# # LightGBM
best_lgbm_params = best_params['LGBM']
best_lgbm_params['n_estimators'] = int(round(best_lgbm_params['n_estimators']))
best_lgbm_params['max_depth'] = int(round(best_lgbm_params['max_depth']))
best_lgbm_params['min_child_samples'] = int(round(best_lgbm_params['min_child_samples']))
lgbm_model = LGBMRegressor(**best_lgbm_params)
lgbm_model.fit(X, y)

# Extra Trees Regressor
best_etr_params = best_params['ETR']
best_etr_params['n_estimators'] = int(round(best_etr_params['n_estimators']))
best_etr_params['max_depth'] = int(round(best_etr_params['max_depth']))
best_etr_params['min_samples_split'] = int(round(best_etr_params['min_samples_split']))
best_etr_params['min_samples_leaf'] = int(round(best_etr_params['min_samples_leaf']))
best_etr_params['max_leaf_nodes'] = int(round(best_etr_params['max_leaf_nodes']))
etr_model = ExtraTreesRegressor(**best_etr_params)
etr_model.fit(X, y)

# CatBoost
best_cb_params = best_params['CBR']
best_cb_params['iterations'] = int(round(best_cb_params['iterations']))
best_cb_params['depth'] = int(round(best_cb_params['depth']))
cb_model = CatBoostRegressor(**best_cb_params, verbose=False)
cb_model.fit(X, y)

# List of models
models = [rfr_model, xgb_model, lgbm_model, etr_model, cb_model]
model_names = ['RandomForest', 'XGBoost', 'LightGBM', 'ExtraTrees', 'CatBoost']

# K-Fold Cross-validation
kf = KFold(n_splits=5, random_state=42, shuffle=True)

# Store RMSE and R2 scores
rmse_scores = []
r2_scores = []

# Evaluate each model using cross-validation
for model in models:
    # RMSE scores
    neg_mse = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=kf)
    rmse_scores.append(np.sqrt(-neg_mse).mean())
    
    # R2 scores
    r2 = cross_val_score(model, X, y, scoring='r2', cv=kf)
    r2_scores.append(r2.mean())
# %%
import matplotlib.pyplot as plt

# Plotting RMSE scores
plt.figure(figsize=(12, 6))
plt.barh(model_names, rmse_scores, color='blue')
plt.xlabel('RMSE')
plt.ylabel('Models')
plt.title('RMSE of Different Models with K-Fold CV')
plt.grid(axis='x')
for i, v in enumerate(rmse_scores):
    plt.text(v, i, " {:.2f}".format(v), va='center', color='black')
plt.show()
# %%
# Plotting R2 scores
plt.figure(figsize=(12, 6))
plt.barh(model_names, r2_scores, color='green')
plt.xlabel('R2 Score')
plt.ylabel('Models')
plt.title('R2 Score of Different Models with K-Fold CV')
plt.grid(axis='x')
for i, v in enumerate(r2_scores):
    plt.text(v, i, " {:.2f}".format(v), va='center', color='black')
plt.show()
# %%
revised_data = pd.read_csv(r'C:\Users\OJH\OneDrive\SKKU\연구자료\Catalyst AI\New Database\Na_W_Mn_Cat_rev.csv')

revised_data.drop(['Preparation'], axis=1, inplace=True)

revised_data.columns = [col.replace('(', '').replace(')', '').replace('/', '_per_').replace(',','') for col in revised_data.columns]

# Remove outliers
revised_data = revised_data[~outliers]

revised_data
# %%
from itertools import product

# Generate 10 linearly spaced values for each of the four features
num_points = 21
features_to_interpolate = ['Temperature', 'pCH4_per_pO2', 'Contact time']
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
scaled_data_df['Y(C2)_predicted'] = xgb_model.predict(scaled_data_df)

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
actual_data = scaled_data_real.iloc[:116]  # Actual data (first 120 rows)
interpolation_data = scaled_data_real.iloc[116:]  # Interpolated data (remaining rows)

# Define pairs of features for the plots
feature_pairs = [
    ('Temperature', 'pCH4_per_pO2'),
    ('Temperature', 'Contact time'),
    ('pCH4_per_pO2', 'Contact time')
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
import plotly.graph_objects as go

# Example: 3D plot for 'Temperature', 'pCH4_per_pO2', and 'Predicted Y(C2)'
fig = go.Figure()

# Actual data points (first 120 rows) as scatter plot
fig.add_trace(go.Scatter3d(
    x=scaled_data_real['Temperature'][:116],
    y=scaled_data_real['pCH4_per_pO2'][:116],
    z=scaled_data_real['Y(C2)_predicted'][:116],
    mode='markers',
    marker=dict(color='red', size=2),
    name='Actual Data'
))

# Predicted data as a surface plot
fig.add_trace(go.Surface(
    x=scaled_data_real['Temperature'][116:],
    y=scaled_data_real['pCH4_per_pO2'][116:],
    z=scaled_data_real['Y(C2)_predicted'][116:],
    name='Predicted Surface'
))

# Set titles and labels
fig.update_layout(
    title='3D Plot of Predicted Y(C2)',
    scene = dict(
        xaxis_title='Temperature',
        yaxis_title='pCH4_per_pO2',
        zaxis_title='Predicted Y(C2)'
    )
)

fig.show()
# %%
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
file_path = r'C:\Users\OJH\OneDrive\SKKU\연구자료\Catalyst AI\New Database\Na_W_Mn_Cat_preprocess_interpolation.csv'  # Replace with the path to your file
data = pd.read_csv(file_path)

data.drop('Unnamed: 0', axis=1, inplace=True)

# Applying MinMax scaling
scaler = MinMaxScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
# %%
# Function to calculate the section index for each data point
def calculate_section_index(value, num_sections):
    return int(value * num_sections)

# Number of sections
num_sections = 20

# Adding section indices for each variable
scaled_data['Temperature_section'] = scaled_data['Temperature'].apply(lambda x: calculate_section_index(x, num_sections))
scaled_data['pCH4_per_pO2_section'] = scaled_data['pCH4_per_pO2'].apply(lambda x: calculate_section_index(x, num_sections))
scaled_data['Contact_time_section'] = scaled_data['Contact time'].apply(lambda x: calculate_section_index(x, num_sections))

# Defining the interpolation and extrapolation sections
interpolation_sections = range(3, 17)  # Sections 3 to 16 (inclusive) for interpolation
extrapolation_sections = list(set(range(num_sections)) - set(interpolation_sections))  # Remaining sections for extrapolation

# Mask for interpolation and extrapolation data
interpolation_mask = (
    scaled_data['Temperature_section'].isin(interpolation_sections) &
    scaled_data['pCH4_per_pO2_section'].isin(interpolation_sections) &
    scaled_data['Contact_time_section'].isin(interpolation_sections)
)
extrapolation_mask = ~interpolation_mask

# Splitting the data
interpolation_data = scaled_data[interpolation_mask]
extrapolation_data = scaled_data[extrapolation_mask]

# Displaying the number of data points in each set
interpolation_data_count = len(interpolation_data)
extrapolation_data_count = len(extrapolation_data)

print("Interpolation Data Count:", interpolation_data_count)
print("Extrapolation Data Count:", extrapolation_data_count)
# %%
# Counting the number of data points in each section for each variable
temp_section_counts = scaled_data['Temperature_section'].value_counts().sort_index()
pCH4_section_counts = scaled_data['pCH4_per_pO2_section'].value_counts().sort_index()
contact_time_section_counts = scaled_data['Contact_time_section'].value_counts().sort_index()

# Creating a DataFrame with section counts
section_counts = pd.DataFrame({
    'Temperature': temp_section_counts, 
    'pCH4_per_pO2': pCH4_section_counts, 
    'Contact time': contact_time_section_counts
})

# Display the section counts
print(section_counts)
# %%
# Function to calculate a combined section index for each data point based on three variables
def calculate_combined_section_index(row, num_sections):
    temp_index = int(row['Temperature'] * num_sections)
    pCH4_index = int(row['pCH4_per_pO2'] * num_sections)
    contact_time_index = int(row['Contact time'] * num_sections)
    return (temp_index, pCH4_index, contact_time_index)

# Number of sections
num_sections = 20

# Adding combined section indices
scaled_data['Combined_Section'] = scaled_data.apply(lambda row: calculate_combined_section_index(row, num_sections), axis=1)

# Counting the number of data points in each combined section
combined_section_counts = scaled_data['Combined_Section'].value_counts().sort_index()

# Display the counts
print(combined_section_counts.head())
# %%
# Function to calculate a combined section index for each data point based on three variables
# Define a function to calculate section indices for each variable
def calculate_section_index(value, num_sections):
    return int(value * num_sections)

# Number of sections
num_sections = 20

num_sections = 20  # Number of sections
min_section_count_threshold = 5  # Minimum count to consider a section high-frequency

# Calculate section indices for each variable
scaled_data['Temp_Section'] = scaled_data['Temperature'].apply(lambda x: calculate_section_index(x, num_sections))
scaled_data['pCH4_Section'] = scaled_data['pCH4_per_pO2'].apply(lambda x: calculate_section_index(x, num_sections))
scaled_data['Contact_Section'] = scaled_data['Contact time'].apply(lambda x: calculate_section_index(x, num_sections))

# Count data points in each section for each variable
temp_section_counts = scaled_data['Temp_Section'].value_counts()
pCH4_section_counts = scaled_data['pCH4_Section'].value_counts()
contact_section_counts = scaled_data['Contact_Section'].value_counts()

# Finding the sections with the highest counts that are continuous
def find_continuous_high_freq_sections(section_counts, min_count_threshold):
    continuous_sections = []
    temp_section = []
    sorted_sections = sorted(section_counts.items(), key=lambda x: x[0])

    for section, count in sorted_sections:
        if count >= min_count_threshold:
            if not temp_section or section == temp_section[-1] + 1:
                temp_section.append(section)
            else:
                continuous_sections.append(temp_section)
                temp_section = [section]
        else:
            if temp_section:
                continuous_sections.append(temp_section)
                temp_section = []

    if temp_section:
        continuous_sections.append(temp_section)

    return max(continuous_sections, key=len) if continuous_sections else []


# Finding continuous high-frequency sections for each variable
temp_continuous_high_freq = find_continuous_high_freq_sections(temp_section_counts, min_section_count_threshold)
pCH4_continuous_high_freq = find_continuous_high_freq_sections(pCH4_section_counts, min_section_count_threshold)
contact_time_continuous_high_freq = find_continuous_high_freq_sections(contact_time_section_counts, min_section_count_threshold)

# Displaying the high-frequency sections for each variable
temp_continuous_high_freq, pCH4_continuous_high_freq, contact_time_continuous_high_freq
# %%
interpolation_mask = (
    scaled_data['Temperature_section'].isin([5, 6, 7]) &
    scaled_data['pCH4_per_pO2_section'].isin([5, 6, 7]) &
    scaled_data['Contact_time_section'].isin([5, 6, 7])
)

# Extrapolation mask: data points not falling within the interpolation sections
extrapolation_mask = ~interpolation_mask

# Splitting the data into interpolation and extrapolation sets
interpolation_data = scaled_data[interpolation_mask]
extrapolation_data = scaled_data[extrapolation_mask]

# Displaying the number of data points in each set
interpolation_data_count = len(interpolation_data)
extrapolation_data_count = len(extrapolation_data)

print("Interpolation Data Count:", interpolation_data_count)
print("Extrapolation Data Count:", extrapolation_data_count)
# %%
interpolation_mask = (
    scaled_data['Temperature_section'].apply(lambda x: x in temp_continuous_high_freq) &
    scaled_data['pCH4_per_pO2_section'].apply(lambda x: x in pCH4_continuous_high_freq) &
    scaled_data['Contact_time_section'].apply(lambda x: x in contact_time_continuous_high_freq)
)

# Extrapolation mask: include data points where at least one variable is outside its high-frequency section
extrapolation_mask = ~interpolation_mask

# Splitting the data into interpolation and extrapolation sets
interpolation_data = scaled_data[interpolation_mask]
extrapolation_data = scaled_data[extrapolation_mask]

# Displaying the number of data points in each set
interpolation_data_count = len(interpolation_data)
extrapolation_data_count = len(extrapolation_data)

print("Interpolation Data Count:", interpolation_data_count)
print("Extrapolation Data Count:", extrapolation_data_count)
# %%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plotting the interpolation data with lower transparency
ax.scatter(interpolation_data['Temperature'], interpolation_data['pCH4_per_pO2'], interpolation_data['Contact time'], 
           color='blue', label='Interpolation Data', s=30, alpha=0.8)

# Plotting the extrapolation data with higher transparency
ax.scatter(extrapolation_data['Temperature'], extrapolation_data['pCH4_per_pO2'], extrapolation_data['Contact time'], 
           color='red', label='Extrapolation Data', s=30, alpha=0.3)

# Setting labels and title
ax.set_xlabel('Temperature')
ax.set_ylabel('pCH4_per_pO2')
ax.set_zlabel('Contact time')
ax.set_title('3D Scatter Plot of Variables')

# Adding a legend
ax.legend()

# Showing the plot
plt.show()
# %%
# Creating 3D scatter plots
fig = plt.figure(figsize=(18, 6))

# Y(C2)_predicted vs Temperature and pCH4_per_pO2
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(interpolation_data['Temperature'], interpolation_data['pCH4_per_pO2'], interpolation_data['Y(C2)_predicted'], color='blue', alpha=0.8)
ax1.scatter(extrapolation_data['Temperature'], extrapolation_data['pCH4_per_pO2'], extrapolation_data['Y(C2)_predicted'], color='red', alpha=0.3)
ax1.set_xlabel('Temperature')
ax1.set_ylabel('pCH4_per_pO2')
ax1.set_zlabel('Y(C2)_predicted')
ax1.set_title('Y(C2)_predicted vs Temperature & pCH4_per_pO2')

# Y(C2)_predicted vs Temperature and Contact time
ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(interpolation_data['Temperature'], interpolation_data['Contact time'], interpolation_data['Y(C2)_predicted'], color='blue', alpha=0.8)
ax2.scatter(extrapolation_data['Temperature'], extrapolation_data['Contact time'], extrapolation_data['Y(C2)_predicted'], color='red', alpha=0.3)
ax2.set_xlabel('Temperature')
ax2.set_ylabel('Contact time')
ax2.set_zlabel('Y(C2)_predicted')
ax2.set_title('Y(C2)_predicted vs Temperature & Contact time')

# Y(C2)_predicted vs pCH4_per_pO2 and Contact time
ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(interpolation_data['pCH4_per_pO2'], interpolation_data['Contact time'], interpolation_data['Y(C2)_predicted'], color='blue', alpha=0.8)
ax3.scatter(extrapolation_data['pCH4_per_pO2'], extrapolation_data['Contact time'], extrapolation_data['Y(C2)_predicted'], color='red', alpha=0.3)
ax3.set_xlabel('pCH4_per_pO2')
ax3.set_ylabel('Contact time')
ax3.set_zlabel('Y(C2)_predicted')
ax3.set_title('Y(C2)_predicted vs pCH4_per_pO2 & Contact time')

plt.tight_layout()
plt.show()
# %%
scaler = MinMaxScaler()
scaler.fit(data[['Temperature', 'pCH4_per_pO2', 'Contact time', 'Y(C2)_predicted']])
original_data = pd.DataFrame(scaler.inverse_transform(scaled_data[['Temperature', 'pCH4_per_pO2', 'Contact time', 'Y(C2)_predicted']]), columns=data.columns)

# Separating the train and test sets based on the original mask
original_train_set = original_data[interpolation_mask]
original_test_set = original_data[extrapolation_mask]
# %%
original_train_set = original_data[interpolation_mask]
original_test_set = original_data[extrapolation_mask]
# %%
# Creating 3D scatter plots
fig = plt.figure(figsize=(18, 6))

# Y(C2)_predicted vs Temperature and pCH4_per_pO2
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(original_train_set['Temperature'], original_train_set['pCH4_per_pO2'], original_train_set['Y(C2)_predicted'], color='blue', alpha=0.8)
ax1.scatter(original_test_set['Temperature'], original_test_set['pCH4_per_pO2'], original_test_set['Y(C2)_predicted'], color='red', alpha=0.3)
ax1.set_xlabel('Temperature')
ax1.set_ylabel('pCH4_per_pO2')
ax1.set_zlabel('Y(C2)_predicted')
ax1.set_title('Y(C2)_predicted vs Temperature & pCH4_per_pO2')

# Y(C2)_predicted vs Temperature and Contact time
ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(original_train_set['Temperature'], original_train_set['Contact time'], original_train_set['Y(C2)_predicted'], color='blue', alpha=0.8)
ax2.scatter(original_test_set['Temperature'], original_test_set['Contact time'], original_test_set['Y(C2)_predicted'], color='red', alpha=0.3)
ax2.set_xlabel('Temperature')
ax2.set_ylabel('Contact time')
ax2.set_zlabel('Y(C2)_predicted')
ax2.set_title('Y(C2)_predicted vs Temperature & Contact time')

# Y(C2)_predicted vs pCH4_per_pO2 and Contact time
ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(original_train_set['pCH4_per_pO2'], original_train_set['Contact time'], original_train_set['Y(C2)_predicted'], color='blue', alpha=0.8)
ax3.scatter(original_test_set['pCH4_per_pO2'], original_test_set['Contact time'], original_test_set['Y(C2)_predicted'], color='red', alpha=0.3)
ax3.set_xlabel('pCH4_per_pO2')
ax3.set_ylabel('Contact time')
ax3.set_zlabel('Y(C2)_predicted')
ax3.set_title('Y(C2)_predicted vs pCH4_per_pO2 & Contact time')

plt.tight_layout()
plt.show()
# %%
original_train_set.to_csv(r'C:\Users\OJH\OneDrive\SKKU\연구자료\Catalyst AI\New Database\Na_W_Mn_train.csv')
# %%
original_test_set.to_csv(r'C:\Users\OJH\OneDrive\SKKU\연구자료\Catalyst AI\New Database\Na_W_Mn_test.csv')
# %%
