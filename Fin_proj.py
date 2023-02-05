import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict, KFold
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

plt.style.use('seaborn')

file = pd.read_csv('Data_2010_new.csv', sep=';')
df = pd.DataFrame(file)
file_test = pd.read_csv('2010_testing.csv', sep=';')
df_test = pd.DataFrame(file_test)


def preprocessing(data):
    data[['price']] = data[['price']].apply(pd.to_numeric)
    data = data[np.abs(data["price"]-data["price"].mean())<=(3*data["price"].std())]

    # print(data.shape)
    data = data.drop_duplicates(keep='last')
    print(data.shape)
    print(data)

    y = data.price
    X = data.drop('status', axis = 1).drop('price', axis = 1)

    return X, y

def model(pipeline, parameters, X_train, y_train, X, y,str1):

    grid_obj = GridSearchCV(estimator=pipeline,
                            param_grid=parameters,
                            cv=3,
                            scoring='r2',
                            verbose=2,
                            n_jobs=1,
                            refit=True)
    grid_obj.fit(X_train, y_train)

    '''Results'''

    results = pd.DataFrame(pd.DataFrame(grid_obj.cv_results_))
    results_sorted = results.sort_values(by=['mean_test_score'], ascending=False)

    print("##### Results")
    print(results_sorted)

    print("best_index", grid_obj.best_index_)
    print("best_score", grid_obj.best_score_)
    print("best_params", grid_obj.best_params_)

    '''Cross Validation'''

    estimator = grid_obj.best_estimator_
    shuffle = KFold(n_splits=10,
                    shuffle=True,
                    random_state=0)
    cv_scores = cross_val_score(estimator,X,y.values.ravel(),cv=shuffle,scoring='r2')
    print("CV Results")
    print("mean_score", cv_scores.mean())

    y_pred = cross_val_predict(estimator, X, y, cv=shuffle)   # training set
    # y_pred = estimator.predict(X)                               # testing set

    plt.scatter(y, y_pred)
    plt.xlim((0,3500000))
    plt.ylim((0,3500000))
    plt.plot([0,3500000], [0,3500000], "g--", lw=1, alpha=0.4)
    plt.xlabel("True prices")
    plt.ylabel("Predicted prices")
    plt.title('Predicted prices (EUR) vs. True prices (EUR)')
    plt.savefig('{}.jpg'.format(str1))
    plt.show()

# Pipeline and Parameters - Linear Regression
pipe_ols = Pipeline([('scl', StandardScaler()),
           ('clf', LinearRegression())])
param_ols = {}

# Pipeline and Parameters - XGBoost
pipe_xgb = Pipeline([('clf', xgb.XGBRegressor())])
param_xgb = {'clf__max_depth':[5],
             'clf__min_child_weight':[6],
             'clf__gamma':[0.01],
             'clf__subsample':[0.7],
             'clf__colsample_bytree':[1]}

# Pipeline and Parameters - KNN
pipe_knn = Pipeline([('clf', KNeighborsRegressor())])
param_knn = {'clf__n_neighbors':[3, 5, 10, 15, 25, 30, 50, 100]}

# Pipeline and Parameters - Lasso
pipe_lasso = Pipeline([('scl', StandardScaler()),('clf', Lasso(max_iter=3000))])
param_lasso = {'clf__alpha': [0.001, 0.01, 0.1, 1, 10, 30, 50, 100]}

# Pipeline and Parameters - Ridge
pipe_ridge = Pipeline([('scl', StandardScaler()),('clf', Ridge())])
param_ridge = {'clf__alpha': [0.001, 0.01, 0.1, 1, 10, 30, 50, 100]}

# Pipeline and Parameters - Polynomial Regression
pipe_poly = Pipeline([('scl', StandardScaler()),('polynomial', PolynomialFeatures()),('clf', LinearRegression())])
param_poly = {'polynomial__degree': [2, 4, 6]}

# Pipeline and Parameters - Decision Tree Regression
pipe_tree = Pipeline([('clf', DecisionTreeRegressor())])
param_tree = {'clf__max_depth': [2, 5, 10, 20, 30],
             'clf__min_samples_leaf': [5,10,50,100,150,200]}

# Pipeline and Parameters - Random Forest
pipe_forest = Pipeline([('clf', RandomForestRegressor())])
param_forest = {'clf__n_estimators': [10, 20, 50, 100],
                'clf__max_features': [None, 1, 2, 3, 5],
                'clf__max_depth': [1, 2, 5, 10]}

# Pipeline and Parameters - MLP Regression
pipe_neural = Pipeline([('scl', StandardScaler()),('clf', MLPRegressor())])
param_neural = {'clf__alpha': [0.001, 0.01, 0.1, 1, 10, 100],
                'clf__hidden_layer_sizes': [(5),(10,10),(7,7,7)],
                'clf__solver': ['lbfgs'],
                'clf__activation': ['relu', 'tanh'],
                'clf__learning_rate' : ['constant', 'invscaling']}

X_train, y_train = preprocessing(df)
X_test, y_test = preprocessing(df_test)

# Execute model hyperparameter tuning and crossvalidation
model(pipe_ols, param_ols, X_train, y_train, X_test, y_test,"ols_tmp")
model(pipe_xgb, param_xgb, X_train, y_train, X_test, y_test,"xgb_tmp")
model(pipe_knn, param_knn, X_train, y_train, X_test, y_test,"knn_tmp")
model(pipe_lasso, param_lasso, X_train, y_train, X_test, y_test,"lasso_tmp")
model(pipe_ridge, param_ridge, X_train, y_train, X_test, y_test,"ridge_tmp")
model(pipe_poly, param_poly, X_train, y_train, X_test, y_test,"poly_tmp")
model(pipe_tree, param_tree, X_train, y_train, X_test, y_test,"tree_tmp")
model(pipe_forest, param_forest, X_train, y_train, X_test, y_test,"forest_tmp")
model(pipe_neural, param_neural, X_train, y_train, X_test, y_test,"neural_tmp")


