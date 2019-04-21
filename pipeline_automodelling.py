from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import neighbors
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
import pandas as pd
from math import sqrt
import auto_modelling

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('mode.chained_assignment', None)


def evaulation(y_pred, y_true, regression, classification):
    l=[]
    e = []
    c = []
    if classification:
        c = ['Accuracy', 'AUC', 'RMSE']
        # Accuracy
        e.append(accuracy_score(y_true, y_pred))
        # AUC
        e.append(roc_auc_score(y_true, y_pred))
        # RMSE
        e.append(sqrt(mean_squared_error(y_true, y_pred)))
    if regression:
        c = ['Mean Absolute Error', 'Mean Squarred Error', 'R2 Score', 'Explained Variance score']
        # TODO: nem tunnek jonak az ertekek
        # Mean Absolute Error
        e.append(mean_absolute_error(y_true, y_pred))
        # Mean Squarred Error
        e.append(mean_squared_error(y_true, y_pred))
        # R2 Score
        e.append(r2_score(y_true, y_pred))
        # Explained Variance score
        e.append(explained_variance_score(y_true, y_pred))
    l.append(e)
    df_l = pd.DataFrame(l, columns = c)
    return df_l

class Missing_Value_Handle(BaseEstimator):
    def __init__(self, x):
        self.x = x

    def fit(self, x, y=None):
        return self

    def transform(self, my_data):
        desc_df = my_data.describe()
        desc_col = desc_df.columns
        for c in desc_col:
            my_data[c].fillna(desc_df[c][3] - 1, inplace=True)
        return my_data

my_data = auto_modelling.my_reader('DataSet_Hitelbiralat_Prepared.csv', separ=';')
#Classfication
my_data['TARGET_LABEL_GOOD'] = 1 - my_data['TARGET_LABEL_BAD']
#target = 'TARGET_LABEL_GOOD'

#Regression
target = 'PERSONAL_NET_INCOME'


train_df, test_df = auto_modelling.my_train_test_split(my_data)
input_vars = auto_modelling.to_pure_numbers(my_data)

#regression or classification
regression, classification = auto_modelling.guess_goal(my_data, target)

#Modelling
n_neighbors = 15

X = train_df[input_vars]

if regression:
    pipe_1 = Pipeline([('missing', Missing_Value_Handle(X)),
                      ('model', LinearRegression(fit_intercept=True))])
    pipe_2 = Pipeline([('missing', Missing_Value_Handle(X)),
                      ('model', neighbors.KNeighborsRegressor(n_neighbors, weights='distance'))])
    pipe_3 = Pipeline([('missing', Missing_Value_Handle(X)),
                      ('model', tree.DecisionTreeRegressor())])
    pipe_dict = {0: 'LinearRegression', 1: 'KNeighborsRegressor', 2: 'DecisionTreeRegressor'}


if classification:
    pipe_1 = Pipeline([('missing', Missing_Value_Handle(X)),
                       ('model', LogisticRegression(random_state=42))])
    pipe_2 = Pipeline([('missing', Missing_Value_Handle(X)),
                       ('model', neighbors.KNeighborsClassifier(n_neighbors))])
    pipe_3 = Pipeline([('missing', Missing_Value_Handle(X)),
                       ('model', tree.DecisionTreeClassifier())])
    pipe_dict = {0: 'LogisticRegression', 1: 'KNeighborsClassifier', 2: 'KNeighborsClassifier'}


# List of pipelines for ease of iteration
pipelines = [pipe_1, pipe_2, pipe_3]

# Fit the pipelines
for pipe in pipelines:
    pipe.fit(train_df[input_vars], train_df[target])

# Compare accuracies
for idx, val in enumerate(pipelines):
    print('%s: ' % (pipe_dict[idx]))
    print(evaulation(val.predict(test_df), test_df[target], regression, classification))
