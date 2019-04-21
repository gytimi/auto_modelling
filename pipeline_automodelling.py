import auto_modelling
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import neighbors
from sklearn import tree

# Ignore some warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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


# FIX reading from file
my_data = auto_modelling.my_reader('DataSet_Hitelbiralat_Prepared.csv', separ=';')
# Classification
my_data['TARGET_LABEL_GOOD'] = 1 - my_data['TARGET_LABEL_BAD']
# target = 'TARGET_LABEL_GOOD'

# Regression
target = 'PERSONAL_NET_INCOME'

# Train-test split
train_df, test_df = auto_modelling.my_train_test_split(my_data)
# Pure numbers will be the input variables
input_vars = auto_modelling.to_pure_numbers(my_data)

# Choosing if it is a regression or classification
regression, classification = auto_modelling.guess_goal(my_data, target)

# Modelling and building the pipeline
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

# List of pipelines
pipelines = [pipe_1, pipe_2, pipe_3]

# Fit the pipelines
for pipe in pipelines:
    pipe.fit(train_df[input_vars], train_df[target])

# Evaluation
for idx, val in enumerate(pipelines):
    print('%s: ' % (pipe_dict[idx]))
    print(auto_modelling.my_evaluation_pipe(val.predict(test_df), test_df[target], regression, classification))
