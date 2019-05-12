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


class MissingValueHandle(BaseEstimator):
    def __init__(self, x):
        self.x = x

    def fit(self, x, y=None):
        return self

    def transform(self, my_data):
        desc_df = my_data.describe()
        desc_col = desc_df.columns
        for c in desc_col:
            if (missing_value_handle == 'min-1'):
                my_data[c].fillna(desc_df[c][3] - 1, inplace=True)
            if (missing_value_handle == 'mean'):
                my_data[c].fillna(desc_df[c][1], inplace=True)
        return my_data


# Reading config file
config_path = './config.txt'
config_file = open(config_path, 'r')
config_str = config_file.readlines()
filename = ''
missing_value_handle = ''
target = ''
for s in config_str:
    mycode = s[0:-1]
    exec(mycode)
config_file.close()

# FIX reading from file
my_data = auto_modelling.my_reader(filename, separ=';')
# Classification
my_data['TARGET_LABEL_GOOD'] = 1 - my_data['TARGET_LABEL_BAD']
# target = 'TARGET_LABEL_GOOD'

# Regression
#target = 'PERSONAL_NET_INCOME'

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
    pipe_1 = Pipeline([('missing', MissingValueHandle(X)),
                      ('model', LinearRegression(fit_intercept=True))])
    pipe_2 = Pipeline([('missing', MissingValueHandle(X)),
                      ('model', neighbors.KNeighborsRegressor(n_neighbors, weights='distance'))])
    pipe_3 = Pipeline([('missing', MissingValueHandle(X)),
                      ('model', tree.DecisionTreeRegressor())])
    pipe_dict = {0: 'LinearRegression', 1: 'KNeighborsRegressor', 2: 'DecisionTreeRegressor'}

if classification:
    pipe_1 = Pipeline([('missing', MissingValueHandle(X)),
                       ('model', LogisticRegression(random_state=42))])
    pipe_2 = Pipeline([('missing', MissingValueHandle(X)),
                       ('model', neighbors.KNeighborsClassifier(n_neighbors))])
    pipe_3 = Pipeline([('missing', MissingValueHandle(X)),
                       ('model', tree.DecisionTreeClassifier())])
    pipe_dict = {0: 'LogisticRegression', 1: 'KNeighborsClassifier', 2: 'KNeighborsClassifier'}

# List of pipelines
pipelines = [pipe_1, pipe_2, pipe_3]

# Fit the pipelines
for pipe in pipelines:
    pipe.fit(train_df[input_vars], train_df[target])

# Open new file
result_path = './result.txt'
result_file = open(result_path,'w')
result_file.write("filename: " + str(filename) + '\n')
result_file.write("target: " + str(target) + '\n')
result_file.write("missing_value_handle: " + str(missing_value_handle) + '\n')
if regression:
    result_file.write("Regression" + '\n')
else:
    result_file.write("Classification" + '\n')
result_file.write('\n' + "Evaluation:" + '\n')

# Evaluation
for idx, val in enumerate(pipelines):
    #print('%s: ' % (pipe_dict[idx]))
    #print(auto_modelling.my_evaluation_pipe(val.predict(test_df), test_df[target], regression, classification))
    result_file.write(str(pipe_dict[idx]) + '\n')
    result_file.write(str(auto_modelling.my_evaluation_pipe(val.predict(test_df), test_df[target], regression, classification)) + '\n')

result_file.close()
