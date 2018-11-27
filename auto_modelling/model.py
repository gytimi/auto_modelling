import pandas as pd
import numpy as np
from sklearn import neighbors
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from math import sqrt

def my_reader(filename, sheetname = 'Sheet1', separ=','):
    filename_list = filename.split('.')
    extension = filename_list[-1]
    if (extension == 'csv'):
        df = pd.read_csv(filename,sep=separ)
    if (extension == 'json'):
        df = pd.read_json(filename)
    if (extension == 'html'):
        df = pd.read_html(filename)
    if (extension == 'xls'):
        xlsx = pd.ExcelFile(filename)
        df = pd.read_excel(xlsx, sheetname)
    if (extension == 'xlsx'):
        xlsx = pd.ExcelFile(filename)
        df = pd.read_excel(xlsx, sheetname)
    if (extension == 'feather'):
        df = pd.read_feather(filename)
    if (extension == 'parquet'):
        df = pd.read_parquet(filename)
    if (extension == 'msg'):
        df = pd.read_msgpack(filename)
    if (extension == 'dta'):
        df = pd.read_stata(filename)
    if (extension == 'sas7bdat'):
        df = pd.read_sas(filename)
    if (extension == 'pkl'):
        df = pd.read_pickle(filename)
    return df
	
def choose_target(mydata):
    data_columns = list(mydata.columns)
    print('These are your columns: ', data_columns)
    t = input('Write here your target: ')
    return t

def guess_goal(mydata, target):
    cardin = dict(mydata.apply(pd.Series.nunique))
    target_type = mydata.dtypes[target]
    if((target_type == 'float64') | (target_type == 'int64')):
        if (cardin[target] > 50):
            regression = True
            classification = False
        else:
            regression = False
            classification = True
    else:
        regression = False
        classification = True
    return regression, classification

def to_pure_numbers(mydata):
    num_type = (mydata.dtypes == "float64") | (mydata.dtypes == "int64")
    return (list((mydata.dtypes[num_type]).keys()))

def missing_value_handle(my_data):
    desc_df = my_data.describe()
    desc_col = desc_df.columns
    for c in desc_col:
        my_data[c].fillna(desc_df[c][3]-1, inplace=True)
    return my_data
		
def my_train_test_split(mydata):
    train_df, test_df = train_test_split(mydata,test_size=0.5)
    return train_df, test_df

def modelling(my_data, regression, classification, train_df, test_df, input_vars, target):
    n_neighbors = 15
    if regression:
        # LinearRegression
        model1 = LinearRegression(fit_intercept=True)
        x_train = train_df[input_vars]
        y_train = train_df[target]
        x_test = test_df[input_vars]
        y_test = test_df[target]
        model1.fit(x_train, y_train)
        test_df['pred1'] = model1.predict(x_test)
        
        #KNN
        model2 = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')
        model2.fit(x_train, y_train)
        test_df['pred2'] = model2.predict(x_test)
        
        #Decision tree
        model3 = tree.DecisionTreeRegressor()
        model3.fit(x_train, y_train)
        test_df['pred3'] = model3.predict(x_test)
        
    if classification:
        model1 = LogisticRegression(C=1.0)
        model1.fit(train_df[input_vars], train_df[target])
        test_df['pred1']= model1.predict( test_df[input_vars] )
        
        #KNN
        model2 = neighbors.KNeighborsClassifier(n_neighbors)
        model2.fit(x_train, y_train)
        test_df['pred2'] = model2.predict(x_test)
        
        #Decision tree
        model3 = tree.DecisionTreeClassifier()
        model3.fit(x_train, y_train)
        test_df['pred3'] = model3.predict(x_test)
    return test_df
		
def my_evaulation(test_df, target, regression, classification):
    y_true = test_df[target]
    l=[]
    if classification:
        c = ['Accuracy', 'AUC', 'RMSE']
        m = ['Evaluation', 'Linear Regression', 'KNN', 'Decision tree' ]
    if regression:
        c = ['Mean Absolute Error', 'Mean Squarred Error', 'R2 Score', 'Explained Variance score' ]
        m= ['Evaluation', 'Logistic Regression', 'KNN', 'Decision tree']
    l.append(c)
    for i in range(1,4):
        e = []
        y_pred = test_df['pred'+str(i)]
        if classification:
            # Accuracy
            e.append(accuracy_score(y_true, y_pred))
            # AUC
            e.append(roc_auc_score(y_true, y_pred))
            # RMSE
            e.append(sqrt(mean_squared_error(y_true, y_pred)))
        if regression:
            # Regression TODO: nem tunnek jonak az ertekek
            # Mean Absolute Error
            e.append(mean_absolute_error(y_true, y_pred))
            # Mean Squarred Error
            e.append(mean_squared_error(y_true, y_pred))
            # R2 Score
            e.append(r2_score(y_true, y_pred))
            # Explained Variance score
            e.append(explained_variance_score(y_true, y_pred))
        l.append(e)
    df_l = pd.DataFrame(l)
    df_l = df_l.transpose()
    df_l.columns = m
    return(df_l)
	
