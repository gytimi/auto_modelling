import auto_modelling
my_data = auto_modelling.my_reader('DataSet_Lakas.csv')
target = auto_modelling.choose_target(my_data)
#target = 'price_created_at'
regression, classification = auto_modelling.guess_goal(my_data, target)
input_vars = auto_modelling.to_pure_numbers(my_data)
my_data = auto_modelling.missing_value_handle(my_data)
train_df, test_df = auto_modelling.my_train_test_split(my_data)
test_df = auto_modelling.modelling(my_data, regression, classification, train_df, test_df, input_vars, target)
print(auto_modelling.my_evaulation(test_df, target, regression, classification))