def remove_diff_columns(train,test):
    train_column = train.columns.values.tolist()
    test_column = test.columns.values.tolist()
    not_in_train = [i for i in test_column if i not in train_column]
    not_in_test  = [i for i in train_column if i not in test_column]    
    train.drop(not_in_test,axis=1,inplace=True)
    test.drop(not_in_train,axis=1,inplace=True)
    return train,test

def bairro(X_train,X_test,y_train,train_rf=False):
    import pandas as pd
    X_train['preco_m2'] = y_train['point_estimate']/X_train['useful_area']
    bairro = X_train.groupby('bairro').mean()
    bairro['bairro faixa'] = pd.qcut(bairro['preco_m2'].sort_values(),10,range(10))
    bairro_dic = bairro['bairro faixa'].to_dict()
    X_train['bairro_faixa'] = X_train['bairro'].map(bairro_dic)
    X_test['bairro_faixa'] = X_test['bairro'].map(bairro_dic)
    X_train['bairro_merged'] = X_train['bairro'].mask(X_train['bairro'].map(X_train['bairro'].value_counts(normalize=True)) < 0.01, 'Other')
    bairro_merged = pd.Series(X_train['bairro_merged'].values,index=X_train['bairro']).to_dict()
    X_test['bairro_merged'] = X_test['bairro'].map(bairro_merged)
    X_train.drop(['preco_m2','bairro'],axis=1,inplace=True)
    X_test.drop(['bairro'],axis=1,inplace=True)
    if train_rf == True:
        X_test['bairro_faixa'].fillna(5,inplace=True)
    X_train = pd.get_dummies(X_train,columns=['bairro_merged'])
    X_test = pd.get_dummies(X_test,columns=['bairro_merged'])
    X_train, X_test = remove_diff_columns(X_train,X_test)
    return X_train,X_test

def others_group(dataframe, variable = 'bairro', threshold = 50):
    threshold = 50
    dataframe_count = dataframe.groupby(variable).agg(count = ('sold', 'count')).reset_index()
    dataframe_count.columns = [variable, f'{variable}_count']
    dataframe_count[f'{variable}_categ'] = 'outros'
    dataframe_count.loc[dataframe_count[f'{variable}_count'] > threshold ,f'{variable}_categ'] = dataframe_count[dataframe_count[f'{variable}_count'] > threshold][variable]
    dataframe = dataframe.merge(dataframe_count, how = 'right', on = variable)
    return dataframe, dataframe_count

def features_localizacao(dataframe, variable):
    features_values = dataframe.groupby(variable).agg(room = ('rooms', 'mean'),
                                   garages = ('garages', 'mean'),
                                   useful_area = ('useful_area', 'mean'),
                                   value = ('value', 'mean'),
                                   time_on_market = ('time_on_market', 'mean')).reset_index()
    features_values.columns= [variable, f'room_{variable}', 
                              f'garages_{variable}', 
                              f'useful_area_{variable}', 
                              f'value_{variable}', 
                              f'time_on_market_{variable}']
    return features_values

    
def adjust_values(variable, max_value):
  if variable <max_value:
    return variable
  else:
    return max_value

def pre_processing(dataframe, max_value_dict):
    variables = max_value_dict.keys()
    for variable in variables:
        if max_value_dict[variable] > -1:
            dataframe[variable] = dataframe[variable]\
                    .apply(adjust_values, args = [max_value_dict[variable]])
    return dataframe

def linear_model(dataframe ,features, target, evaluate = True):
  from sklearn.linear_model import Lasso
  from sklearn.model_selection import cross_validate
  #from mapie.regression import MapieRegressor
  X = dataframe[features]
  y = dataframe[target]
  linear_model = Lasso(alpha=10)
  #mapie = MapieRegressor()
  if evaluate:
    scoring = ['neg_mean_absolute_error', 'neg_mean_squared_error', 
               'neg_mean_absolute_percentage_error']
    score = cross_validate(estimator=linear_model, X=X, y=y, 
                           scoring=scoring, cv=5)
    return score
  else:
    model = linear_model.fit(X,y)
    return model




def price_prediction_model(dataframe, features, target, pre_processing_max_value_dict,
                           pre_processing_ = True):
    if pre_processing_:
        dataframe = pre_processing(dataframe, pre_processing_max_value_dict)
    score = linear_model(dataframe, features, target, evaluate = True)
    model= linear_model(dataframe, features, target, evaluate = False)
    return dataframe, score, model
def predict(target_dataframe, model, features, prediction_column = 'prediction',
            pre_processing_max_value_dict = None, pre_processing_ = True):
  if pre_processing_:
      target_dataframe = pre_processing(target_dataframe, pre_processing_max_value_dict)
  X = target_dataframe[features]
  prediction = model.predict(X)
  target_dataframe[prediction_column] = prediction
  return target_dataframe
    
def evaluate_time(dataframe, target_column, prediction_column):
    sold= dataframe[dataframe['sold']==1]
    not_sold= dataframe[dataframe['sold']==0]
    sold['time_rate_error'] = (sold[target_column]-sold[prediction_column])/sold[target_column]
    sold['error'] = (sold[target_column]-sold[prediction_column])
    not_sold['time_rate_error'] = 0
    not_sold['error'] = 0
    not_sold.loc[not_sold[target_column]>not_sold[prediction_column], 'error'] = (not_sold[target_column]-not_sold[prediction_column])
    not_sold.loc[not_sold[target_column]>not_sold[prediction_column], 'time_rate_error'] = (not_sold[target_column]-not_sold[prediction_column])/not_sold[target_column]
    return sold, not_sold

def portfolio_otimization(values, weights, constraint):
  #knaspascak_otimization
  import pulp
  import pandas as pd
  values, weights = values.to_dict(), weights.to_dict()
  items = list(sorted(values.keys()))
  # Create model
  m = pulp.LpProblem("Knapsack", pulp.LpMaximize)
  # Variables
  x = pulp.LpVariable.dicts('x', items, lowBound=0, upBound=1, cat=pulp.LpInteger)
  # Objective
  m += sum(values[i]*x[i] for i in items)
  # Constraint
  m += sum(weights[i]*x[i] for i in items) <= constraint
  # Optimize
  m.solve()
  # Print the status of the solved LP
  print("Status = %s" % pulp.LpStatus[m.status])
  optimization_list = []
  for i in items:
      optimization_list.append([x[i].name, values[i], weights[i], x[i].varValue])
  objective_value = pulp.value(m.objective)
  optimization_results = pd.DataFrame(optimization_list, 
                                      columns = ['index', 'lucro_por_dia', 'value_compra' ,'portfolio'])
  optimization_results['index'] = optimization_results['index'].str.split('_').str[-1].apply(int)
  return optimization_results, objective_value