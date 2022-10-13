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

def feature_engineering(dataframe_train, dataframe_test):
    dataframe_train, bairros_categ = others_group(dataframe_train, variable = 'bairro', threshold = 50)
    bairros_features = features_localizacao(dataframe_train, variable = 'bairro_categ')
    dataframe_train = dataframe_train.merge(bairros_features, on='bairro_categ').drop(['bairro_count', 'bairro'], axis=1)
    dataframe_test = dataframe_test.merge(bairros_categ, on='bairro').drop(['bairro_count', 'bairro'], axis=1).fillna('outros')
    dataframe_test = dataframe_test.merge(bairros_features, on='bairro_categ')#.drop(['bairro_count', 'bairro'], axis=1).fillna('outros')
    return dataframe_train, dataframe_test

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


