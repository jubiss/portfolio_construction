import pandas as pd
import prediction_functions as pf
import testing_model as tm
import feature_engineering as fe

def get_data():
    listings = pd.read_csv('simulated_listings1.csv')
    target = pd.read_csv('target_apartments1.csv')
    dados_gps = pd.read_csv('enderecos_problema.csv')[['endereço', 'latitude', 'longitude']]
    listings= listings.merge(dados_gps, how='inner', on=['latitude', 'longitude'])
    listings['bairro'] = listings['endereço'].str.split(',').str[-4].str.split('-').str[-1]
    listings = listings.drop(['latitude', 'longitude', 'endereço'], axis=1)
    target= target.merge(dados_gps, how='inner', on=['latitude', 'longitude'])
    target['bairro'] = target['endereço'].str.split(',').str[-4].str.split('-').str[-1]
    target= target.drop(['latitude', 'longitude', 'endereço'], axis=1)
    return listings, target
listings, target = get_data()

#Model construction
features_price = ['garages', 'useful_area', 'value_bairro_categ']
target_price = ['value']
features_time_to_sell = ['rooms', 'garages', 'useful_area', 'interior_quality', 'value','room_bairro_categ',
       'garages_bairro_categ', 'useful_area_bairro_categ',
       'value_bairro_categ', 'time_on_market_bairro_categ']

features_time_to_sell_prediction = ['rooms', 'garages', 'useful_area', 
                                    'interior_quality', 'price_prediction', 'room_bairro_categ',
       'garages_bairro_categ', 'useful_area_bairro_categ',
       'value_bairro_categ', 'time_on_market_bairro_categ']

target_time_to_sell = ['time_on_market']


pre_processing_price = {'rooms':5, 'garages':5, 'useful_area':400}
pre_processing_time = {'rooms':4, 'garages':4, 'useful_area':300, 'value':2.5*10**6}


#
listings= pf.pre_processing(listings, pre_processing_price)
score_price_ = tm.model_validation(listings, features_price, target_price, time_model = False)
score_time_= tm.model_validation(listings, features_time_to_sell, target_time_to_sell, time_model = True)

listings, target = fe.feature_engineering(listings, target)
dataframe_processed_price, score_price, model_price = pf.price_prediction_model(listings, 
                                                 features_price, 
                                                 target_price, 
                                                 pre_processing_price,
                                                 pre_processing_ = True)
dataframe_processed_time, score_time, model_time = pf.price_prediction_model(listings[listings['sold']==1], 
                                                 features_time_to_sell, 
                                                 target_time_to_sell, 
                                                 pre_processing_time,
                                                 pre_processing_ = True)
# Test time
dataframe_time = pf.predict(listings, model_time, features_time_to_sell,
                         prediction_column='market_prediction', 
                         pre_processing_max_value_dict = pre_processing_time,
                         pre_processing_  = True)
sold, not_sold = pf.evaluate_time(dataframe_time, 'time_on_market', 'market_prediction')

# Predictions

target_pred = pf.predict(target, model_price, features_price,
                         prediction_column='price_prediction', 
                         pre_processing_max_value_dict = pre_processing_price,
                         pre_processing_  = True)
target_pred = pf.predict(target_pred, model_time, features_time_to_sell_prediction,
                         prediction_column='market_prediction', 
                         pre_processing_max_value_dict = pre_processing_time,
                         pre_processing_  = True)

#Lucro por dia
target_pred['lucro'] = target_pred['price_prediction'] - target_pred['value']
target_pred['lucro_por_dia'] = target_pred['lucro']/target_pred['market_prediction']

weights = target_pred['value']
values = target_pred['lucro_por_dia']
capacity = 150*10**6
optimization_results, objective_value = pf.portfolio_otimization(values, weights, capacity)
target_prediction_opt = target_pred.join(optimization_results, how = 'inner', rsuffix= '_opt')