import feature_engineering as fe
def model_validation(dataframe, features, target, time_model = False, model = None):
    from sklearn.model_selection import  KFold
    from sklearn.linear_model import LinearRegression    

    mape_results = []    
    cv_outer = KFold(n_splits=4,shuffle=True)        
    for train_ix, test_ix in cv_outer.split(dataframe):
        dataframe_train, dataframe_test = dataframe.iloc[train_ix, :], dataframe.iloc[test_ix,:]
        dataframe_train, dataframe_test = fe.feature_engineering(dataframe_train, dataframe_test )
        X_train, X_test = dataframe_train[features] ,dataframe_test[features]
        y_train, y_test = dataframe_train[target] ,dataframe_test[target] 
        
        linear = LinearRegression()
        if model == None:
            model = linear.fit(X_train, y_train)
        prediction = model.predict(X_test)
        
        dataframe_test['y_true'] = y_test
        dataframe_test['y_pred'] = prediction
        dataframe_test= evaluate_model(dataframe_test, y_true = 'y_true', 
                                        y_pred = 'y_pred', time_model = time_model)
        #return dataframe_test
        mape_results.append(dataframe_test['error_mape'].mean())
    return mape_results
        
def evaluate_model(dataframe, y_true, y_pred, time_model = False):
    import numpy as np
    error_column = 'error_mape'
    dataframe[error_column] = (dataframe[y_true] - dataframe[y_pred])/np.abs(dataframe[y_true])
    if time_model:
        dataframe.loc[(dataframe['sold']==1) & (dataframe[error_column] > 0), error_column] = 0
        dataframe.loc[(dataframe['sold']==1) & (dataframe[error_column] < 0), error_column] = -dataframe[(dataframe['sold']==1) & (dataframe[error_column] < 0)][error_column]
        dataframe.loc[(dataframe['sold']==0) & (dataframe[error_column] < 0), error_column] = 0
    else:
        dataframe[error_column] = np.abs(dataframe[error_column])
    return dataframe
