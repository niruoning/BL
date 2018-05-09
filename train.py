import numpy as np
import pandas as pd
import lightgbm as lgb
import datetime

def sub(train, test, features):
    X_train, X_test = train[features], test[features]
    y_train = train['buy']
    clf = lgb.LGBMClassifier(max_depth=3)
    clf.fit(X_train,y_train)
    pred= clf.predict_proba(X_test)[:,1]
    test['Prob'] = pred
    test['pred_date'] = datetime.datetime(2017, 5, 15)
    prediction = test[['user_id','Prob','pred_date']]
    prediction.sort_values(by = 'Prob', ascending = False, inplace = True)
    prediction[['user_id', 'pred_date']][:50000].to_csv('./result.csv', index=None)
    
def obtaincol(df, delete):
    ColumnName = list(df.columns)
    for i in delete:
        if i in ColumnName:
            ColumnName.remove(i)
    return ColumnName   
if __name__ == '__main__':
    train_data = pd.read_csv('./traina.csv')
    test_data = pd.read_csv('./testa.csv')
    NonTrainableFeatures = ['buy','nextbuy','o_date','a_date','PredictDays','user_id']
    sub(train_data, test_data, obtaincol(train_data, NonTrainableFeatures))