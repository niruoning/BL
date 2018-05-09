import pandas as pd
import numpy as np
import datetime

sku = pd.read_csv('./jdata_sku_basic_info.csv', )
action = pd.read_csv('./jdata_user_action.csv', parse_dates=['a_date'])
basic_info = pd.read_csv('./jdata_user_basic_info.csv')
order = pd.read_csv('./jdata_user_order.csv', parse_dates=['o_date'])
order = order.drop_duplicates(subset =  ['user_id','o_id'],keep='last')


order = pd.merge(order, sku, on='sku_id', how='left')
action = pd.merge(action, sku, how='left', on='sku_id')

order['o_month_series'] = pd.to_datetime(order['o_date']).dt.month + (pd.to_datetime(order['o_date']).dt.year - 2016) * 12 - 5 
action['a_month_series'] = pd.to_datetime(action['a_date']).dt.month + (pd.to_datetime(action['a_date']).dt.year - 2016) * 12 - 5

first_day = datetime.datetime.strptime('2016-05-01 00:00:00', '%Y-%m-%d %H:%M:%S')
order['c_day_series'] = (order['comment_create_tm'] - first_day).apply(lambda x: x.days)
order['o_day_series'] = (order['o_date'] - first_day).apply(lambda x: x.days)
action['a_day_series'] = (action['a_date'] - first_day).apply(lambda x: x.days)

def ActionFeatures(Startday, PrepareDays, PredictDays, temp, dftemp):
    tempfeature = temp[temp.a_day_series < Startday][temp.a_day_series >= Startday-PrepareDays].reset_index(drop=True)
    templabel = temp[temp.a_day_series >= Startday][temp.a_day_series < Startday+PredictDays].reset_index(drop=True)
    dftemp = pd.merge(dftemp, templabel[['user_id','a_date']].drop_duplicates(subset = 'user_id', keep='last'), on = 'user_id',how='left').fillna(0)

    for f in ['price', 'para_1', 'para_2', 'para_3', 'a_num','a_type','a_month_series','a_day_series']:
        a = tempfeature[['user_id',f]].groupby(['user_id']).mean().reset_index()
        a.columns = ['user_id', '{}_a_ave'.format(f)]
        dftemp = pd.merge(dftemp,a, how = 'left', on = 'user_id')
        a = tempfeature[['user_id',f]].groupby(['user_id']).std().reset_index()
        a.columns = ['user_id', '{}_a_std'.format(f)]
        dftemp = pd.merge(dftemp,a, how = 'left', on = 'user_id')
        a = tempfeature[['user_id',f]].groupby(['user_id']).sum().reset_index()
        a.columns = ['user_id', '{}_a_sum'.format(f)]
        dftemp = pd.merge(dftemp,a, how = 'left', on = 'user_id')
        a = tempfeature[['user_id',f]].groupby(['user_id']).apply(pd.DataFrame.kurt).drop(labels='user_id',axis = 1).reset_index()
        a.columns = ['user_id', '{}_a_kurtosis'.format(f)]
        dftemp = pd.merge(dftemp,a, how = 'left', on = 'user_id')
        a = tempfeature[['user_id',f]].groupby(['user_id']).skew().reset_index()
        a.columns = ['user_id', '{}_a_skew'.format(f)]
        dftemp = pd.merge(dftemp,a, how = 'left', on = 'user_id')
        dftemp = pd.merge(dftemp, tempfeature[['user_id',f]].drop_duplicates(subset = 'user_id', keep = 'last'), how = 'left', on = 'user_id')
    dftemp['CreateGroup'] = Startday
    return dftemp

def OrderFeatures(Startday, PrepareDays, PredictDays, temp, dftemp):
    tempfeature = temp[temp.o_day_series < Startday][temp.o_day_series >= Startday-PrepareDays].reset_index(drop=True)
    templabel = temp[temp.o_day_series >= Startday][temp.o_day_series < Startday+PredictDays].reset_index(drop=True)
    templabel['buy'] = 1
    templabel['nextbuy'] = templabel['o_day_series'] - Startday
    dftemp = pd.merge(dftemp, templabel[['user_id','buy']].drop_duplicates(subset = 'user_id', keep='first'), on = 'user_id',how='left').fillna(0)
    dftemp = pd.merge(dftemp, templabel[['user_id','nextbuy']].drop_duplicates(subset = 'user_id', keep='first'), on = 'user_id',how='left').fillna(0)
    dftemp = pd.merge(dftemp, templabel[['user_id','o_date']].drop_duplicates(subset = 'user_id', keep='first'), on = 'user_id',how='left').fillna(0)
    dftemp = pd.merge(dftemp,monthcnt, how = 'left', on = 'user_id')
    for f in ['price', 'para_1', 'para_2', 'para_3', 'o_sku_num','o_month_series','o_day_series']:
        a = tempfeature[['user_id',f]].groupby(['user_id']).mean().reset_index()
        a.columns = ['user_id', '{}_o_ave'.format(f)]
        dftemp = pd.merge(dftemp,a, how = 'left', on = 'user_id')
        a = tempfeature[['user_id',f]].groupby(['user_id']).std().reset_index()
        a.columns = ['user_id', '{}_o_std'.format(f)]
        dftemp = pd.merge(dftemp,a, how = 'left', on = 'user_id')
        a = tempfeature[['user_id',f]].groupby(['user_id']).sum().reset_index()
        a.columns = ['user_id', '{}_o_sum'.format(f)]
        dftemp = pd.merge(dftemp,a, how = 'left', on = 'user_id')
        a = tempfeature[['user_id',f]].groupby(['user_id']).apply(pd.DataFrame.kurt).drop(labels='user_id',axis = 1).reset_index()
        a.columns = ['user_id', '{}_o_kurtosis'.format(f)]
        dftemp = pd.merge(dftemp,a, how = 'left', on = 'user_id')
        a = tempfeature[['user_id',f]].groupby(['user_id']).skew().reset_index()
        a.columns = ['user_id', '{}_o_skew'.format(f)]
        dftemp = pd.merge(dftemp,a, how = 'left', on = 'user_id')
        dftemp = pd.merge(dftemp, tempfeature[['user_id',f]].drop_duplicates(subset = 'user_id', keep = 'last'), how = 'left', on = 'user_id')
    dftemp['CreateGroup'] = Startday
    return dftemp

Train = []
Predict = []
for index, oc in enumerate(np.unique(order.cate)):
    o_temp = order[(order.cate == oc)].reset_index(drop=True)
    o_temp = o_temp.sort_values(by=['user_id','o_day_series']).reset_index(drop=True)
    a_temp = action[(action.cate == oc)].reset_index(drop=True)
    a_temp = a_temp.sort_values(by=['user_id','a_day_series']).reset_index(drop=True)

    o_df = []
    a_df = []
    for Startday in range(180, 335, 10):
        print('creating dataset from {} day'.format(Startday))
        o_df.append(OrderFeatures(Startday, 180, 31, o_temp, basic_info[:]))
        a_df.append(ActionFeatures(Startday, 180, 31, a_temp, basic_info[:]))
    o_df = pd.concat(o_df).reset_index(drop=True)
    a_df = pd.concat(a_df).reset_index(drop=True)
    traindf = pd.merge(o_df, a_df,on = ['user_id', 'CreateGroup'], how = 'left')
    traindf['cate'] = oc
    Train.append(traindf)
    if (oc == 30) | (oc == 101):
        o_df = OrderFeatures(180, 180, 31, o_temp, basic_info[:])
        a_df = ActionFeatures(180, 180, 31, a_temp, basic_info[:])
        predf = pd.merge(o_df, a_df,on = ['user_id', 'CreateGroup'], how = 'left')\
        predf['cate'] = oc
        Predict.append(predf)

Train = pd.concat(Train).reset_index(drop=True)
Predict = pd.concat(Predict).reset_index(drop=True)

Train.to_csv('traina.csv', index = None)
Predict.to_csv('testa', index = None)
