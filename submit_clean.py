import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as rfr

#load in data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#fill blanks by taking means of columns
train['total_positive_reviews'].fillna((train['total_positive_reviews'].mean()), inplace=True)
train['total_negative_reviews'].fillna((train['total_negative_reviews'].mean()), inplace=True)
test['total_positive_reviews'].fillna((test['total_positive_reviews'].mean()), inplace=True)
test['total_negative_reviews'].fillna((test['total_negative_reviews'].mean()), inplace=True)

#train: convert useful columns values to dummies
g_train = train['genres'].str.get_dummies(sep=',')
c_train = train['categories'].str.get_dummies(sep=',')
t_train = train['tags'].str.get_dummies(sep=',')
u_train = pd.concat([train,g_train,c_train,t_train],axis=1)

#test: convert useful columns values to dummies
g_test = test['genres'].str.get_dummies(sep=',')
c_test = test['categories'].str.get_dummies(sep=',')
t_test = test['tags'].str.get_dummies(sep=',')
u_test = pd.concat([test,g_test,c_test,t_test],axis=1)

#columns alignment between train and test
col_train = u_train.columns.values.tolist()
col_test = u_test.columns.values.tolist()

miss_test = pd.DataFrame(index = range(len(u_test)))
for i in range(len(col_train)):
    if col_train[i] not in col_test:
        miss_test[col_train[i]] = '0'

miss_train = pd.DataFrame(index = range(len(u_train)))
for i in range(len(col_test)):
    if col_test[i] not in col_train:
        miss_train[col_test[i]] = '0'

u_train = pd.concat([u_train,miss_train],axis=1)
u_test = pd.concat([u_test,miss_test],axis=1)

u_train = u_train.loc[:,~u_train.columns.duplicated()]
u_test = u_test.loc[:,~u_test.columns.duplicated()]

#modeling: select necessary features X
u_test = u_test[list(u_train.columns)]
col_del = ['playtime_forever','id','price','genres','categories','tags'\
           ,'purchase_date','release_date']
labels = np.array(u_train['playtime_forever'])
features = u_train.drop(col_del, axis = 1)
feature_list = list(features.columns)
featrues = np.array(features)

#train: fit training data using random forest
rf = rfr(n_estimators = 1000,random_state = 42)
rf.fit(features, labels)
u_train['yhat'] = rf.predict(features)

#test: fit testing data with above model
features_test = u_test.drop(col_del, axis = 1)
feature_list_test = list(features_test.columns)
featrues_test = np.array(features_test)
u_test['playtime_forever'] = rf.predict(features_test)

#result to csv
output = u_test[['id','playtime_forever']]
output.to_csv('submission.csv',index=False)




