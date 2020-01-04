import pymrmr
import numpy as np
np.random.seed(0)
import src.utils
from sklearn.utils import shuffle
import pandas as pd
from sklearn.feature_selection import SelectFromModel,RFE
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier

def mrmr(x_train, y_train, n_features=40):
    x_train.insert(loc=0, column='class', value=y_train)
    features = pymrmr.mRMR(x_train, 'MIQ', n_features)

    return features


def lasso(x_train, y_train, n_features=40):
    clf = SelectFromModel(linear_model.Lasso(alpha=0.01), max_features=n_features)
    clf.fit(x_train, y_train)
    lasso_support = clf.get_support()
    features = df.loc[:, lasso_support].columns.tolist()

    return features


def rfe(x_train, y_train, n_features=40):
    rfe_selector = RFE(estimator=linear_model.LogisticRegression(random_state=0), n_features_to_select=n_features)
    rfe_selector.fit(x_train, y_train)
    rfe_support = rfe_selector.get_support()
    rfe_features = x_train.loc[:, rfe_support].columns.tolist()

    return rfe_features


def deep_forest(x_train, y_train, n_features=40):
    features_list_path = './feature_selection/deep_forest_features_list'
    df = pd.read_csv(features_list_path, header=None, sep=',')
    features_list = df.loc[:,0].tolist()

    return features_list[:n_features]

InputFile = '../data/diabimmune_karelia_metaphlan_table.txt'
MetadataFile = '../data/metadata.csv'
_, num_features, subjects, meta_file, time_points, data = src.utils .lstm_raw_input(MetadataFile, InputFile)

data = data.T
data = shuffle(data)

(x_train,y_train),(x_validate,y_validate),(x_test,y_test),\
(x_train_ids,y_validate_ids,y_test_ids) = src.utils.split_dataset(subjects, data, time_points,meta_file)

df = np.concatenate([x_train, x_validate, x_test])
labels = np.concatenate([y_train, y_validate, y_test])
labels= [1 if l == True else 0 for l in labels]
ids = x_train_ids + y_validate_ids + y_test_ids

df = pd.DataFrame(df,index=ids, columns=data.columns)
###############mrmr####################
# features = mrmr(df, labels)
# df = pd.DataFrame(df,index=ids, columns=features)
# df.to_csv('mrmr.txt',sep='\t')
###############Lasso####################
# features = lasso(df, labels)
# df = pd.DataFrame(df,index=ids, columns=features)
# df.to_csv('lasso.txt',sep='\t')
###############rfe####################
# features = rfe(df, labels)
# df = pd.DataFrame(df,index=ids, columns=features)
# df.to_csv('rfe.txt',sep='\t')
###############deep forest####################
# features = deep_forest(df,labels)
# df = pd.DataFrame(df,index=ids, columns=features)
# df.to_csv('./feature_selection/deep_forest.txt',sep='\t')


df_hat = pd.read_csv('./feature_selection/deep_forest.txt',sep='\t',index_col=0,header=0)

rf = RandomForestClassifier(n_estimators=500, random_state=0)
rf.fit()




