from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def metadata(file_path):
    meta = pd.read_csv(file_path, sep=",", index_col=False)
    meta = meta.dropna(subset=['gid_wgs'])

    meta = meta.dropna(subset=['allergy_milk','allergy_egg','allergy_peanut'])

    allergy_milk = meta.allergy_milk.values.tolist()
    allergy_egg = meta.allergy_egg.values.tolist()
    allergy_peanut = meta.allergy_peanut.values.tolist()

    allergy = []
    for milk, egg, peanut in zip(allergy_milk, allergy_egg, allergy_peanut):
        if milk or egg or peanut:
            allergy.append(True)
        else:
            allergy.append(False)

    meta['allergy'] = allergy
    return meta

def time_points_data(df_meta, df_data):
    timepoints = defaultdict(list)
    groups = df_meta.groupby('subjectID')
    for name, group in groups:
        gid_wgs = group['gid_wgs'].values.tolist()
        if len(gid_wgs) > 1:
            timepoints.setdefault(name, []).extend(gid_wgs)
        else:
            df_meta = df_meta.drop(df_meta[df_meta['subjectID'] == name].index)
            df_data = df_data.drop(columns = gid_wgs)
    return timepoints, df_meta, df_data


def df_genus_features(file_path):
    df = pd.read_csv(file_path, sep="\t", index_col=0)
    # df = (df - df.min()) / (df.max() - df.min()) don't use

    feature_list = list(df.index)
    results = []
    for feature in feature_list:
        fs = feature.split('|')
        if len(fs) == 6:
            results.append(feature)

    return df.loc[results]


def lstm_raw_input(meta_file_name, data_file_name):
    meta_file = metadata(meta_file_name)
    data_file = df_genus_features(data_file_name)

    w_ids = set(data_file.columns.values).intersection(set(meta_file['gid_wgs'].values))
    meta_file = meta_file[meta_file['gid_wgs'].isin(w_ids)]
    data_file = data_file.loc[:, w_ids]

    time_points, meta_file, data_file = time_points_data(meta_file, data_file)
    subjects = list(time_points.keys())

    _, counts = np.unique(meta_file['subjectID'], return_counts=True)
    maxLen = max(counts)
    print(np.sum(counts))

    numFeatures = len(data_file.index)

    print("samples FIN="+ str(len(meta_file[meta_file['country'] == 'FIN'])))
    print("samples RUS=" + str(len(meta_file[meta_file['country'] == 'RUS'])))
    print("samples EST=" + str(len(meta_file[meta_file['country'] == 'EST'])))

    groups = meta_file.groupby('country')
    for name, group in groups:
        subjectIDs = set(group['subjectID'].values.tolist())
        print("subjects " +name +"=" + str(len(subjectIDs)))

    # data_file[data_file.columns] = MinMaxScaler().fit_transform(data_file[data_file.columns])

    return maxLen, numFeatures, subjects, meta_file, time_points, data_file


def lstm_latent_input(meta_file_name, data_file_name):
    meta_file = metadata(meta_file_name)
    data_file = pd.read_csv(data_file_name, sep='\t')
    # data_file = (data_file - data_file.min()) / (data_file.max() - data_file.min())

    w_ids = set(data_file.columns.values).intersection(set(meta_file['gid_wgs'].values))
    meta_file = meta_file[meta_file['gid_wgs'].isin(w_ids)]
    data_file = data_file.loc[:, w_ids]

    time_points, meta_file, data_file = time_points_data(meta_file, data_file)
    subjects = list(time_points.keys())

    _, counts = np.unique(meta_file['subjectID'], return_counts=True)
    maxLen = max(counts)
    print(np.sum(counts))

    numFeatures = len(data_file.index)

    print("samples FIN=" + str(len(meta_file[meta_file['country'] == 'FIN'])))
    print("samples RUS=" + str(len(meta_file[meta_file['country'] == 'RUS'])))
    print("samples EST=" + str(len(meta_file[meta_file['country'] == 'EST'])))

    groups = meta_file.groupby('country')
    for name, group in groups:
        subjectIDs = set(group['subjectID'].values.tolist())
        print("subjects " + name + "=" + str(len(subjectIDs)))

    return maxLen, numFeatures, subjects, meta_file, time_points, data_file


def add_ae_latent_header():
    meta = '../data/metadata.csv'
    data = '../data/diabimmune_karelia_metaphlan_table.txt'
    _, _, _, _, _, data_file = lstm_raw_input(meta, data)

    latent_file = '../data/diabimmune_latent_40_new.csv'
    df = pd.read_csv(latent_file, sep=',', header=None)
    df.to_csv('../data/diabimmune_ae_latent_40_header.csv', sep='\t',
              header=data_file.columns, index=None)


def split_dataset(subjects, data, time_points, meta_data):
    np.random.shuffle(subjects)
    test_split_index = int(0.9 * len(subjects))
    x_train_subjects, x_test_subjects = subjects[0:test_split_index], subjects[test_split_index:]
    validate_split_index = int(0.9 * len(x_train_subjects))
    x_train_subjects, x_validate_subjects = x_train_subjects[0:validate_split_index], x_train_subjects[
                                                                                      validate_split_index:]

    x_train_samples = []
    for subject_id in x_train_subjects:
        x_train_samples.extend(time_points[subject_id])

    x_validate_samples = []
    for subject_id in x_validate_subjects:
        x_validate_samples.extend(time_points[subject_id])

    x_test_samples = []
    for subject_id in x_test_subjects:
        x_test_samples.extend(time_points[subject_id])

    train_x = data.loc[x_train_samples, :]
    validate_x = data.loc[x_validate_samples, :]
    test_x = data.loc[x_test_samples, :]

    train_y = meta_data.loc[meta_data['gid_wgs'].isin(x_train_samples)]['allergy'].values
    validate_y = meta_data.loc[meta_data['gid_wgs'].isin(x_validate_samples)]['allergy'].values
    test_y = meta_data.loc[meta_data['gid_wgs'].isin(x_test_samples)]['allergy'].values
    return (train_x.values,train_y), (validate_x.values,validate_y), (test_x.values,test_y), \
           (x_train_samples, x_validate_samples, x_test_samples)

if __name__ == '__main__':
    meta = '../data/metadata.csv'

    data = '../data/diabimmune_karelia_metaphlan_table.txt'
    _, _, _, _, _, data_file = lstm_raw_input(meta, data)
    #
    # add_ae_latent_header()

    aa = metadata(meta)

    print('aa')

    # latent_data = '../data/diabimmune_ae_latent_25_header.csv'
    # _, _, _, _, _, data_file = lstm_latent_input(meta, latent_data)

