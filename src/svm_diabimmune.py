
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import sys
import numpy as np
import os
import struct
import argparse
from array import array as pyarray
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, matthews_corrcoef
import pandas as pd
import utils


parser = argparse.ArgumentParser(description='SVM on data')

parser.add_argument("--data", help="raw or latent-40 or deep_forest or mrmr or lasso or rfe")

args = parser.parse_args()


if __name__ == '__main__':
	if args.data == None:
		print("Please specify raw or latent for data flag")
	else:
		dataset=args.data
		svm_accuracy = []
		svm_roc_auc = []
		svm_precision = []
		svm_recall = []
		svm_f_score = []
		svm_pred = []
		svm_prob = []
		svm_mcc = []

		meta_file = '../data/metadata.csv'
		data_file = '../data/diabimmune_karelia_metaphlan_table.txt'
		_, _, _, fp, time_point, data = utils.lstm_latent_input(meta_file, data_file)
		allergy = fp["allergy"]
		allergy = pd.factorize(allergy)
		subject = fp["subjectID"].values

		labels = allergy[1]
		allergy = allergy[0]

		subject_data = {'ID': subject, 'label': allergy}
		split_df = pd.DataFrame(data=subject_data).groupby("ID").median()

		split_sub = split_df.index.values
		split_lab = np.array(split_df[["label"]].as_matrix()).reshape(-1)

		sample_ids = []
		for suject_id in subject:
			sample_ids += time_point.get(suject_id)

		if dataset == "latent-40":
			data = pd.read_csv("./feature_selection/latent40.txt", index_col=0, header=None, sep='\t')
			data = data[data.columns[:-1]]
			data = data.loc[sample_ids, :]
			data = data.as_matrix()
		elif dataset == "raw":
			data = data.transpose().as_matrix()
		elif dataset == "deep_forest":
			data = pd.read_csv("./feature_selection/deep_forest.txt", index_col=0, header=0, sep='\t')
			data = data.loc[sample_ids, :]
			data = data.as_matrix()
		elif dataset == "lasso":
			data = pd.read_csv("./feature_selection/lasso.txt", index_col=0, header=0, sep='\t')
			data = data.loc[sample_ids, :]
			data = data.as_matrix()
		elif dataset == "mrmr":
			data = pd.read_csv("./feature_selection/lasso.txt", index_col=0, header=0, sep='\t')
			data = data.loc[sample_ids, :]
			data = data.as_matrix()
		elif dataset == "rfe":
			data = pd.read_csv("./feature_selection/rfe.txt", index_col=0, header=0, sep='\t')
			data = data.loc[sample_ids, :]
			data = data.as_matrix()
		else:
			exit()

		skf = StratifiedKFold(n_splits=10, shuffle=True)
		for id_train_index, id_test_index in skf.split(split_sub, split_lab):

			train_index = []
			test_index = []
			for i in range(0, len(subject)):
				if subject[i] in split_sub[id_train_index]:
					train_index.append(i)
				else:
					test_index.append(i)


			x = data[train_index]
			y = allergy[train_index]
			tx = data[test_index]
			ty = allergy[test_index]

			x = (x - x.min())/(x.max() - x.min())
			#x = x.fillna(value=0)
			tx = (tx - tx.min())/(tx.max() - tx.min())
			#x = tx.fillna(value=0)

			cv_grid = [ {'C': [1, 10, 100, 1000], 'kernel': ['linear']}, {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']} ]
			clf = GridSearchCV(SVC(C=1, probability=True), param_grid=cv_grid, cv=skf, n_jobs=-1, scoring="accuracy")
			clf.fit(x, y)
			prob = [row[1] for row in clf.predict_proba(tx)]
			pred = [row for row in clf.predict(tx)]
			svm_accuracy.append(clf.score(tx,ty))

			test_data = {"ID": subject[test_index], "prob":prob, "y": ty, "pred":pred }
			test_df = pd.DataFrame(data=test_data)
			test_df = test_df.groupby("ID").median()

			prob = test_df["prob"]
			ty = test_df["y"]
			pred = [round(x) for x in (test_df["pred"]-0.10)]

			svm_roc_auc.append(roc_auc_score(ty, prob))
			svm_precision.append(precision_score(ty, pred, average='weighted'))
			svm_recall.append(recall_score(ty, pred, average='weighted'))
			svm_f_score.append(f1_score(ty, pred, average='weighted'))
			svm_pred.append(pred)
			svm_prob.append(prob)
			svm_mcc.append(matthews_corrcoef(ty, pred))

		print("Accuracy = " + str(np.mean(svm_accuracy)) + " (" + str(np.std(svm_accuracy)) + ")\n")
		print(svm_accuracy)
		print("\n\nROC AUC = " + str(np.mean(svm_roc_auc)) + " (" + str(np.std(svm_roc_auc)) + ")\n")
		print(svm_roc_auc)
		print("\n\nMCC = " + str(np.mean(svm_mcc)) + " (" + str(np.std(svm_mcc)) + ")\n")
		print(svm_mcc)
		print("\n\nPrecision = " + str(np.mean(svm_precision)) + " (" + str(np.std(svm_precision)) + ")\n")
		print("Recall = " + str(np.mean(svm_recall)) + " (" + str(np.std(svm_recall)) + ")\n")
		print("F1 = " + str(np.mean(svm_f_score)) + " (" + str(np.std(svm_f_score)) + ")\n")


		f = open('./results/'+ dataset + "_svm.txt", 'w')
		f.write("Mean Accuracy: " + str(np.mean(svm_accuracy)) + " (" + str(np.std(svm_accuracy))+ ")\n")
		f.write(str(svm_accuracy) + "\n")
		f.write("\nMean ROC: " + str(np.mean(svm_roc_auc)) + " (" + str(np.std(svm_roc_auc))+ ")\n")
		f.write(str(svm_roc_auc) + "\n")
		f.write("\nMean MCC: " + str(np.mean(svm_mcc)) + " (" + str(np.std(svm_mcc))+ ")\n")
		f.write(str(svm_mcc) + "\n")
		f.write("\nMean Precision: " + str(np.mean(svm_precision)) + " (" + str(np.std(svm_precision))+ ")\n")
		f.write(str(svm_precision) + "\n")
		f.write("\nMean Recall: " + str(np.mean(svm_recall)) + " (" + str(np.std(svm_recall))+ ")\n")
		f.write(str(svm_recall) + "\n")
		f.write("\nMean F-score: " + str(np.mean(svm_f_score)) + " (" + str(np.std(svm_f_score))+ ")\n")
		f.write(str(svm_f_score) + "\n")

		for i in range(0,10):
			f.write("\nPredictions for " + str(i) + "\n")
			f.write("\n" + str(svm_pred[i]) + "\n")
			f.write("\n" + str(svm_prob[i]) + "\n")
		f.close()   

