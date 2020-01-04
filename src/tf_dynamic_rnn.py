""" Dynamic Recurrent Neural Network.

TensorFlow implementation of a Recurrent Neural Network (LSTM) that performs
dynamic computation over sequences with variable length. This example is using
a toy dataset to classify linear sequences. The generated sequences have
variable length.

Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function
from numpy import random
random.seed(1)
from tensorflow import set_random_seed
set_random_seed(0)
import tensorflow as tf
import utils
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import roc_auc_score, matthews_corrcoef
import Ordered_Neurons_LSTM as oredered_lstm
from sklearn.model_selection import  StratifiedKFold


# ====================
#  TOY DATA GENERATOR
# ====================
class Time_series_data(object):
    """ Generate sequence of data with dynamic length.
    This class generate samples for training:
    - Class 0: linear sequences (i.e. [0, 1, 2, 3,...])
    - Class 1: random sequences (i.e. [1, 3, 10, 7,...])

    NOTICE:
    We have to pad each sequence to reach 'max_seq_len' for TensorFlow
    consistency (we cannot feed a numpy array with inconsistent
    dimensions). The dynamic calculation will then be perform thanks to
    'seqlen' attribute that records every actual sequence length.
    """

    def __init__(self, data, time_points, subjects, max_seq_len, meta, num_features):
        self.data = data
        self.time_points = time_points
        self.subjects = subjects
        self.meta = meta
        self.max_seq_len = max_seq_len
        self.num_features = num_features
        self.current_batch_index = 0
        self.batch_index = 0

    def next(self, batch_size=1):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        batch_data = []
        batch_labels = []

        seq_len = []

        #  reset the index in the new epoch
        self.batch_index = min(self.batch_index + batch_size, len(self.subjects))
        while self.current_batch_index < self.batch_index:
            s = self.subjects[self.current_batch_index]
            s_samples = self.time_points[s]
            data_samples = data.loc[s_samples, :]
            e = data_samples.values.tolist()
            e += [[0.] * self.num_features for i in range(self.max_seq_len - len(s_samples))]
            batch_data.append(e)
            seq_len.append(len(s_samples))
            r = self.meta.loc[self.meta["subjectID"] == s, "allergy"]
            for target in np.unique(r.values):
                if target:
                    batch_labels.append([1, 0])
                else:
                    batch_labels.append([0, 1])
            self.current_batch_index = self.current_batch_index + 1

        return batch_data, batch_labels, seq_len

    def have_next(self):
        return self.batch_index != len(self.subjects)


parser = argparse.ArgumentParser(description='RF on data')

parser.add_argument("--data", help="raw or latent-40 or deep_forest or mrmr or lasso or rfe")
parser.add_argument("--cell", help="lstm or gru or on-lstm", default='lstm')

args = parser.parse_args()

if __name__ == '__main__':
    if args.data == None:
        print("Please specify raw or latent for data flag")
    else:
        dataset = args.data
        cell_type = args.cell
    # ==========
    #   MODEL
    # ==========

    # Parameters
    learning_rate = 0.001
    training_steps = 100
    batch_size = 5
    display_step = 10
    lambda_l2=0.01


    # Network Parameters
    n_hidden = 64  # hidden layer num of features
    n_classes = 2  # linear sequence or not

    InputFile = '../data/diabimmune_karelia_metaphlan_table.txt'
    MetadataFile = '../data/metadata.csv'



    seq_max_len, num_features, subjects, meta_file, time_points, data = utils.lstm_raw_input(MetadataFile, InputFile)

    sample_ids = []
    for suject_id in subjects:
        sample_ids += time_points.get(suject_id)

    if dataset == "latent-40":
        data = pd.read_csv("./feature_selection/latent40.txt", index_col=0, header=None, sep='\t')
        data = data[data.columns[:-1]]
        data = data.loc[sample_ids, :]
        num_features = 40
    elif dataset == "raw":
        data = data.transpose()
    elif dataset == "deep_forest":
        data = pd.read_csv("./feature_selection/deep_forest.txt", index_col=0, header=0, sep='\t')
        data = data.loc[sample_ids, :]
        num_features = 40
    elif dataset == "lasso":
        data = pd.read_csv("./feature_selection/lasso.txt", index_col=0, header=0, sep='\t')
        data = data.loc[sample_ids, :]
        num_features = 40
    elif dataset == "mrmr":
        data = pd.read_csv("./feature_selection/lasso.txt", index_col=0, header=0, sep='\t')
        data = data.loc[sample_ids, :]
        num_features = 40
    elif dataset == "rfe":
        data = pd.read_csv("./feature_selection/rfe.txt", index_col=0, header=0, sep='\t')
        data = data.loc[sample_ids, :]
        num_features = 40
    else:
        exit()

    subjects_labels = []
    for subject_id in subjects:
        tmp = meta_file[meta_file['subjectID']==subject_id]
        tmp = list(set(tmp['allergy'].values))
        if len(tmp) >1:
            raise Exception('the subject with different labels')
        if tmp[0]:
            subjects_labels.append(1)
        else:
            subjects_labels.append(0)


    # tf Graph input
    x = tf.placeholder("float", [None, seq_max_len, num_features])
    y = tf.placeholder("float", [None, n_classes])
    # A placeholder for indicating each sequence length
    seqlen = tf.placeholder(tf.int32, [None])

    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }


    def dynamicRNN(x, seqlen, weights, biases):
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.unstack(x, seq_max_len, 1)

        # Define a lstm cell with tensorflow
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
        if cell_type == 'gru':
            lstm_cell = tf.contrib.rnn.GRUCell(n_hidden)
        elif cell_type == 'on-lstm':
            lstm_cell = oredered_lstm.ON_LSTM(num_units=n_hidden, chunk_size=2)
        else:
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)

        # Get lstm cell output, providing 'sequence_length' will perform dynamic
        # calculation.
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32,
                                                    sequence_length=seqlen)

        # When performing dynamic calculation, we must retrieve the last
        # dynamically computed output, i.e., if a sequence length is 10, we need
        # to retrieve the 10th output.
        # However TensorFlow doesn't support advanced indexing yet, so we build
        # a custom op that for each sample in batch size, get its length and
        # get the corresponding relevant output.

        # 'outputs' is a list of output at every timestep, we pack them in a Tensor
        # and change back dimension to [batch_size, n_step, n_input]
        outputs = tf.stack(outputs)
        outputs = tf.transpose(outputs, [1, 0, 2])

        # Hack to build the indexing and retrieve the right output.
        batch_size = tf.shape(outputs)[0]
        # Start indices for each sample
        index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
        # Indexing
        outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

        # Linear activation, using outputs computed above
        return tf.matmul(outputs, weights['out']) + biases['out']


    pred = dynamicRNN(x, seqlen, weights, biases)


    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))

    # Loss function using L2 Regularization
    tv = tf.trainable_variables()
    regularizer = tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])
    cost = tf.reduce_mean(cost + lambda_l2 * regularizer)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    # Evaluate model
    y_pred = tf.argmax(pred, 1)
    y_true = tf.argmax(y, 1)
    correct_pred = tf.equal(y_pred, y_true)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


    # Start training
    with tf.Session() as sess:
        # Run the initializer
        sess.run(tf.global_variables_initializer())
        skf = StratifiedKFold(n_splits=10, shuffle=True)
        acc_list = []
        auc_list = []
        mcc_list = []
        for id_train_index, id_test_index in skf.split(subjects, subjects_labels):
            train_subjects = [subjects[train_idx] for train_idx in id_train_index]
            test_subjects = [subjects[test_idx] for test_idx in id_test_index]
            train = Time_series_data(data, time_points, train_subjects, seq_max_len, meta_file, num_features)
            test = Time_series_data(data, time_points, test_subjects, seq_max_len, meta_file, num_features)
            for step in range(1, training_steps + 1):
                while train.have_next():
                    batch_x, batch_y, batch_seqlen = train.next(batch_size)
                    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
                    acc, loss = sess.run([accuracy, cost], feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
                    print(" Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
            validate_data, validate_label, seq_len = test.next(len(test.subjects))
            acc, loss, y_p, y_t = sess.run([accuracy, cost, y_pred, y_true], feed_dict={x: validate_data, y: validate_label, seqlen: seq_len})
            auc = roc_auc_score(y_t, y_p)
            print("validate loss="+str(loss)+",validate acc="+str(acc) +",validate auc="+str(auc))
            mcc = matthews_corrcoef(y_t, y_p)

            acc_list.append(acc)
            auc_list.append(auc)
            mcc_list.append(mcc)

        f = open("./results/" + dataset + "_" + cell_type + ".txt", 'w')
        f.write("\nMean Accuracy: " + str(np.mean(acc_list)) + " (" + str(
            np.std(acc_list)) + ")\n")
        f.write(str(acc_list) + "\n")
        f.write("\nMean ROC: " + str(np.mean(auc_list)) + " (" + str(np.std(auc_list)) + ")\n")
        f.write(str(auc_list) + "\n")
        f.write("\nMCC: " + str(np.mean(mcc_list)) + " (" + str(np.std(mcc_list)) + ")\n")
        f.write(str(mcc_list) + "\n")
        f.close()
