import tensorflow as tf
from functools import reduce
import numpy as np
import src.utils as utils
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

np.random.seed(1)


class Batch_data_generator(object):

    def __init__(self, data):
        self.data = shuffle(data)
        self.current_batch_index = 0
        self.batch_index = 0

    def next(self, batch_size=16):
        batch_data = []
        #  reset the index in the new epoch
        self.batch_index = min(self.batch_index + batch_size, len(self.data))
        while self.current_batch_index < self.batch_index:
            batch_data.append(data.iloc[self.current_batch_index])
            self.current_batch_index = self.current_batch_index + 1
        return batch_data

    def have_next(self):
        return self.batch_index != len(self.data)

def weight_variable(shape, name):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1), name)


def bias_variable(shape, name):
    return tf.Variable(tf.zeros(shape=shape), name)


def build_sae(num_input, num_hidden_1, num_hidden_2):
    W_e_1 = weight_variable([num_input, num_hidden_1], "w_e_1")
    b_e_1 = bias_variable([num_hidden_1], "b_e_1")
    h_e_1 = tf.nn.relu(tf.add(tf.matmul(x, W_e_1), b_e_1))

    W_e_2 = weight_variable([num_hidden_1, num_hidden_2], "w_e_2")
    b_e_2 = bias_variable([num_hidden_2], "b_e_2")
    h_e_2 = tf.nn.relu(tf.add(tf.matmul(h_e_1, W_e_2), b_e_2))

    W_d_1 = weight_variable([num_hidden_2, num_hidden_1], "w_d_1")
    b_d_1 = bias_variable([num_hidden_1], "b_d_1")
    h_d_1 = tf.nn.relu(tf.add(tf.matmul(h_e_2, W_d_1), b_d_1))

    W_d_2 = weight_variable([num_hidden_1, num_input], "w_d_2")
    b_d_2 = bias_variable([num_input], "b_d_2")
    h_d_2 = tf.nn.sigmoid(tf.add(tf.matmul(h_d_1, W_d_2), b_d_2))

    return [h_e_1, h_e_2], [W_e_1, W_e_2, W_d_1, W_d_2], h_d_2


def kl_div(rho, rho_hat):
    kl_1 = rho * tf.log(rho) - rho * tf.log(rho_hat) + (1 - rho) * tf.log(1 - rho) - (1 - rho) * tf.log(1 - rho_hat)
    return kl_1


def next_batch(num, data):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data.iloc[i] for i in idx]

    return np.asarray(data_shuffle)

def split_dataset(data):
    test_split_index = int(0.9 * len(data))
    x_train, x_test = data[0:test_split_index], data[test_split_index:]
    validate_split_index = int(0.9 * len(x_train))
    x_train, x_validate = x_train[0:validate_split_index], x_train[validate_split_index:]

    return x_train, x_validate, x_test

if __name__ == '__main__':

    InputFile = '../data/diabimmune_karelia_metaphlan_table.txt'
    MetadataFile = '../data/metadata.csv'
    seq_max_len, num_features, subjects, meta_file, time_points, data = utils.lstm_raw_input(MetadataFile, InputFile)
    data = data.T
    train, validate, test = split_dataset(data)


    learning_rate = 1e-4
    alpha = 12-3
    beta = 3
    epoch = 5

    batch_size = 32




    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 8), sharey=True, sharex=True)
    neurons = [[128, 64], [128, 32], [128, 16], [64, 32], [64, 16], [32, 16]]
    for i in range(len(neurons)):

        tf.reset_default_graph()
        sess = tf.InteractiveSession()
        x = tf.placeholder(tf.float32, shape=[None, num_features])

        first_hidden, second_hidden = neurons[i]

        h, w, x_reconstruct = build_sae(num_features, first_hidden, second_hidden)

        kl_div_loss = reduce(lambda x, y: x + y, map(lambda x: tf.reduce_sum(kl_div(0.01, tf.reduce_mean(x, 0))), h))
        # kl_div_loss = tf.reduce_sum(kl_div(0.02, tf.reduce_mean(h[0],0)))
        l2_loss = reduce(lambda x, y: x + y, map(lambda x: tf.nn.l2_loss(x), w))

        # loss = tf.reduce_mean(tf.pow(x_reconstruct - x, 2)) + alpha * l2_loss + beta * kl_div_loss
        loss = tf.reduce_mean(tf.pow(x_reconstruct - x, 2))+ alpha * l2_loss
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


        train_loss_list = []
        validate_loss_list = []

        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for k in range(epoch):
            genenator = Batch_data_generator(data)
            train_loss =0
            batch_idx = 0
            while genenator.have_next():
                batch_idx = batch_idx + 1
                batch = genenator.next(batch_size)
                _,train_loss = sess.run([optimizer,loss], feed_dict={x: batch})
                train_loss += train_loss
            train_loss_list.append(train_loss / batch_idx)
            _, validate_loss = sess.run([optimizer, loss], feed_dict={x: validate})
            validate_loss_list.append(validate_loss)

        _, test_loss = sess.run([optimizer, loss], feed_dict={x: test})
        print("test loss ===" + str(test_loss) + "========================")

        row_idx, col_idx = i // 3, i % 3
        ax[row_idx,col_idx].plot(train_loss_list,color='black')
        ax[row_idx,col_idx].plot(validate_loss_list,color='red')

        ax[row_idx,col_idx].set_ylabel('Loss')
        ax[row_idx,col_idx].set_xlabel('Epoch')
        ax[row_idx,col_idx].legend(['training_loss', 'validation_loss'])
        ax[row_idx, col_idx].set_title(str(first_hidden)+"X"+str(second_hidden)+"X"+str(first_hidden))

    plt.subplots_adjust(hspace=0.3, wspace=0.5)
    plt.savefig('tf_auto_encoder.png', bbox_inches='tight')
    plt.show()
