import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)
from tensorflow import set_random_seed
set_random_seed(1)
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.utils import shuffle
from keras.optimizers import Adam
from keras.optimizers import Adadelta
from keras import regularizers
from sklearn.metrics import mean_squared_error
from keras.utils import plot_model
import src.utils as utils
from keras import backend as K
import pandas as pd



InputFile = '../data/diabimmune_karelia_metaphlan_table.txt'
MetadataFile = '../data/metadata.csv'
_, num_features, subjects, meta_file, time_points, data = utils.lstm_raw_input(MetadataFile, InputFile)

data = data.T
data = shuffle(data)
(x_train,y_train),(x_validate,y_validate),(x_test,y_test),\
(x_train_ids,y_validate_ids,y_test_ids) = utils.split_dataset(subjects, data, time_points,meta_file)

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_validate = x_validate.reshape((len(x_validate), np.prod(x_validate.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

neurons = [[128, 64],[128, 50],[128, 40],[128, 32],[128, 25],[128, 16]]
# neurons = [[128, 40]]

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 8), sharey=True, sharex=True)
for i in range(len(neurons)):
    first_hidden, second_hidden = neurons[i]
    input = Input(shape=(num_features,))
    encoder = Dense(first_hidden, activation='relu')(input)
    encoder = Dense(second_hidden, activation='relu', name='latent')(encoder)
    decoder = Dense(first_hidden, activation='relu')(encoder)
    decoder = Dense(num_features, activation='relu')(decoder)

    autoencoder = Model(input, decoder)

    plot_model(autoencoder, to_file='keras_ae_model.png', show_shapes=True)

    optimizer = Adam(lr=0.001)
    autoencoder.compile(optimizer=optimizer, loss='mse')
    history = autoencoder.fit(x_train, x_train,
                              epochs=100,
                              batch_size=8,
                              shuffle=True,
                              validation_data=(x_validate, x_validate))

    if second_hidden == 40:
        latent_layer_model = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('latent').output)
        df = np.concatenate([x_train, x_validate, x_test])
        latent_layer_model_output = latent_layer_model.predict(df)
        labels = np.concatenate([y_train, y_validate, y_test])
        ids = x_train_ids + y_validate_ids + y_test_ids

        latent = pd.DataFrame(latent_layer_model_output, index=ids)
        latent['label'] = labels

        latent.to_csv('latent40.txt', header=False, sep='\t')

    row_idx, col_idx = i // 3, i % 3
    ax[row_idx, col_idx].plot(history.history['loss'], color='black')
    ax[row_idx, col_idx].plot(history.history['val_loss'], color='red')

    # ax[row_idx, col_idx].set_ylim([0,0.1])
    ax[row_idx, col_idx].set_ylabel('Loss')
    ax[row_idx, col_idx].set_xlabel('Epoch')
    ax[row_idx, col_idx].legend(['training_loss', 'validation_loss'])
    ax[row_idx, col_idx].set_title(str(first_hidden) + "X" + str(second_hidden) + "X" + str(first_hidden))
plt.subplots_adjust(hspace=0.3, wspace=0.5)
plt.savefig('auto_encoder.png', bbox_inches='tight')
plt.show()
