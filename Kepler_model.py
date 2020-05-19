from keras.layers import Dense, Flatten, Dropout, Input, Permute, CuDNNLSTM, BatchNormalization, MaxPool1D
from keras.models import Sequential, load_model, Model
from keras.layers.convolutional import (Conv1D, MaxPooling1D)
from keras.layers.merge import concatenate
from keras.utils.vis_utils import plot_model
import environment


def build_Kepler_CNN():

    # 2001/201: length of the sample
    # 1 : y coordinate

    global_view_shape = (2001, 1)
    input01 = Input(shape=global_view_shape)
    # 1st input model
    permute11 = Permute((2, 1))(input01)
    lstm11 = CuDNNLSTM(16, return_sequences=True)(permute11)
    lstm12 = CuDNNLSTM(32, return_sequences=True)(lstm11)
    lstm13 = CuDNNLSTM(64, return_sequences=True)(lstm12)
    lstm14 = CuDNNLSTM(128)(lstm13)
    out11 = Dropout(0.25)(lstm14)

    # 2nd input model
    conv21 = Conv1D(filters=16, kernel_size=11, activation='relu')(input01)
    pool21 = MaxPool1D(strides=4)(conv21)
    batch21 = BatchNormalization()(pool21)
    conv22 = Conv1D(filters=32, kernel_size=11, activation='relu')(batch21)
    pool22 = MaxPool1D(strides=4)(conv22)
    batch22 = BatchNormalization()(pool22)
    conv23 = Conv1D(filters=64, kernel_size=11, activation='relu')(batch22)
    pool23 = MaxPool1D(strides=4)(conv23)
    batch23 = BatchNormalization()(pool23)
    conv24 = Conv1D(filters=128, kernel_size=11, activation='relu')(batch23)
    pool24 = MaxPool1D(strides=4)(conv24)
    flat21 = Flatten()(pool24)
    drop21 = Dropout(0.25)(flat21)
    out21 = Dense(64, activation='relu')(drop21)

    # merge input models
    merge = concatenate([out11, out21])

    # interpretation model
    hidden1 = Dense(512, activation='relu')(merge)
    hidden1 = Dropout(0.5)(hidden1)
    hidden2 = Dense(512, activation='relu')(hidden1)
    hidden2 = Dropout(0.5)(hidden2)
    hidden3 = Dense(512, activation='relu')(hidden2)
    hidden3 = Dropout(0.5)(hidden3)
    hidden4 = Dense(512, activation='relu')(hidden3)
    hidden4 = Dropout(0.5)(hidden4)

    output = Dense(environment.NB_CLASSES, activation='softmax')(hidden4)

    model = Model(inputs=input01, outputs=output)
    # print(model.summary())
    # plot_model(model, to_file='model_plot.png',
    # show_shapes = True, show_layer_names = True)
    # summarize layers

    return model
