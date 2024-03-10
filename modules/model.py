import keras
from keras.models import Sequential 
from keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = keras.layers.StringLookup(vocabulary=vocab, oov_token="")

def mod_create():
    model = Sequential()
    model.add(Conv3D(128,3,input_shape = (75,46,140,1),padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(Conv3D(256,3,padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(Conv3D(75,3,padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(LSTM(128,kernel_initializer='Orthogonal',return_sequences=True)))
    model.add(Dropout(0.5))

    model.add(Bidirectional(LSTM(128,kernel_initializer='Orthogonal',return_sequences=True)))
    model.add(Dropout(0.5))

    model.add(Dense(char_to_num.vocabulary_size()+1,kernel_initializer='he_normal',activation='softmax'))

    return model


