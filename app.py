import gdown
import tensorflow as tf
from modules import data_gen as dg
from modules import model as mod
from modules import training as tr
import keras
import numpy as np

# To download and extract data from a zip file
#url = 'https://drive.google.com/uc?id=1YlvpDLix3S-U8fd-gqRwPcWXAXm8JwjL'
#output = 'data.zip'
#gdown.download(url, output, quiet=False)
#gdown.extractall('data.zip')

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]

char_to_num = keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

# To process our data

data = tf.data.Dataset.list_files('./data/s1/*.mpg')
data = data.shuffle(500,reshuffle_each_iteration=False)
data = data.map(dg.mappable_func)
data = data.padded_batch(2,padded_shapes=([75,None,None,None],[40]))
data = data.prefetch(tf.data.AUTOTUNE)
train = data.take(450)
test = data.skip(450)

#frames ,alignments = data.as_numpy_iterator().next()


# create our deep learning model
model = mod.mod_create()

# Training our model
tr_model = tr.comp(model,train,test)


model.load_weights('models/checkpoint')

# test on video

sample = dg.load_data(tf.convert_to_tensor('.\\data\\s1\\bbaf3s.mpg'))



yhat = model.predict(tf.expand_dims(sample[0],axis=0))
decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()
print('~'*100, 'PREDICTIONS')
print([tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in decoded])

