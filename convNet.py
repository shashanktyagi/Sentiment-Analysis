import sys
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Convolution1D
from keras.layers import Dropout, GlobalAveragePooling1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras import optimizers
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import TensorBoard, EarlyStopping


try:
    use_gpu = sys.argv[1]
except:
    print 'Usage: python mlp.py <use_gpu>'

if use_gpu:
    print 'Using GPU.'
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    # config.allow_soft_placement = True
    set_session(tf.Session(config=config))



print 'Loading data...'
data = np.load('processed_data.npy').item()
reviews_feats = data['features']
ratings = data['ratings']


max_review_length = 500
X = sequence.pad_sequences(reviews_feats, maxlen=max_review_length)
ratings = np.array(ratings)

m,n = X.shape
print 'Total data size: {}'.format((m,n))

# split data into training, validation and test set
train_idx = np.load('train_idx.npy')
test_idx = np.load('test_idx.npy')

X_train = X[train_idx]
y_train = ratings[train_idx]
X_test = X[test_idx]
y_test = ratings[test_idx]
val_ratio = 0.1

print 'Training data size: {}'.format(X_train.shape)
print 'Test data size: {}'.format(X_test.shape)
print 'Validation ratio: {} % of training data'.format(val_ratio*100)

vocab_size = 5000
embedding_size = 32

# define model
model = Sequential()
model.add(Embedding(vocab_size, embedding_size,
                    input_length=max_review_length))
model.add(Convolution1D(64, 3, activation='relu', padding='same'))
model.add(Convolution1D(64, 3, activation='relu', padding='same'))
model.add(MaxPooling1D(3))
model.add(Convolution1D(128, 3, activation='relu', padding='same'))
model.add(Convolution1D(128, 3, activation='relu', padding='same'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.2))
model.add(Dense(1, activation=None))

optim = optimizers.Adam(lr=0.001,
                        decay=0.001)
model.compile(loss='mse',
              optimizer='adam',
              metrics = ['mse'])
tensorboard = TensorBoard(log_dir='./logs', write_graph=True)
earlystopping = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=2,
                              verbose=1,
                              mode='auto')


model.fit(X_train, y_train,
          batch_size=64,
          epochs=20,
          callbacks=[tensorboard, earlystopping],
          validation_split=val_ratio,
          shuffle=True,
          verbose=1)

results = model.evaluate(X_test, y_test, verbose=0)
print 'Test RMSE: {}'.format(results[0]**0.5)


