import sys
import numpy as np
import tensorflow as tf

from keras.models import Model
from keras.layers import Input, Dense, Convolution1D, Flatten, merge
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
user_item = np.load('user_item.npy')

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
user_item_train = user_item[train_idx]
y_train = ratings[train_idx]

X_test = X[test_idx]
user_item_test = user_item[test_idx]
y_test = ratings[test_idx]
val_ratio = 0.1

print 'Training data size: {}'.format(X_train.shape)
print 'Test data size: {}'.format(X_test.shape)
print 'Validation ratio: {} % of training data'.format(val_ratio*100)

num_users = 123960
num_items = 50052
latent_size = 20
vocab_size = 5000
embedding_size = 32

# define model
feat = Input(shape=(n,))
x = Embedding(vocab_size, embedding_size,
              input_length=max_review_length)(feat)
x = Convolution1D(64, 3, activation='relu', padding='same')(x)
x = Convolution1D(64, 3, activation='relu', padding='same')(x)
x = MaxPooling1D(3)(x)
x = Convolution1D(128, 3, activation='relu', padding='same')(x)
x = Convolution1D(128, 3, activation='relu', padding='same')(x)
x = GlobalAveragePooling1D()(x)
x = Dropout(0.2)(x)

u = Input(shape=(1,))
i = Input(shape=(1,))
u_latent = Embedding(num_users, latent_size, input_length=1)(u)
u_latent = Flatten()(u_latent)
i_latent = Embedding(num_items, latent_size, input_length=1)(i)
i_latent = Flatten()(i_latent)
mf = merge([u_latent, i_latent], mode='mul')

merged = merge([x, mf], mode='concat')
pred = Dense(1, activation=None)(merged)

model = Model(input=[feat, u, i],
              output=pred)

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


model.fit([X_train, user_item_train[:,0], user_item_train[:,1]],
          y_train,
          batch_size=64,
          epochs=20,
          callbacks=[tensorboard, earlystopping],
          validation_split=val_ratio,
          shuffle=True,
          verbose=1)

results = model.evaluate([X_test, user_item_test[:,0], user_item_test[:,1]],
                         y_test, verbose=0)
print 'Test RMSE: {}'.format(results[0]**0.5)


