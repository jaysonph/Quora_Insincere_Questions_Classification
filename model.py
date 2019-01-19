import keras
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate, Lambda
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from keras.models import Model


def model_lstm_du(embedding_matrix):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.1)(x)
    x1 = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    x2 = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    x = concatenate([x1,x2])
    x = SpatialDropout1D(0.1)(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    conc = Dense(64, activation="relu")(conc)
    conc = Dropout(0.1)(conc)
    outp = Dense(1, activation="sigmoid")(conc)
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
    return model
    
def model_lstm_atten(embedding_matrix):
    inp = Input(shape=(maxlen,))       # input sentence
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)  # max_features=95000, embed_size=300
    x = SpatialDropout1D(0.1)(x)
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
    x = SpatialDropout1D(0.1)(x)
    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    x = SpatialDropout1D(0.1)(x)
    x = Attention(maxlen)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
    return model
    
def model_gru_srk_atten(embedding_matrix):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.1)(x)
    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    x = Attention(maxlen)(x) # New
    x = Dense(16, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
    return model 
