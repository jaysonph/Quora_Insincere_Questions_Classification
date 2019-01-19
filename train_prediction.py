import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.linear_model import LinearRegression
from attention import *
from preprocessing import *
from load_wordemb import *
from model import *


def train_pred(model, epochs=2, callback=None):
    for e in range(epochs):
        model.fit(train_X, train_y, batch_size=512, epochs=1, validation_data=(val_X, val_y))
        pred_val_y = model.predict([val_X], batch_size=1024, verbose=0)

        best_thresh = 0.5
        best_score = 0.0
        for thresh in np.arange(0.1, 0.501, 0.01):
            thresh = np.round(thresh, 2)
            score = metrics.f1_score(val_y, (pred_val_y > thresh).astype(int))
            if score > best_score:
                best_thresh = thresh
                best_score = score

        print("Val F1 Score: {:.4f}".format(best_score))

    pred_test_y = model.predict([test_X], batch_size=1024, verbose=0)
    return pred_val_y, pred_test_y, best_score
    
train_X, val_X, test_X, train_y, val_y, word_index = load_and_prec()
embedding_matrix_1 = load_glove(word_index)
embedding_matrix_2 = load_fasttext(word_index)
embedding_matrix_3 = load_para(word_index)

embedding_matrix = np.add(0.74*embedding_matrix_1, 0.26*embedding_matrix_3)


# Training and storing results of different models
outputs = []

pred_val_y, pred_test_y, best_score = train_pred(model_gru_srk_atten(embedding_matrix_1), epochs = 2)
outputs.append([pred_val_y, pred_test_y, best_score, 'gru atten srk Glove'])

pred_val_y, pred_test_y, best_score = train_pred(model_gru_srk_atten(embedding_matrix), epochs = 2)
outputs.append([pred_val_y, pred_test_y, best_score, 'gru atten srk'])

pred_val_y, pred_test_y, best_score = train_pred(model_lstm_du(embedding_matrix_1), epochs = 2)
outputs.append([pred_val_y, pred_test_y, best_score, 'LSTM DU Glove'])

pred_val_y, pred_test_y, best_score = train_pred(model_lstm_du(embedding_matrix), epochs = 2)
outputs.append([pred_val_y, pred_test_y, best_score, 'LSTM DU'])

pred_val_y, pred_test_y, best_score = train_pred(model_lstm_atten(embedding_matrix_1), epochs = 3)
outputs.append([pred_val_y, pred_test_y, best_score, '2 LSTM w/ attention GloVe'])

pred_val_y, pred_test_y, best_score = train_pred(model_lstm_atten(embedding_matrix), epochs = 3)
outputs.append([pred_val_y, pred_test_y, best_score, '2 LSTM w/ attention'])


# Blending prediction results from different models
outputs.sort(key=lambda x: x[2])
X = np.asarray([outputs[i][0] for i in range(len(outputs))])
X = X[...,0]
reg = LinearRegression().fit(X.T, val_y)
pred_val_y = np.sum([outputs[i][0] * reg.coef_[i] for i in range(len(outputs))], axis = 0)

# Searching best threshold
thresholds = []
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    res = metrics.f1_score(val_y, (pred_val_y > thresh).astype(int))
    thresholds.append([thresh, res])
    print("F1 score at threshold {0} is {1}".format(thresh, res))
    
thresholds.sort(key=lambda x: x[1], reverse=True)
best_thresh = thresholds[0][0]



pred_test_y = np.sum([outputs[i][1] * reg.coef_[i] for i in range(len(outputs))], axis = 0)
pred_test_y = (pred_test_y > best_thresh).astype(int)
test_df = pd.read_csv("../input/test.csv", usecols=["qid"])
out_df = pd.DataFrame({"qid":test_df["qid"].values})
out_df['prediction'] = pred_test_y
out_df.to_csv("submission.csv", index=False)
