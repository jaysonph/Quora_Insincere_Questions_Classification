import numpy as np
import pandas as pd
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# Function to convert to lowercase and standardize the format of all punction space
def clean_txt(train,test):
    train["question_text"] = train["question_text"].str.lower()
    test["question_text"] = test["question_text"].str.lower()

    puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
     '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
     '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
     '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
     '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]
    def clean_text(x):

        x = str(x)
        for punct in puncts:
            x = x.replace(punct, f' {punct} ')
        return x


    train["question_text"] = train["question_text"].apply(lambda x: clean_text(x))
    test["question_text"] = test["question_text"].apply(lambda x: clean_text(x))
    return train, test


def load_and_prec():
    train_df = pd.read_csv("../input/train.csv")
    test_df = pd.read_csv("../input/test.csv")
    print("Train shape : ",train_df.shape)
    print("Test shape : ",test_df.shape)
    
    train_df, test_df = clean_txt(train_df, test_df)
    
    ## fill up the missing values
    train_X_orig = train_df["question_text"].fillna("_##_").values
    test_X = test_df["question_text"].fillna("_##_").values

    ## Tokenize the sentences
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(train_X_orig))
    train_X_orig = tokenizer.texts_to_sequences(train_X_orig)
    test_X = tokenizer.texts_to_sequences(test_X)

    ## Pad the sentences 
    train_X_orig = pad_sequences(train_X_orig, maxlen=maxlen)
    test_X = pad_sequences(test_X, maxlen=maxlen)

    ## Get the target values
    train_y_orig = train_df['target'].values 
    
    #shuffling the data
    DATA_SPLIT_SEED = 2018
    splits = list(StratifiedKFold(n_splits=12, shuffle=True, random_state=DATA_SPLIT_SEED).split(train_X_orig, train_y_orig))
    for idx, (train_idx, valid_idx) in enumerate(splits):
        if idx == 5:
            train_X = train_X_orig[train_idx]
            train_y = train_y_orig[train_idx]
            val_X = train_X_orig[valid_idx]
            val_y = train_y_orig[valid_idx]
        else:
            continue
    
    return train_X, val_X, test_X, train_y, val_y, tokenizer.word_index
