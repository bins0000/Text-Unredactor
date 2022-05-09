import glob
import io
import os
import pdb
import sys
import re

import nltk
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
from nltk import ne_chunk

import numpy as np
import pandas as pd

import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import ensemble
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def tfidf(dataset):
    documents = []
    
    # this way is for reading the .txt files from IMDb dataset
    '''
    for thefile in glob.glob(filepath)[:50]:
        with io.open(thefile, 'r', encoding='utf-8') as fyl:
            text = fyl.read()
            documents.append(text)
    '''

    # this way is for the collaborated unredactor.tsv
    tfidf_vectorizer = TfidfVectorizer()
    x = tfidf_vectorizer.fit_transform(dataset)
    idf = tfidf_vectorizer.idf_
    tfidf_dict = dict(zip(tfidf_vectorizer.get_feature_names_out(), idf))
    return tfidf_dict


def get_features(text):
    block = '\u2588'
    features_list = []
    l = len(text)
    left_of_name = r'(\w*|\W)\s*'+ block + r'+\s*' + block + r'+\s*' + block + r'+\s*' + block + r'+\s*'+ block + r'+'
    name = block + r'+\s*' + block + r'+\s*' + block + r'+\s*' + block + r'+\s*' + block + r'+'
    right_of_name = block + r'+\s*' + block + r'+\s*' + block + r'+\s*' + block + r'+\s*' + block +r'+\s*(\W{0,1}\w*)'
    name_redacted = re.findall(name, text)
    left = re.findall(left_of_name, text)
    right = re.findall(right_of_name, text)
    if len(name_redacted) == 0:
        
        '''
        index = collab_train_x.index(text)
        collab_train_x.pop(index)
        collab_train_y.pop(index)
        '''
        feature_dict = {}
        feature_dict['error'] = 1
        feature_dict['name_length'] = 0
        feature_dict['spaces'] = 0
        feature_dict['left_name'] = 0
        feature_dict['right_name'] = 0
        feature_dict['len_chars'] = l
        features_list.append(feature_dict)

    else:
        for i in range(len(name_redacted)):
            feature_dict = {}

            if left[i] in tfidf_dict.keys():
                tfidf_left = tfidf_dict[left[i]]
            else:
                tfidf_left = 0

            if right[i] in tfidf_dict.keys():
                tfidf_right = tfidf_dict[right[i]]
            else:
                tfidf_right = 0

            feature_dict['name_length'] = len(name_redacted[i])
            feature_dict['spaces'] = name_redacted[i].count(' ')
            feature_dict['left_name'] = tfidf_left
            feature_dict['right_name'] = tfidf_right
            feature_dict['len_chars'] = l
            feature_dict['error'] = l
            features_list.append(feature_dict)

    return features_list

def main():

    # train  
    train_data = []
    for text in collab_train_x[:850]:
        #print(i)
        feature = get_features(text)
        train_data.extend(feature)
    #print(len(train_data))
    
    #print(collab_train_x[1877])

        
    dict_vectorizer = DictVectorizer()
    train_features = dict_vectorizer.fit_transform(train_data).toarray()
    train_labels = np.asarray(collab_train_y[:len(train_data)])
    RFmodel = ensemble.RandomForestClassifier()
    RFmodel.fit(train_features, train_labels[:len(train_data)])


    # test
    
    test_predict = []
    for text in collab_test_x:
        test_data = get_features(text)
        if len(test_data) > 0:
            test_features = dict_vectorizer.fit_transform(test_data).toarray()
            prediction = RFmodel.predict(test_features)
            test_predict.append(prediction[0])

    
    #print(test_predict)
    #print('\n\n')
    #print(collab_test_y)
    data_tuples = list(zip(test_predict,collab_test_x))
    print(pd.DataFrame(data_tuples, columns=['Predicted Names','Redacted Text']))

    
    # computing scores
    ## Accuracy, Precision, Recall
    #accuracy = metrics.accuracy_score(collab_test_y, test_predict)

    #print("Accuracy: ",round(accuracy,2))
    print("Scores: ")
    print(metrics.classification_report(collab_test_y, test_predict))




    
if __name__ == "__main__":
    block = '\u2588'
    # get the data from unredactor.tsv
    collab_train_x = []
    collab_train_y = []
    collab_test_x = []
    collab_test_y = []
    collab_val_x = []
    collab_val_y = []
    with open ('unredactor.tsv', 'r', encoding = "utf-8") as f:
        data_rows = f.readlines()
        for row in data_rows:
            dataInRow = row.split('\t')
            if 'train' in dataInRow[1]:
                collab_train_x.append(dataInRow[3])
                collab_train_y.append(dataInRow[2])
            elif 'test' in dataInRow[1]:
                collab_test_x.append(dataInRow[3])
                collab_test_y.append(dataInRow[2])
            elif 'validation' in dataInRow[1]:
                collab_val_x.append(dataInRow[3])
                collab_val_y.append(dataInRow[2])
    
    # get rid of one data that has double redaction!!!!
    collab_train_y.pop(1877)
    collab_train_x.pop(1877)

    #print('train x = ', len(collab_train_x))
    #print('train y = ', len(collab_train_y))
    #print('test = ',len(collab_test_x))
    #print('val = ',len(collab_val_x))
    #tfidf_dict = train_tfidf("training_data/*.txt")
    
    # get tfidf vector
    tfidf_dict = tfidf(collab_train_x + collab_val_x)
    
    # call main
    main()
