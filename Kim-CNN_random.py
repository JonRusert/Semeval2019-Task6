#Kim-CNN_random.py
#Author: Jon Rusert
#Example Use: python3 Kim-CNN_random.py trainData testData outputFile

import warnings
import json
from operator import itemgetter
import time


import io


import random
from os import walk

from keras.preprocessing import text

global clf
import sys
from keras.models import model_from_json
import csv
import io
from keras.preprocessing.text import Tokenizer,one_hot
from keras.preprocessing.sequence import pad_sequences

from keras.models import Model,Sequential
from keras.layers import Input
from keras.layers import Dense,Reshape,Concatenate, Merge
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding,LSTM
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils.np_utils import to_categorical
import numpy as np
import os
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from datetime import datetime
import nltk



# load a clean dataset
def load_dataset(trFile, teFile):
    labelsAsNums = {}
    numsAsLabels = {}
    labelNum = 0
    numTweets = 0
    tmpTweets = []
    testTweets = []

    x_train = []
    y_train = []
    x_test = []



    #load in train tweets and corresponding labels
    with open(trFile, 'r') as csvfile:
        tweetreader = csv.reader(csvfile, delimiter='\t')
        for tweet in tweetreader:
            text = tweet[1].lower().strip()
            x_train.append(' '.join(nltk.word_tokenize(text)))
            if tweet[2] not in labelsAsNums:
                labelsAsNums[tweet[2]] = labelNum
                numsAsLabels[labelNum] = tweet[2]
                labelNum += 1
            y_train.append(labelsAsNums[tweet[2]])
            

    #load in test tweets and corresponding labels
    with open(teFile, 'r') as csvfile:
        tweetreader = csv.reader(csvfile, delimiter='\t')
        for tweet in tweetreader:
            testTweets.append(tweet)
            text = tweet[1].lower().strip()
            text = " ".join(nltk.word_tokenize(text))
            x_test.append(text)
            


    print(max(y_train))
    return x_train, y_train, x_test, labelNum, testTweets, labelsAsNums, numsAsLabels


#train word embeddings
def train_word_embeddings():
    pass


# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def create_tokenizer_char(lines):

    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(lines)
    return tokenizer


def fill_in_missing_words_with_zeros(embeddings_index, word_index, EMBEDDING_DIM):
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def fill_in_missing_words_with_random(embeddings_index, word_index, EMBEDDING_DIM):
    embedding_matrix = np.random.random_sample((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


# calculate the maximum document length
def max_length(lines):
    return max([len(s.split()) for s in lines])


def load_model(filename, embDim):
    embeddings_index = {}
    f = open(filename, encoding='utf-8')
    for line in f:
        line
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
            if len(coefs) == embDim:
                embeddings_index[word] = coefs
        except:
            # print(values)
            c=1
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index


# encode a list of lines
def encode_text(tokenizer, lines, length):
    # integer encode
    encoded = tokenizer.texts_to_sequences(lines)
    # pad encoded sequences
    padded = pad_sequences(encoded, maxlen=length, padding='post')
    return padded


def define_model_CNN_word_word(length, vocab_size,embedding_matrix,word_index,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH, numClasses):

 
    conv1 = Sequential()
    conv1.add(Embedding(len(word_index) + 1,
                         EMBEDDING_DIM,
                         weights=[embedding_matrix],
                         input_length=MAX_SEQUENCE_LENGTH,
                         trainable=False))
    conv1.add(Reshape((MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, 1)))
    conv1.add(Conv2D(100, (3, EMBEDDING_DIM), activation='relu'))
    conv1.add(MaxPooling2D((MAX_SEQUENCE_LENGTH-3+1,1)))

    conv2 = Sequential()
    conv2.add(Embedding(len(word_index) + 1,
                         EMBEDDING_DIM,
                         weights=[embedding_matrix],
                         input_length=MAX_SEQUENCE_LENGTH,
                         trainable=False))
    conv2.add(Reshape((MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, 1)))
    conv2.add(Conv2D(100, (4, EMBEDDING_DIM), activation='relu'))
    conv2.add(MaxPooling2D((MAX_SEQUENCE_LENGTH -4 + 1,1)))

    conv3 = Sequential()
    conv3.add(Embedding(len(word_index) + 1,
                         EMBEDDING_DIM,
                         weights=[embedding_matrix],
                         input_length=MAX_SEQUENCE_LENGTH,
                         trainable=False))
    conv3.add(Reshape((MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, 1)))
    conv3.add(Conv2D(100, (5, EMBEDDING_DIM), activation='relu'))
    conv3.add(MaxPooling2D((MAX_SEQUENCE_LENGTH-5 + 1, 1)))
    
    model = Sequential() 
    model.add(Merge([conv1, conv2, conv3], mode = 'concat', concat_axis=-1))

    model.add(Flatten())

    model.add(Dense(128, activation = 'relu'))

    model.add(Dropout(0.5))

    model.add(Dense(numClasses, activation = 'softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    model.summary()

    return model



def CNN_word_word(outFile):
    # load training dataset

    trainLines, trainLabels, testLines, numClasses, testTweets, labelsAsNums, numsAsLabels = load_dataset(trainFile, testFile)
    print('nc', numClasses)
    # create tokenizer
    tokenizer = create_tokenizer(trainLines)
    sequences = tokenizer.texts_to_sequences(trainLines)
    test_sequences = tokenizer.texts_to_sequences(testLines)

    # calculate max document length
    MAX_SEQUENCE_LENGTH = max_length(trainLines)

    # calculate vocabulary size
    vocab_size = len(tokenizer.word_index) + 1
    print('Max document length: %d' % MAX_SEQUENCE_LENGTH)
    print('Vocabulary size: %d' % vocab_size)

    # encode data
    # trainX = encode_text(tokenizer, trainLines, MAX_SEQUENCE_LENGTH)
    # print(trainX.shape)

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    x_test = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    # labels = np.asarray(trainLabels)
    labels = to_categorical(trainLabels, num_classes=numClasses)
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    # split the data into a training set and a validation set
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(0.10 * data.shape[0])  # validation set 10 percent
    print(nb_validation_samples)
    x_train = data
    y_train = labels

    # x_val = data[-nb_validation_samples:]
    # y_val = labels[-nb_validation_samples:]

    #print(x_train.shape)
    #print(y_train.shape)
    # print(x_val.shape)
    # print(y_val.shape)


    EMBEDDING_DIM = 100
    word_index = tokenizer.word_index
    print("loading GLOVE model")
    embedding_matrix = load_model('glove/glove.twitter.27B.100d.txt', EMBEDDING_DIM)
    print("Filling non existing words")
    embedding_matrix = fill_in_missing_words_with_random(embedding_matrix, tokenizer.word_index, EMBEDDING_DIM)
    # embedding_matrix = fill_in_missing_words_with_zeros({}, tokenizer.word_index, EMBEDDING_DIM)

    # define model
    model = define_model_CNN_word_word(MAX_SEQUENCE_LENGTH, vocab_size,embedding_matrix,word_index,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH, numClasses)

    model.fit( [x_train, x_train, x_train], y_train, 
              epochs=30, batch_size=50, verbose =0)
    #print(model.summary())
    # save the model
    # model.model.save()

    #loss, acc = model.evaluate([x_test, x_test, x_test],y_test, verbose=0)
    #print('Test Accuracy: %f' % (acc*100))
    predictions = model.predict([x_test, x_test, x_test], verbose=0)
    
    #x_test and testTweets are parallel arrays corresponding to the same set of tweets
    for k in range(len(predictions)):
        #print(predictions[k])
        curProbs = predictions[k]
        outStr = str(testTweets[k][0]) + ',' + numsAsLabels[np.argmax(curProbs)] 
    
        
        outFile.write(outStr + '\n')
        

    return model
    #return model




trainFile =  sys.argv[1]
testFile = sys.argv[2]
outputFile = sys.argv[3]

#open output file
outFile = open(outputFile, 'w')

accuracies = []
#cnn_word_word,model = CNN_word_word(i, cityFormats, outFile)
model = CNN_word_word(outFile)
#print("Accuracy for fold ", i, ":", cnn_word_word*100)
#accuracies.append(cnn_word_word)
    
#print("Overall Accuracy for " , tweetFile ," with subset of:", tweetSubset, "Accuracy :" ,sum(accuracies)/len(accuracies)*100 )
