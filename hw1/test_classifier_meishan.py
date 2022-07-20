#!/usr/bin/env python
# coding: utf-8

# ## CS 6120: Natural Language Processing - Prof. Ahmad Uzair
# 
# ### Assignment 1: Naive Bayes - Test classifier

# # Q8. Modularize your calssifier (10 points)
# 1. Convert your code into a python module text_classifier.py
# 
# 2. The user should be able to launch the application on command prompt using python test_classifier.py command. The module will automatically load the model paramters from a local file of your choice and be ready to take the input from user on command prompt. The program will preprocess user input, tokenize and predict the class.
# 
# 3. Your module will take the input from user and output sentiment class in an indefinite loop. The output should printout the probabilities for each input token along with the final classification decision. Program will quit if user enters X.


import numpy as np
import math
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import string
from sklearn.covariance import log_likelihood
from sklearn.utils import resample
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os

warnings.filterwarnings("ignore")

def Data_pre_process(data):
    # ## Reading the data
    df = pd.read_csv(data, sep = ',', encoding = 'latin-1', usecols = lambda col: col not in ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])

    df_majority = df.loc[df["sentiment"] == "positive"]
    df_minority = df.loc[df["sentiment"] == "negative"]

    negative_upsample = resample(df_minority, replace = True, 
                            n_samples = df_majority.shape[0],
                            random_state = 101)

    df_upsampled = pd.concat([df_majority, negative_upsample])  # concat two data frames i,e majority class data set and upsampled minority class data set
    df_upsampled = df_upsampled.sample(frac = 1)

    return(df_upsampled)

def clean_review(review):
    '''
    Input:
        review: a string containing a review.
    Output:
        review_cleaned: a processed review. 

    '''
    stopw = stopwords.words('english')
    Lemma = WordNetLemmatizer()
    
    review = review.lower()
    review = re.sub(r"http\S+", " ", review)
    review = re.sub(r"www.\S+", " ", review)
    review = re.sub(r"<br />", " ", review)
    
    rmv_punc = "".join([r for r in review if not r in string.punctuation])
    tokens = word_tokenize(rmv_punc)
    words = [t for t in tokens if not t in stopw and t.isalpha()]
    
    # lemmatization
    review_cleaned = " ".join([Lemma.lemmatize(w) for w in words])

    return review_cleaned

def find_occurrence(frequency, word, label):
    '''
    Params:
        frequency: a dictionary with the frequency of each pair (or tuple)
        word: the word to look up
        label: the label corresponding to the word
    Return:
        n: the number of times the word with its corresponding label appears.
    '''
    n = frequency.get((word, label), 0)
    
    return n


# ### Converting output to numerical format:
def Coverting2n(data):
    output_map = {'positive': 0, 'negative': 1}
    data = data.map(output_map)

def review_counter(output_occurrence, reviews, positive_or_negative):
    '''
    Params:
        output_occurrence: a dictionary that will be used to map each pair to its frequency
        reviews: a list of reviews
        positive_or_negative: a list corresponding to the sentiment of each review (either 0 or 1)
    Return:
        output: a dictionary mapping each pair to its frequency
    '''
    ## Steps :
    # define the key, which is the word and label tuple
    # if the key exists in the dictionary, increment the count
    # else, if the key is new, add it to the dictionary and set the count to 1
    
    for label, review in zip(positive_or_negative, reviews):
        split_review = clean_review(review).split()
#         split_review = clean_review(review)
        for word in split_review:
            output_occurrence[(word, label)] = output_occurrence.get((word, label), 0) +1
    
    return output_occurrence
   

def train_naive_bayes(freqs, train_x, train_y):
    '''
    Input:
        freqs: dictionary from (word, label) to how often the word appears
        train_x: a list of reviews
        train_y: a list of labels correponding to the reviews (0,1)
    Output:
        logprior: the log prior. (equation 3 above)
        loglikelihood: the log likelihood of you Naive bayes equation. (equation 6 above)
    '''
    loglikelihood = {}
    logprior = 0

    # calculate V, the number of unique words in the vocabulary
    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab)
    
    # calculate num_pos and num_neg - the total number of positive and negative words for all documents
    num_pos = num_neg = 0
    frequency_dict = freqs
    
    for pair in frequency_dict.keys():
        # # if the label is zero
        if pair[1] == 0:

            # Increment the number of positive words by the count for this (word, label) pair
            num_pos += frequency_dict[pair]

        # else, the label is negative
        else:

            # increment the number of negative words by the count for this (word,label) pair
            num_neg += frequency_dict[pair] 

    # Calculate num_doc, the number of documents
    num_doc = len(train_x)

    # Calculate D_pos, the number of positive documents 
    pos = neg = 0
    
    for i in train_y:
        if i == 0:
            pos += 1
        else:
            neg += 1
            
    pos_num_docs = pos

    # Calculate D_neg, the number of negative documents 

    neg_num_docs = neg

    # Calculate logprior
    logprior = math.log(neg_num_docs) - math.log(pos_num_docs)

    # For each word in the vocabulary...
    for word in vocab:
        # get the positive and negative frequency of the word
        freq_pos = find_occurrence(freqs, word, 0)
        freq_neg = find_occurrence(freqs, word, 1)

        # calculate the probability that each word is positive, and negative
        p_w_pos = (freq_pos+1)/(num_pos+V)
        p_w_neg = (freq_neg+1)/(num_neg+V)

        # calculate the log likelihood of the word
        loglikelihood[word] = math.log(p_w_neg/p_w_pos)


    return logprior, loglikelihood

def naive_bayes_predict(review, logprior, loglikelihood):
    '''
    Params:
        review: a string
        logprior: a number
        loglikelihood: a dictionary of words mapping to numbers
    Return:
        total_prob: the sum of all the loglikelihoods of each word in the review (if found in the dictionary) + logprior (a number)

    '''
    
      # process the review to get a list of words
    word_l = word_tokenize(clean_review(review))

    # initialize probability to zero
    total_prob = 0

    # add the logprior
    total_prob = logprior

    for word in word_l:

        # check if the word exists in the loglikelihood dictionary
        if word in loglikelihood:
            # add the log likelihood of that word to the probability
            total_prob += loglikelihood[word]

    if total_prob > 0:
        total_prob = 1
    else:
        total_prob = 0

    return total_prob

def new_naive_bayes_predict(review, logprior, loglikelihood):
    '''
    Params:
        review: a string
        logprior: a number
        loglikelihood: a dictionary of words mapping to numbers
    Return:
        total_prob: the sum of all the loglikelihoods of each word in the review (if found in the dictionary) + logprior (a number)

    '''
    
      # process the review to get a list of words
    word_l = word_tokenize(clean_review(review))

    # initialize probability to zero
    total_prob = 0

    # add the logprior
    total_prob = logprior
    prob_list = {}

    for word in word_l:

        # check if the word exists in the loglikelihood dictionary
        if word in loglikelihood:
            # add the log likelihood of that word to the probability
            total_prob += loglikelihood[word]
            prob_list[word] = loglikelihood[word]

    if total_prob > 0:
        total_prob = 1
    else:
        total_prob = 0

    return total_prob, prob_list

def test_naive_bayes(test_x, test_y, logprior, loglikelihood):
    """
    Input:
        test_x: A list of reviews
        test_y: the corresponding labels for the list of reviews
        logprior: the logprior
        loglikelihood: a dictionary with the loglikelihoods for each word
    Output:
        accuracy: (# of reviews classified correctly)/(total # of reviews)
    """
    accuracy = 0  

    
    y_hats = []
    for review in test_x:
        # if the prediction is > 0
        if naive_bayes_predict(review, logprior, loglikelihood) > 0:
            # the predicted class is 1
            y_hat_i = 1
        else:
            # otherwise the predicted class is 0
            y_hat_i = 0

        # append the predicted class to the list y_hats
        y_hats.append(y_hat_i)

    # error is the average of the absolute values of the differences between y_hats and test_y
    error = np.mean(y_hats - test_y)

    accuracy = 1 - error

    return accuracy

def create_cm(X, y_true, logprior, loglikelihood):
    y_pre = []
    for review in X:
        pre = naive_bayes_predict(review, logprior, loglikelihood)
        y_pre.append(pre)

    cm = confusion_matrix(y_true, y_pre)
    
    return cm

def Confusion_Matrix(X, y, size, seed):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size, 
                                                        random_state = seed)
    output_map = {'positive': 0, 'negative': 1}
    y_train =  y_train.map(output_map)
    y_test = y_test.map(output_map)
    
    freqs = review_counter({}, X_train, y_train)
    logprior, loglikelihood = train_naive_bayes(freqs, X_train, y_train)
    
    cm_train = create_cm(X_train, y_train, logprior, loglikelihood)
    cm_test = create_cm(X_test, y_test, logprior, loglikelihood)
    
    return logprior, loglikelihood, cm_train, cm_test

def Custom_Training(data, size, seed):
    df_upsampled = Data_pre_process(data)

    X = df_upsampled['review']
    y = df_upsampled['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size, 
                                                        random_state = seed)
    output_map = {'positive': 0, 'negative': 1}
    y_train =  y_train.map(output_map)
    y_test = y_test.map(output_map)    

    freqs = review_counter({}, X_train, y_train)
    logprior, loglikelihood = train_naive_bayes(freqs, X_train, y_train) 

    result, prob_list = new_naive_bayes_predict(review, logprior, loglikelihood)  

    return result, prob_list                                             


def local_model():
    # load the local model paramters
    loglikelihood = {}
    with open("Loglikelihood.txt","r") as L:
        for line in L.readlines():
            k = line.split(" ")[0]
            v = line.split(" ")[1]
            loglikelihood[k] = float(v)

    return loglikelihood

def Process_result(result, prob_list):
    if result == 0:
        seg = "positive"
    else:
        seg = "negative"

    print("The loglikelihood for each input is: ", prob_list, "\n")
    print("The classification of the review: [", review, "] is:", result)
    print("Meaning that the sentiment of this review is: ", seg)


if __name__ == '__main__':
    print ("\n", "Hello! Welcome to use Meishan's Review Classifier!")

    while True:
        review = input("\n Please provide a review for classification, or press 'X' to quit. \n")
        if review.upper() == 'X':
           break

        else:
            choice = input("\n Please choose:\n 1. Use pre-trained model. \n 2. Re-train the model. \n (Enter a number to choose [1] or [2], or press 'X' to quit.) \n")

            if choice.upper() == 'X':
                break

            elif int(choice) == 1:
                loglikelihood = local_model()
                logprior = 0.011864817717441412 

                result, prob_list = new_naive_bayes_predict(review, logprior, loglikelihood)
                Process_result(result, prob_list)

            elif int(choice) == 2:
                size = input("\n Please enter the test size of the new model: \n (Enter the percentage of samples: 0 < percentage < 1, or enter the number of samples.) \n (Press 'X' to quit.) \n")

                if size.upper() == 'X':
                    break
        
                else: 
                    size = float(size)

                    if size < 0 or size > 1 :
                        size = abs(int(size))
                        print("After checking and processing, size = ", size)
                    
                    seed = input("\n Please enter the random state of the new model: \n (Enter an integer as seed) \n (Press 'X' to quit.) \n")

                    if seed.upper() == 'X':
                        break

                    else:
                        seed = abs(int(float(seed)))
                        print("After checking and processing, seed = ", seed)
                        data = input("\n Please upload the dataset at the same folder and enter the name here. (example: self_dataset.csv) \n  Or press 'Y' to use local dataset \n (Press 'X' to quit.) \n")
                        print("Please wait one minute for training the new model. Thank you for your patience.", "\n")

                        if data.upper() == 'X':
                            break

                        elif data.upper() == 'Y':
                            data = "movie_reviews.csv"
                            result, prob_list = Custom_Training(data, size, seed)
                            Process_result(result, prob_list)
                        
                        elif os. path.isfile(data) == False:
                            print("File not exists, please double check that the file name is entered correctly and try again. \n ")
                            break

                        else:
                            result, prob_list = Custom_Training(data, size, seed)
                            Process_result(result, prob_list) 

    print("Thank you for using Meishan's Review Clasifier, goodbye!")
                        
