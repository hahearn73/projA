import os
import pickle

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import sklearn.ensemble
import sklearn.metrics
from sklearn.feature_extraction.text import CountVectorizer as Vectorizer

from cross_validation import *

import warnings
warnings.filterwarnings("ignore")

BOW_FILE = "bow_columns_list_problem2.txt"
REGRESSION_PKL_FILE = "regression_mdl_problem2.pkl"
WEBSITE_MAPPING = {'imdb': 0, 'amazon': 1, 'yelp': 2}
RANDOM_STATE = 132
NUM_FOLDS = 6

N_JOBS = -1

def write_out_model(model, filename=REGRESSION_PKL_FILE):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
        print(f'model saved as: {filename}')

def make_bag_of_words(text_series, ngram_range=(1, 1), max_features=1000, binary=True):
    vectorizer = Vectorizer(ngram_range=ngram_range, max_features=max_features, stop_words='english', binary=binary)
    bow_matrix = vectorizer.fit_transform(text_series)
    bow_df = pd.DataFrame(bow_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    return bow_df

def make_bag_of_words_from_vocab(text_series, ngram_range=(1, 1), max_features=1000, vocabulary_file=BOW_FILE, binary=True):
    with open(vocabulary_file, 'r') as file:
        vocabulary = file.read().splitlines()
    text_series = [text_list[1] for text_list in text_series]
    vectorizer = Vectorizer(ngram_range=ngram_range, max_features=max_features, stop_words='english', vocabulary=vocabulary, binary=binary)
    bow_matrix = vectorizer.fit_transform(text_series)
    bow_df = pd.DataFrame(bow_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    return bow_df

def find_min_tuple(tuples, place=1):
    min_tuple = ()
    min = 1000000
    for tup in tuples:
        if tup[place] < min:
            min = tup[place]
            min_tuple = tup
    return min_tuple


def test_n_estimators(x_train_array, y_train_array, range_list):
    tuples = []
    for n_estimators in range_list:
        model = sklearn.ensemble.RandomForestClassifier(n_estimators=n_estimators)
        train_err_per_fold, test_error_per_fold = train_models_and_calc_scores_for_n_fold_cv(model, x_train_array, y_train_array.ravel(), n_folds=NUM_FOLDS, random_state=RANDOM_STATE)
        avg_test_error = np.average(test_error_per_fold)
        avg_train_error = np.average(train_err_per_fold)

        new_tuple = (n_estimators, avg_test_error, avg_train_error)
        tuples.append(new_tuple)
        print(new_tuple)


    ### GRAPH TEST AND TRAIN

    return find_min_tuple(tuples)
    

def test_max_depth(x_train_array, y_train_array, range_list):
    tuples = []
    for max_depth in range_list:
        model = sklearn.ensemble.RandomForestClassifier(max_depth=max_depth)
        train_err_per_fold, test_error_per_fold = train_models_and_calc_scores_for_n_fold_cv(model, x_train_array, y_train_array.ravel(), n_folds=NUM_FOLDS, random_state=RANDOM_STATE)
        avg_test_error = np.average(test_error_per_fold)
        avg_train_error = np.average(train_err_per_fold)

        new_tuple = (max_depth, avg_test_error, avg_train_error)
        tuples.append(new_tuple)
        print(new_tuple)

        
    ### GRAPH TEST AND TRAIN

    return find_min_tuple(tuples)


def test_min_samples_split(x_train_array, y_train_array, range_list):
    tuples = []
    for min_samples_split in range_list:
        model = sklearn.ensemble.RandomForestClassifier(min_samples_split=min_samples_split)
        train_err_per_fold, test_error_per_fold = train_models_and_calc_scores_for_n_fold_cv(model, x_train_array, y_train_array.ravel(), n_folds=NUM_FOLDS, random_state=RANDOM_STATE)
        avg_test_error = np.average(test_error_per_fold)
        avg_train_error = np.average(train_err_per_fold)

        new_tuple = (min_samples_split, avg_test_error, avg_train_error)
        tuples.append(new_tuple)
        print(new_tuple)

        
    ### GRAPH TEST AND TRAIN

    return find_min_tuple(tuples)


def test_min_samples_leaf(x_train_array, y_train_array, range_list):
    tuples = []
    for min_samples_leaf in range_list:
        model = sklearn.ensemble.RandomForestClassifier(min_samples_leaf=min_samples_leaf)
        train_err_per_fold, test_error_per_fold = train_models_and_calc_scores_for_n_fold_cv(model, x_train_array, y_train_array.ravel(), n_folds=NUM_FOLDS, random_state=RANDOM_STATE)
        avg_test_error = np.average(test_error_per_fold)
        avg_train_error = np.average(train_err_per_fold)

        new_tuple = (min_samples_leaf, avg_test_error, avg_train_error)
        tuples.append(new_tuple)
        print(new_tuple)

        
    ### GRAPH TEST AND TRAIN

    return find_min_tuple(tuples)
    

def find_best_hyperparameters(x_train_array, y_train_array,n_estimators_range, max_depth_range, max_samples_range):
    tuples = []
    # total_iter = len(n_estimators_range) * len(max_depth_range) * len(min_samples_split_range) * len(min_samples_leaf_range)
    iter = 0
    total_iter = len(n_estimators_range) * len(max_depth_range) * len(max_samples_range)
    print(total_iter)
    
    for max_depth in max_depth_range:
        for max_samples in max_samples_range:
            for n_estimators in n_estimators_range:
                iter = iter + 1
                model = sklearn.ensemble.RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples, n_jobs=N_JOBS, random_state=RANDOM_STATE, warm_start=True)
                train_err_per_fold, test_error_per_fold = train_models_and_calc_scores_for_n_fold_cv(model, x_train_array, y_train_array.ravel(), n_folds=NUM_FOLDS, random_state=RANDOM_STATE)
                avg_test_error = np.average(test_error_per_fold)
                avg_train_error = np.average(train_err_per_fold)

                # new_tuple = (n_estimators, max_depth, min_samples_split, min_samples_leaf, avg_test_error, avg_train_error)
                new_tuple = (avg_test_error, avg_train_error, model, n_estimators, max_depth, max_samples)
                tuples.append(new_tuple)
                print(f"{iter}/{total_iter}", new_tuple)
    return find_min_tuple(tuples, place=0)



def main(data_dir='./data_reviews'):
    x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    text = x_train_df['text']

    # choose bow
    bow_max_features = 1000
    bow = make_bag_of_words(x_train_df['text'], max_features=bow_max_features)
    # export as list
    with open(BOW_FILE, 'w') as file:
        for column in bow.columns:
            file.write(column + '\n')
    
    # make final training data
    x_train_df = pd.concat([x_train_df, bow], axis=1).drop(columns=['text', 'website_name'], axis=1)
    # x_train_df['website_name'].replace(website_mapping, inplace=True)
    x_train_array = x_train_df.to_numpy()
    y_train_array = y_train_df.to_numpy()


    ## RANGES
    n_estimators_range = range(100, 1001, 100)
    max_depth_range = range(100, 1001, 100)
    max_samples_range = range(100, 501, 100)

    print("ranges set")
    ret = find_best_hyperparameters(x_train_array, y_train_array, n_estimators_range, max_depth_range, max_samples_range)
    print(ret)
    write_out_model(ret[2], filename="warm_start_mdl.pkl")

    ## ESTIMATORS
    # ret = test_n_estimators(x_train_array, y_train_array, n_estimators_range)
    # print(ret)


    ## DEPTH
    # ret = test_max_depth(x_train_array, y_train_array, max_depth_range)
    # print(ret)

    
if __name__=='__main__':
    main()