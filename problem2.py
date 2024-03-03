import os

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import sklearn.ensemble
import sklearn.metrics
from sklearn.feature_extraction.text import CountVectorizer as Vectorizer

from cross_validation import *

BOW_FILE = "bow_columns_list_problem2.txt"
REGRESSION_PKL_FILE = "regression_mdl_problem2.pkl"
WEBSITE_MAPPING = {'imdb': 0, 'amazon': 1, 'yelp': 2}
RANDOM_STATE = 132
NUM_FOLDS = 6

N_JOBS = -1

def make_bag_of_words(text_series, ngram_range=(1, 1), max_features=1000, binary=True):
    vectorizer = Vectorizer(ngram_range=ngram_range, max_features=max_features, stop_words='english', binary=binary)
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
    

def find_best_hyperparameters(x_train_array, y_train_array, n_estimators_range, max_depth_range, min_samples_split_range, min_samples_leaf_range):
    tuples = []
    total_iter = len(n_estimators_range) * len(max_depth_range) * len(min_samples_split_range) * len(min_samples_leaf_range)
    iter = 0

    for n_estimators in n_estimators_range:
        for max_depth in max_depth_range:
            for min_samples_split in min_samples_split_range:
                for min_samples_leaf in min_samples_leaf_range:
                    iter = iter + 1
                    model = sklearn.ensemble.RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
                    train_err_per_fold, test_error_per_fold = train_models_and_calc_scores_for_n_fold_cv(model, x_train_array, y_train_array.ravel(), n_folds=NUM_FOLDS, random_state=RANDOM_STATE)
                    avg_test_error = np.average(test_error_per_fold)
                    avg_train_error = np.average(train_err_per_fold)

                    new_tuple = (n_estimators, max_depth, min_samples_split, min_samples_leaf, avg_test_error, avg_train_error)
                    tuples.append(new_tuple)
                    print(f"{iter}/{total_iter}", new_tuple)
    return find_min_tuple(tuples)



def main(data_dir='./data_reviews'):
    x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    text = x_train_df['text']

    # choose bow
    bow_max_features = 1000
    bow = make_bag_of_words(x_train_df['text'], max_features=bow_max_features)
    # export as list
    # with open(BOW_FILE, 'w') as file:
    #     for column in bow.columns:
    #         file.write(column + '\n')
    
    # make final training data
    x_train_df = pd.concat([x_train_df, bow], axis=1).drop(columns=['text', 'website_name'], axis=1)
    # x_train_df['website_name'].replace(website_mapping, inplace=True)
    x_train_array = x_train_df.to_numpy()
    y_train_array = y_train_df.to_numpy()


    ## RANGES
    n_estimators_range = range(10, 500, 100)
    max_depth_range = range(100, 1100, 100)
    min_samples_split_range = range(1, 11, 1)
    min_samples_leaf_range = range(1, 11, 1)

    ret = find_best_hyperparameters(x_train_array, y_train_array, n_estimators_range, max_depth_range, min_samples_split_range, min_samples_leaf_range)

    ## ESTIMATORS
    # ret = test_n_estimators(x_train_array, y_train_array, n_estimators_range)
    # print(ret)


    ## DEPTH
    # ret = test_max_depth(x_train_array, y_train_array, max_depth_range)
    # print(ret)


    ## SPLIT
    # min_samples_split_range = [float(x) / 10 for x in min_samples_split_range]
    # ret = test_min_samples_split(x_train_array, y_train_array, min_samples_split_range)
    # print(ret)


    ## LEAF
    # # min_samples_leaf_range = [float(x) / 10 for x in min_samples_leaf_range]
    # ret = test_min_samples_leaf(x_train_array, y_train_array, min_samples_leaf_range)
    # print(ret)
    
if __name__=='__main__':
    main()