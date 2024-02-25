import json
import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.metrics
from sklearn.feature_extraction.text import CountVectorizer

from cross_validation import *
BOW_FILE = "bow_columns_list.txt"

def make_bag_of_words_from_vocab(text_series, ngram_range=(1, 1), max_features=1000, vocabulary_file=BOW_FILE):
    with open(vocabulary_file, 'r') as file:
        vocabulary = file.read().splitlines()
    text_series = [text_list[1] for text_list in text_series]
    vectorizer = CountVectorizer(ngram_range=ngram_range, max_features=max_features, stop_words='english', vocabulary=vocabulary)
    bow_matrix = vectorizer.fit_transform(text_series)
    bow_df = pd.DataFrame(bow_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    return bow_df

def make_bag_of_words(text_series, ngram_range=(1, 1), max_features=1000):
    vectorizer = CountVectorizer(ngram_range=ngram_range, max_features=max_features, stop_words='english')
    bow_matrix = vectorizer.fit_transform(text_series)
    bow_df = pd.DataFrame(bow_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    return bow_df

def make_bags_of_words(text_series, smallest, largest, max_features=None):
    df = pd.DataFrame(index=range(smallest, largest+1), columns=range(smallest, largest+1))
    for i in range(smallest, largest+1):
        for j in range(i, largest+1):
            df[i][j] = make_bag_of_words(text_series, ngram_range=(i, j), max_features=max_features)
    return df

def find_best_hyperparameters(x_train_array, y_train_array):
    # penalties = ['l1', 'l2', 'elasticnet']
    penalties = ['l2']
    # C_grid = np.logspace(0, 2, 4)
    C_grid = [1.0]
    # solvers = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
    solvers = ['lbfgs']
    fit_intercepts = [True, False]
    # fit_intercepts = [True]
    # tols = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1]
    tols = [1e-4]
    print(len(tols) * len(fit_intercepts) * len(solvers) * len(C_grid) * len(penalties))

    num_folds = 6
    random_state = 132

    models_and_hyperparameters = []

    curr_itr = 0
    for penalty in penalties:
        for C in C_grid:
            for solver in solvers:
                for fit_intercept in fit_intercepts:
                    for tol in tols:
                        curr_itr += 1
                        if solver == 'lbfgs' and penalty != 'l2':
                            penalty = 'l2'
                        model = sklearn.linear_model.LogisticRegression(penalty=penalty, C=C, solver=solver, fit_intercept=fit_intercept, tol=tol, max_iter=1000)
                        model.fit(x_train_array, y_train_array.ravel())

                        # determine error
                        train_err_per_fold, test_error_per_fold = train_models_and_calc_scores_for_n_fold_cv(model, x_train_array, y_train_array.ravel(), n_folds=num_folds, random_state=random_state)

                        # probas and rocauc
                        y_train_proba = model.predict_proba(x_train_array)[:,1]
                        rocauc = sklearn.metrics.roc_auc_score(y_train_array, y_train_proba)

                        models_and_hyperparameters.append((model, penalty, C, solver, fit_intercept, tol, np.average(train_err_per_fold), rocauc))
                        print(curr_itr, penalty, C, solver, fit_intercept, tol, np.average(train_err_per_fold), rocauc)

    min_err = 1000000
    best_model = ()
    for models_and_hyperparameter in models_and_hyperparameters:
        # print(models_and_hyperparameter)
        curr_err = models_and_hyperparameter[6]
        if curr_err < min_err:
            min_err = curr_err
            best_model = models_and_hyperparameter
    return best_model


def main(data_dir='./data_reviews'):
    x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))

    # bows_df_10 = make_bags_of_words(x_train_df['text'], 1, 4, max_features=10)
    # bows_df_25 = make_bags_of_words(x_train_df['text'], 1, 4, max_features=25)
    # bows_df_50 = make_bags_of_words(x_train_df['text'], 1, 4, max_features=50)
    # bows_df_100 = make_bags_of_words(x_train_df['text'], 1, 4, max_features=100)
    # bows_df_250 = make_bags_of_words(x_train_df['text'], 1, 4, max_features=250)
    # bows_df_500 = make_bags_of_words(x_train_df['text'], 1, 4, max_features=500)
    # bows_df_1000 = make_bags_of_words(x_train_df['text'], 1, 4, max_features=1000)
    # bows_df_2000 = make_bags_of_words(x_train_df['text'], 1, 4, max_features=2000)
    bows_df_10 = make_bag_of_words(x_train_df['text'], ngram_range=(1, 1), max_features=10)
    bows_df_25 = make_bag_of_words(x_train_df['text'], ngram_range=(1, 1), max_features=25)
    bows_df_50 = make_bag_of_words(x_train_df['text'], ngram_range=(1, 1), max_features=50)
    bows_df_100 = make_bag_of_words(x_train_df['text'], ngram_range=(1, 1), max_features=100)
    bows_df_250 = make_bag_of_words(x_train_df['text'], ngram_range=(1, 1), max_features=250)
    bows_df_500 = make_bag_of_words(x_train_df['text'], ngram_range=(1, 1), max_features=500)
    bows_df_1000 = make_bag_of_words(x_train_df['text'], ngram_range=(1, 1), max_features=1000)

    
    # choose bow
    bow = make_bag_of_words(x_train_df['text'], ngram_range=(1, 1), max_features=1000)
    print(bow.columns)
    # export as list
    with open(BOW_FILE, 'w') as file:
        for column in bow.columns:
            file.write(column + '\n')

    # make final training data
    print(x_train_df)
    x_train_df = pd.concat([x_train_df, bow], axis=1).drop(columns=['text', 'website_name'], axis=1)
    website_mapping = {'imdb': 0, 'amazon': 1, 'yelp': 2}
    print(x_train_df)
    x_train_array = x_train_df.to_numpy()
    y_train_array = y_train_df.to_numpy()
    model = sklearn.linear_model.LogisticRegression()
    model.fit(x_train_array, y_train_array.ravel())
    
    # find best hyperparameters
    best_model_and_hyperparameters = find_best_hyperparameters(x_train_array=x_train_array, y_train_array=y_train_array)
    print(best_model_and_hyperparameters)
    best_model = best_model_and_hyperparameters[0]
    print(best_model)

    # pickle model
    with open('regression_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)



if __name__ == '__main__':
    main()
