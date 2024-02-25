import numpy as np
import sklearn
from sklearn.metrics import zero_one_loss

from performance_metrics import calc_root_mean_squared_error
from proba_metrics import calc_mean_binary_cross_entropy_from_probas


def train_models_and_calc_scores_for_n_fold_cv(
        estimator, x_NF, y_N, n_folds=3, random_state=0):
    ''' Perform n-fold cross validation for a specific sklearn estimator object

    Args
    ----
    estimator : any regressor object with sklearn-like API
        Supports 'fit' and 'predict' methods.
    x_NF : 2D numpy array, shape (n_examples, n_features) = (N, F)
        Input measurements ("features") for all examples of interest.
        Each row is a feature vector for one example.
    y_N : 1D numpy array, shape (n_examples,)
        Output measurements ("responses") for all examples of interest.
        Each row is a scalar response for one example.
    n_folds : int
        Number of folds to divide provided dataset into.
    random_state : int or numpy.RandomState instance
        Allows reproducible random splits.

    Returns
    -------
    train_error_per_fold : 1D numpy array, size n_folds
        One entry per fold
        Entry f gives the error computed for train set for fold f
    test_error_per_fold : 1D numpy array, size n_folds
        One entry per fold
        Entry f gives the error computed for test set for fold f
    '''
    train_error_per_fold = np.zeros(n_folds, dtype=np.float32)
    test_error_per_fold = np.zeros(n_folds, dtype=np.float32)

    # TODO define the folds here by calling your function
    # e.g. ... = make_train_and_test_row_ids_for_n_fold_cv(...)
    train_ids_per_fold, test_ids_per_fold = make_train_and_test_row_ids_for_n_fold_cv(x_NF.shape[0], n_folds, random_state=random_state)

    # TODO loop over folds and compute the train and test error
    for i in range(n_folds):
        # xs
        train_fold_x = x_NF[train_ids_per_fold[i]]
        test_fold_x = x_NF[test_ids_per_fold[i]]

        # ys
        train_fold_y = y_N[train_ids_per_fold[i]]
        test_fold_y = y_N[test_ids_per_fold[i]]

        # make the fit
        estimator.fit(train_fold_x, train_fold_y)

        # get pred
        train_pred_y = estimator.predict(train_fold_x)
        test_pred_y = estimator.predict(test_fold_x)

        # calc probas
        proba_train_x = estimator.predict_proba(train_fold_x)[:,1]
        proba_test_x = estimator.predict_proba(test_fold_x)[:,1]

        # calc errors
        train_error_per_fold[i] = sklearn.metrics.zero_one_loss(train_fold_y, proba_train_x >= .5)
        train_error_per_fold[i] = sklearn.metrics.zero_one_loss(test_fold_y, proba_test_x >= .5)


    return train_error_per_fold, test_error_per_fold


def make_train_and_test_row_ids_for_n_fold_cv(
        n_examples=0, n_folds=3, random_state=0):
    ''' Divide row ids into train and test sets for n-fold cross validation.

    Will *shuffle* the row ids via a pseudorandom number generator before
    dividing into folds.

    Args
    ----
    n_examples : int
        Total number of examples to allocate into train/test sets
    n_folds : int
        Number of folds requested
    random_state : int or numpy RandomState object
        Pseudorandom number generator (or seed) for reproducibility

    Returns
    -------
    train_ids_per_fold : list of 1D np.arrays
        One entry per fold
        Each entry is a 1-dim numpy array of unique integers between 0 to N
    test_ids_per_fold : list of 1D np.arrays
        One entry per fold
        Each entry is a 1-dim numpy array of unique integers between 0 to N

    Guarantees for Return Values
    ----------------------------
    Across all folds, guarantee that no two folds put same object in test set.
    For each fold f, we need to guarantee:
    * The *union* of train_ids_per_fold[f] and test_ids_per_fold[f]
    is equal to [0, 1, ... N-1]
    * The *intersection* of the two is the empty set
    * The total size of train and test ids for any fold is equal to N
    '''
    if hasattr(random_state, 'rand'):
        # Handle case where provided random_state is a random generator
        # (e.g. has methods rand() and randn())
        random_state = random_state # just remind us we use the passed-in value
    else:
        # Handle case where we pass "seed" for a PRNG as an integer
        random_state = np.random.RandomState(int(random_state))

    # TODO obtain a shuffled order of the n_examples
    shuffled_order = np.arange(n_examples)
    random_state.shuffle(shuffled_order)

    train_ids_per_fold = list()
    test_ids_per_fold = list()
    
    # TODO establish the row ids that belong to each fold's
    fold_size = n_examples // n_folds
    remainder = n_examples % n_folds
    start_idx = 0
    for i in range(n_folds):
        end_idx = start_idx + fold_size
        if i < remainder: # add 1 id to each fold less than number of remainder ids
            end_idx = end_idx + 1
        test_ids_per_fold.append(shuffled_order[start_idx:end_idx])
        train_ids = np.concatenate([shuffled_order[:start_idx], shuffled_order[end_idx:]])
        train_ids_per_fold.append(train_ids)
        start_idx = end_idx
            
    return train_ids_per_fold, test_ids_per_fold
























####################################################################################################
# This is the 2024S version of this assignment. Please do not remove or make changes to this block.# 
# Otherwise, you submission will be viewed as files copied from other resources.                   # 
####################################################################################################



