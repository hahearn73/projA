import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
from sklearn.feature_extraction.text import CountVectorizer

def make_bag_of_words(text_series, ngram_range=(1, 1), max_features=None):
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

def main(data_dir='./data_reviews'):
    x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))

    bows_df_10 = make_bags_of_words(x_train_df['text'], 1, 4, max_features=10)
    bows_df_25 = make_bags_of_words(x_train_df['text'], 1, 4, max_features=25)
    bows_df_50 = make_bags_of_words(x_train_df['text'], 1, 4, max_features=50)
    bows_df_100 = make_bags_of_words(x_train_df['text'], 1, 4, max_features=100)
    bows_df_250 = make_bags_of_words(x_train_df['text'], 1, 4, max_features=250)
    bows_df_500 = make_bags_of_words(x_train_df['text'], 1, 4, max_features=500)
    bows_df_1000 = make_bags_of_words(x_train_df['text'], 1, 4, max_features=1000)

    print(bows_df_50[1][2])

    
if __name__ == '__main__':
    main()
