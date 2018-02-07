import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.utils import resample


def load_data_csv(path):
    df = pd.read_csv(path, names=['rating', 'title', 'review'])
    if df.duplicated('review').any():
        df = df.drop_duplicates('review')
    df['rating'] = df['rating'].replace([1, 2], 0)
    df['rating'] = df['rating'].replace([4, 5], 1)
    df['rating'] = df['rating'].replace([3], 2)
    return df

def load_data_json(path_or_object):
    df = pd.read_json(path_or_object)
    if df.duplicated('review').any():
        df = df.drop_duplicates('review')

    return df


def resample_dataframe(df):
    smallest_class = df['rating'].value_counts().argmin()
    num_samples =  df['rating'].value_counts().min()
    downsampled_df = df[df.rating==smallest_class]
    for cl in df['rating'].unique():
        if cl != smallest_class:
            df_majority = df[df.rating==cl]
            df_majority_downsampled = resample(df_majority,
                                           replace=False,
                                           n_samples=num_samples,
                                           random_state=123)
            downsampled_df = downsampled_df.append(df_majority_downsampled)
    return downsampled_df


def process_data(df, max_num_words, num_wors_per_seq, labels=False):
    tok = Tokenizer(max_num_words)
    new_df = pd.DataFrame()
    new_df['merged'] = df['title'].astype(str) + '  ' + df['review'].astype(str)

    tok.fit_on_texts(new_df['merged'])
    sequences = tok.texts_to_sequences(new_df['merged'])
    padded_sequences = pad_sequences(sequences, maxlen=num_wors_per_seq)
    X = np.array(padded_sequences)
    y = None
    if labels:
        y = np.array(df['rating'])
    return (X, y)


if __name__ == '__main__':
    df = pd.read_csv('../data/test.csv', names=['rating', 'title', 'review'])
    if df.duplicated('review').any():
        df = df.drop_duplicates('review')

    resampled_df = resample_dataframe(df)

    X, y = process_data(resampled_df, 50000, 100)
    print(X.shape)
    print(y.shape)