'''
    process data from IMDb Dataset, available at
    https://datasets.imdbws.com/

    running this script takes a couple minutes on a laptop
'''

import numpy as np
import pandas as pd


def encode_movie(tags):
    '''provide movie encoding based on the tagged genres'''
    enc = np.sum([genres_one_hot[genres.index(tag)] for tag in tags.split(',')], axis=0)
    return enc


if __name__ == '__main__':

    # select parameters for the constructed dataset
    NUM_MOVIES = 10000
    NUM_ACTIONS = 100

   # read and merge movies data, remove movies without genres
    try:
        ratings = pd.read_csv('./IMDb/ratings.tsv', sep='\t')
    except:
        print('\nratings.tsv file is not found, please download it at',
              'https://datasets.imdbws.com/title.ratings.tsv.gz and extract to ./IMDb/')
        raise SystemExit
    try:
        basics = pd.read_csv('./IMDb/basics.tsv', sep='\t', low_memory=False)
    except:
        print('\nbasics.tsv file is not found, please download it at',
              'https://datasets.imdbws.com/title.basics.tsv.gz and extract to ./IMDb/')
        raise SystemExit
    movies = ratings.merge(basics, on='tconst').dropna()
    movies = movies[movies.genres != '\\N']

    # get NUM_MOVIES most reviewed movies
    movies = movies.sort_values('numVotes', ascending=False).reset_index(drop=True)[:NUM_MOVIES]
    movies = movies[['tconst', 'primaryTitle', 'genres', 'averageRating', 'numVotes']]

    # genres extracted from NUM_MOVIES most reviewed movies
    genres = list(np.unique(np.concatenate([tags.split(',') for tags in movies.genres.values])))
    genres_one_hot = np.eye(len(genres))

    # encode and save processed movies data
    movies['encoding'] = movies.genres.apply(encode_movie)
    movies.to_csv('./IMDb/movies.csv', index=False)

    # get NUM_ACTIONS most popular movies
    movies_popular = movies[:NUM_ACTIONS]
    movies_popular.to_csv('./IMDb/movies_popular.csv', index=False)

