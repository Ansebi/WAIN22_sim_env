'''
    process data from MovieLens 25M Dataset, available at
    https://grouplens.org/datasets/movielens/25m/

    running this script takes about 10 minutes on a laptop
'''

import numpy as np
import pandas as pd


def encode_movie(tags):
    '''provide movie encoding based on the tagged genres'''
    enc = np.sum([genres_one_hot[genres.index(tag)] for tag in tags.split('|')], axis=0)
    return enc

def encode_user(history):
    '''provide user encoding based on their normalized movie ratings'''
    movieIds = history.movieId.to_list()
    ratings = history.rating.to_numpy() / 5
    enc = np.sum(ratings * movies[movies.movieId.isin(history.movieId)].encoding.values)
    return enc


if __name__ == '__main__':

    # select parameters for the constructed dataset
    NUM_STATES = 10000
    NUM_ACTIONS = 100

    # genres provided by MovieLens 25M Dataset
    genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
              'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
              'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    genres_one_hot = np.eye(len(genres))

    # read movies data, remove movies without genres and IMAX tag
    try:
        movies = pd.read_csv('./MovieLens25M/movies.csv')
    except:
        print('\nmovies.csv file is not found, please download it at',
              'https://grouplens.org/datasets/movielens/25m/ and extract to ./MovieLens25M/')
        raise SystemExit
    movies.genres = movies.genres.str.replace('\|?IMAX', '', regex=True)
    movies = movies[movies.genres != '(no genres listed)']
    movies = movies[movies.genres != ''].reset_index(drop=True)

    # encode and save processed movies data
    movies['encoding'] = movies.genres.apply(encode_movie)
    movies.to_csv('./MovieLens25M/movies_processed.csv', index=False)

    # read ratings data and drop removed movies
    try:
        ratings = pd.read_csv('./MovieLens25M/ratings.csv')
    except:
        print('\ratings.csv file is not found, please download it at',
              'https://grouplens.org/datasets/movielens/25m/ and extract to ./MovieLens25M/')
        raise SystemExit
    ratings = ratings[ratings.movieId.isin(movies.movieId) == True]

    # encode users based on their ratings and save processed user data
    users = ratings.groupby('userId').apply(encode_user).reset_index()
    users.columns = ['userId', 'encoding']
    users.to_csv('./MovieLens25M/users_processed.csv', index=False)

    # get NUM_ACTIONS most popular movies
    popular_ids = ratings.movieId.value_counts().index[:NUM_ACTIONS].to_list()
    movies_popular = movies[movies.movieId.isin(popular_ids)].reset_index(drop=True)
    movies_popular.to_csv('./MovieLens25M/movies_popular.csv', index=False)

    # get NUM_STATES most active users
    active_ids = ratings.userId.value_counts().index[:NUM_STATES].to_list()
    users_active = users[users.userId.isin(active_ids)].reset_index(drop=True)
    users_active.to_csv('./MovieLens25M/users_active.csv', index=False)

