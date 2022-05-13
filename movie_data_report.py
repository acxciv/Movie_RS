import json
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append('../DataSet')
DATA_DIR = './DataSet'
PATH_TO_TESTS = Path(__file__).parent
credit_movies_data = "tmdb_5000_credits.csv"
movies_data = "tmdb_5000_movies.csv"
credit_file_path = os.path.join(PATH_TO_TESTS, DATA_DIR, credit_movies_data)
movie_file_path = os.path.join(PATH_TO_TESTS, DATA_DIR, movies_data)

""" Reading movie dataset from csv files """
credit = pd.read_csv(credit_file_path)
movie = pd.read_csv(movie_file_path)

credit.columns = ['id', 'tittle', 'cast', 'crew']
all_movies = movie.merge(credit, on='id')  # merging movie and credit data set based on ID
del movie
del credit

""" Change the given string format of release_date to datetime format """
all_movies["release_date"] = pd.to_datetime(all_movies['release_date'])
all_movies['release_year'] = all_movies['release_date'].dt.year
all_movies['release_month'] = all_movies['release_date'].dt.month_name()
del all_movies["release_date"]

"""
Change all columns that have json string into json format and
eliminate ID since ID is not necessary for users 
"""

json_columns = {'cast', 'crew', 'genres', 'keywords', 'production_countries',
                'production_companies', 'spoken_languages'}

for c in json_columns:
    all_movies[c] = all_movies[c].apply(json.loads)
    if c != "crew":  # We need other information more than just the name
        all_movies[c] = all_movies[c].apply(lambda row: [i["name"] for i in row])

"""
Create director writer and producer columns from crew column of the data set
"""


def get_role(role, row):
    person_name = [i['name'] for i in row if i['job'] == role]
    return person_name[0] if len(person_name) else np.nan


all_movies["director"] = all_movies["crew"].apply(partial(get_role, "Director"))
all_movies["writer"] = all_movies["crew"].apply(partial(get_role, "Writer"))
all_movies["producer"] = all_movies["crew"].apply(partial(get_role, "Producer"))
del all_movies["crew"]

"""
Fill the missing values with the most frequent value using fillna method from pandas library
"""
for col in ["runtime", "release_year", "release_month"]:
    all_movies[col] = all_movies[col].fillna(all_movies[col].mode().iloc[0])

"""
Display all movies from the data set
"""


def lookup_all_movies():
    return all_movies['title']


""" 
Display movies with tile and cast only
"""


def lookup_title_casts():
    title_casts = all_movies[['title', 'cast']]
    return title_casts
