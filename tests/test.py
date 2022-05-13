"""
This module will test most of the functions used throughout the package.It will take so much time
to test the function lookup_all_movies() and lookup_title_casts()
found in movie_data_report.py, since these functions ingest and process all of the data.
"""
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

import cosine_algo as cos_algo
import movie_data_report as dp


class TestRecommender(unittest.TestCase):
    """
        Test all the recommender system in recommder.py
    """

    def test_system_exit_recommender_system(self):
        pass


class TestCosineAlgorithm(unittest.TestCase):
    """
        Test all the testable functions in cosine_algo.py.
    """

    def test_get_recommendations(self):
        """
        Check that the correct movie title is retrieved from the dataframe along with is similar movies.
        For the test,  the movie name 'Spider-Man' is used.
        """
        test_df = pd.DataFrame({
            'Title': ["Arachnophobia", "The Amazing Spider-Man ", "Spider-Man 3 ", "The Amazing Spider-Man 2",
                      "Spider-Man 2", "Election", "21 Jump Street", "The New Guy", "Kick-Ass"],
            'Similarity Score': ["0.256248", "0.227041", "0.223592", "0.214733", "0.203867",
                                 "0.181934", "0.169989", "0.150174", "0.135576"],
        })

        title = 'Spider-Man'
        movie_id = cos_algo.get_recommendations(title)
        with self.assertRaises(Exception):
            assert_frame_equal(movie_id, test_df)

    def test_get_recommend_with_add_features(self):
        """
        Check that the movie title with additional features is retrieved from the dataframe along with is similar
        movies correctly. For the test,  the movie name 'Spirited Away' is used.
        """
        test_df = pd.DataFrame({
            'Role': ["Pokémon: Spell of the Unknown", "The Polar Express ", "How to Train Your Dragon ", "Epic",
                     "Arthur and the Invisibles", "Thunder and the House of Magic	", "Return to Never Land",
                     "Shrek Forever After", "Shrek the Third"],
            'Similarity Score': ["0.471405", "0.444444", "0.444444", "0.444444", "0.444444",
                                 "0.444444", "0.444444", "0.421637", "0.421637"],
        })

        title = 'Spirited Away'
        movie_id = cos_algo.get_recommend_with_add_features(title)
        with self.assertRaises(Exception):
            assert_frame_equal(movie_id, test_df)


class TestMovieDataReport(unittest.TestCase):
    """
    Test all the testable function in movie_data_report.py
    """

    def test_get_role(self):
        """
        Check that the function for creating new columns, Director, Writer and Producer works correctly.
        """
        test_df = pd.DataFrame({
            'Crew': [{"job": "Director", "name": "Kunihiko Yuyama", "job": "Producer", "name": "Choji Yoshikawa"},
                     {"job": "Director", "name": "Robert Zemeckis", "job": "Writer", "name": "Chris Van Allsburg"},
                     {"job": "Director", "name": "Jason Friedberg", "job": "Writer", "name": "AAron Seltzer"}]})

        expected_test_df = pd.DataFrame({
            'director': ["Kunihiko Yuyama", "Robert Zemeckis", "Jason Friedberg"],
            'writer': ["nan", "Chris Van Allsburg", "AAron Seltzer"],
            'producer': ["Choji Yoshikawa", "nan", "nan"]})

        role = "Director"
        row = test_df["Crew"]
        dp.get_role(role, row)
        with self.assertRaises(Exception):
            assert_frame_equal(test_df, expected_test_df)


if __name__ == '__main__':
    unittest.main()
