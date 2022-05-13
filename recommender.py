from cosine_algo import get_recommendations, get_recommend_with_add_features
from movie_data_report import lookup_all_movies, lookup_title_casts


# This is the user interface
def recommender_system():
    # Menu
    choose_option = input("Choose a command \n"
                          "1. Look up all Movies \n"
                          "2. Get A Recommendation of Similar Movies \n"
                          "3. Exit the Program. \n")

    # Display all movies
    if choose_option == "1":
        print(lookup_all_movies())
        try:
            title_genres = input("Press 1 to See Movies with Title and Cast Only \n")
            if title_genres == "1":
                print(lookup_title_casts())
                SystemExit
        except KeyboardInterrupt:
            SystemExit

    if choose_option == "3":
        SystemExit

    # Search movies
    if choose_option == "2":
        try:
            search_movies = input("Please Type :\n "
                                  "\t   Your Favorite Movie Name(e.g., Spider-Man) or\n "
                                  "\t   Movie Name with Cast and Genres by Separating Commas(e.g., Spider-Man, "
                                  "Tom Holland, Action) \n")
            count = 0
            if ',' in search_movies:
                ++ count
                search_movies_partition = search_movies.partition(",")[count]
                if get_recommend_with_add_features(search_movies_partition).empty:
                    recommender_system()
                else:
                    print(get_recommend_with_add_features(search_movies_partition))

            else:
                if get_recommendations(search_movies).empty:
                    recommender_system()
                else:
                    print(get_recommendations(search_movies))
                    SystemExit
        except KeyboardInterrupt:
            SystemExit
