import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import movie_data_report as mv

lemmatizer = WordNetLemmatizer()  # Performing a word formation process by usingWordNetLemmatizer from NLTK library


""" Data-preprocessing"""
def preprocess_sentences(text):
    text = text.lower()
    temp_sentences = []
    words = nltk.word_tokenize(text)
    tags = nltk.pos_tag(words)
    for i, word in enumerate(words):
        if tags[i][1] in VERB_CODES:
            lemma_sentences = lemmatizer.lemmatize(word, 'v')
        else:
            lemma_sentences = lemmatizer.lemmatize(word)
        if lemma_sentences not in stop_words and lemma_sentences.isalpha():
            temp_sentences.append(lemma_sentences)

    final_sentences = ' '.join(temp_sentences)
    final_sentences = final_sentences.replace("n't", " not")
    final_sentences = final_sentences.replace("'m", " am")
    final_sentences = final_sentences.replace("'s", " is")
    final_sentences = final_sentences.replace("'re", " are")
    final_sentences = final_sentences.replace("'ll", " will")
    final_sentences = final_sentences.replace("'ve", " have")
    final_sentences = final_sentences.replace("'d", " would")

    return final_sentences


""" Pre-processing the movie overview(plot summary) 
    by using NTLK processing techniques.
"""
movie_data = mv.all_movies
movie_data['overview'] = movie_data['overview'].fillna('')
stop_words = set(stopwords.words('english'))
VERB_CODES = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
movie_data['overview_preprocessed'] = movie_data['overview'].apply(preprocess_sentences)
# print(movie_data.head())


""" CosineSimilarity algorithm is used 
    to calculate the similarity scores of movies.
"""
movie_data['overview_preprocessed'] = movie_data['overview_preprocessed'].fillna('')
tfidf = TfidfVectorizer() # Vectorizing pre-processed the movie overview using TF-IDF Vecorization

tfidf_matrix = tfidf.fit_transform(movie_data['overview_preprocessed'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movie_data.index, index=movie_data['title']).drop_duplicates()


""" Cosine Similarity algorithm to find similar movies with additional features """
features = ['cast', 'genres']
top_num = 5  # leave only top values for some features
for feature in features:
    movie_data[feature] = movie_data[feature].apply(lambda x: x[:top_num] if isinstance(x, list) else [])

"""Remove spaces for some features """


def data_cleaning(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    elif isinstance(x, str):
        return str.lower(x.replace(" ", ""))
    else:
        return ''


""" 
Clean the data
"""
features = ['cast', 'genres']

for feature in features:
    movie_data[feature] = movie_data[feature].apply(data_cleaning)

""" 
Create the combined features column to the dataset
"""


def combined_features(x):
    return ' '.join([' '.join(x[f]) if isinstance(x[f], list) else str(x[f]) for f in features])


movie_data["combined_features"] = movie_data.apply(combined_features, axis=1)
count = CountVectorizer(stop_words='english')  # CountVectorizer will be used to remove stop_words
count_vector = count.fit_transform(movie_data['combined_features'])
cosine_sim2 = cosine_similarity(count_vector, count_vector)
movie_df = movie_data.reset_index()
indices = pd.Series(movie_df.index, index=movie_df['title'])


""" 
Building Recommendation Function 
"""


def get_recommendations(title, cosine_sim=cosine_sim):
    try:
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:10]
        movie_indices = [i[0] for i in sim_scores]
        movie_similarity = [i[1] for i in sim_scores]

        return pd.DataFrame(zip(movie_data['title'].iloc[movie_indices], movie_similarity),
                            columns=["Title", "Similarity Score"])
    except KeyError:
        print("Invalid Movie Name. Please Type the Correct Movie Name.")
        return pd.DataFrame()


""" 
Recommendation System with movie name and additional features 
"""


def get_recommend_with_add_features(title):
    return get_recommendations(title, cosine_sim2)
