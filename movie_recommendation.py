from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import scipy.sparse as sp

# convert movie title to lower caser
def get_data():
    movies_data = pd.read_csv('main_data.csv')
    movies_data['original_title'] = movies_data['original_title'].str.lower()
    return movies_data

# combine cast and genres columns in data_combine and drop others columns
def combine_data(data):
    data_combine = data.drop(columns=['movie_id', 'original_title', 'overview'])
    data_combine['combine'] = data_combine[data_combine.columns[0:2]].apply(lambda x: ','.join(x.dropna().astype(str)),
                                                                            axis=1)

    data_combine = data_combine.drop(columns=['cast', 'genres'])
    return data_combine

# calculate the similarity between
def transform_data(data_combine, data_overview):
    # perform tokenization and assign number to each word, numbers are the position in the sparse vector
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(data_combine['combine'])
    # count_matrix contains for each document the number encoded for the specific word and number of occurence in the document
    # for example "(0, 38)	1" 38 is assigned to the word 'action' and it is present one time in the first document

    # use term frequency inverse document frequency to know the most significant words and see which movie has the same
    # most significant word
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data_overview['overview'].values.astype('U'))

    # we combine the two matrix  and calculate the cosinus between vectors A and B
    # to calculate and find the most similar movie to the user input
    combine_sparse = sp.hstack([count_matrix, tfidf_matrix], format='csr')
    cosine_sim = cosine_similarity(combine_sparse, combine_sparse)

    return cosine_sim


def recommend_movies(title, data, combine, transform):
    # assign an index for each movie present in the dataset
    indices = pd.Series(data.index, index=data['original_title'])
    # index contains the user input movie index
    index = indices[title]
    # get the similarity score between user input and movies in the dataset
    sim_scores = list(enumerate(transform[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # keep only the eleven first movies but not the first one of the eleven because
    # the first one is the user input movie itself
    sim_scores = sim_scores[1:11]

    movie_indices = [i[0] for i in sim_scores]

    movie_id = data['movie_id'].iloc[movie_indices]
    movie_title = data['original_title'].iloc[movie_indices]
    movie_genres = data['genres'].iloc[movie_indices]

    recommendation_data = pd.DataFrame(columns=['Movie_Id', 'Name', 'Genres'])

    recommendation_data['Movie_Id'] = movie_id
    recommendation_data['Name'] = movie_title
    recommendation_data['Genres'] = movie_genres

    return recommendation_data


def results(movie_name):
    movie_name = movie_name.lower()
    find_movie = get_data()
    if movie_name not in find_movie['original_title'].unique():
        return 'Movie not in Database'
    else:
        combine_result = combine_data(find_movie)
        transform_result = transform_data(combine_result, find_movie)
        recommendations = recommend_movies(movie_name, find_movie, combine_result, transform_result)
        return recommendations.to_dict('records')

# return the list of the first 10 recommended movies
def get_name(result):
    lists = []
    for i in range(len(result)):
        lists.append(result[i]['Name'].title())
    return lists