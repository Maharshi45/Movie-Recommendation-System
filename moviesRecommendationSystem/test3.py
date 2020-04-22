import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
# import pickle
# from moviesRecommendationSystem import matrix_factorization_utilities
# import scipy.sparse as sp
from scipy.sparse.linalg import svds

ratings = pd.read_csv('../Dataset/ratings.csv')
ratings = ratings[['userId', 'movieId', 'rating']]
ratings_df = ratings.groupby(['userId', 'movieId']).aggregate(np.max)

count_ratings = ratings.groupby('rating').count()
count_ratings['perc_total'] = round(count_ratings['userId'] * 100 / count_ratings['userId'].sum(), 1)

movie_list = pd.read_csv('../Dataset/movies.csv')
tags = pd.read_csv('../Dataset/tags.csv')

genre_list = ""
for index, row in movie_list.iterrows():
    genre_list += row.genres + "|"
# split the string into a list of values
genre_list_split = genre_list.split('|')
# de-duplicate values
new_list = list(set(genre_list_split))
# remove the value that is blank
new_list.remove('')
# inspect list of genres
# new_list

# Enriching the movies dataset by adding the various genres columns.
movies_with_genres = movie_list.copy()

for genre in new_list:
    movies_with_genres[genre] = movies_with_genres.apply(lambda _: int(genre in _.genres), axis=1)

# Calculating the sparsity
no_of_users = len(ratings['userId'].unique())
no_of_movies = len(ratings['movieId'].unique())

sparsity = round(1.0 - len(ratings) / (1.0 * (no_of_movies * no_of_users)), 3)
# print(sparsity)

# Finding the average rating for movie and the number of ratings for each movie
avg_movie_rating = pd.DataFrame(ratings.groupby('movieId')['rating'].agg(['mean', 'count']))
# avg_movie_rating['movieId']= avg_movie_rating.index

# Get the average movie rating across all movies
avg_rating_all = ratings['rating'].mean()
# avg_rating_all
# set a minimum threshold for number of reviews that the movie has to have
min_reviews = 1
# min_reviews
movie_score = avg_movie_rating.loc[avg_movie_rating['count'] > min_reviews]


# movie_score.head()

# create a function for weighted rating score based off count of reviews
def weighted_rating(x, m=min_reviews, C=avg_rating_all):
    v = x['count']
    R = x['mean']
    # Calculation based on the IMDB formula
    return (v / (v + m) * R) + (m / (m + v) * C)


# Calculating the weighted score for each movie
movie_score['weighted_score'] = movie_score.apply(weighted_rating, axis=1)
movie_score.head()

# join movie details to movie ratings
movie_score = pd.merge(movie_score, movies_with_genres, on='movieId')
# join movie links to movie ratings
# movie_score = pd.merge(movie_score,links,on='movieId')
# movie_score.head()


# list top scored movies over the whole range of movies
# print(pd.DataFrame(movie_score.sort_values(['weighted_score'], ascending=False)[
#                        ['title', 'count', 'mean', 'weighted_score', 'genres']][:10]))

# # Gives the best movies according to genre based on weighted score which is calculated using IMDB formula
# def best_movies_by_genre(genre,top_n):
#     return pd.DataFrame(movie_score.loc[(movie_score[genre]==1)].sort_values(['weighted_score'],ascending=False)[['title','count','mean','weighted_score']][:top_n])
#
# #run function to return top recommended movies by genre
# print(best_movies_by_genre('Comedy',10))

ratings_df = pd.pivot_table(ratings, index='userId', columns='movieId', aggfunc=np.max)
# print(ratings_df.head())

ratings_movies = pd.merge(ratings, movie_list, on='movieId')


# print(ratings_movies.head())

def get_other_movies(movie_name):
    # get all users who watched a specific movie
    df_movie_users_series = ratings_movies.loc[ratings_movies['title'] == movie_name]['userId']
    # convert to a data frame
    df_movie_users = pd.DataFrame(df_movie_users_series, columns=['userId'])
    # get a list of all other movies watched by these users
    other_movies = pd.merge(df_movie_users, ratings_movies, on='userId')
    # get a list of the most commonly watched movies by these other user
    other_users_watched = pd.DataFrame(other_movies.groupby('title')['userId'].count()).sort_values('userId',
                                                                                                    ascending=False)
    other_users_watched['perc_who_watched'] = round(
        other_users_watched['userId'] * 100 / other_users_watched['userId'][0], 1)
    return other_users_watched[:10]




# temp = get_other_movies('Die Hard 2 (1990)')
#
# def getMovieData(temp):
#     temp = str(temp).split('\n')
#     # print(temp[3])
#     temp2 = []
#     for i in range(3, len(temp)):
#         temp3 = temp[i].split()[0:-2]
#         temp4 = ""
#         for j in temp3:
#             temp4 = temp4 + " " + j
#         temp2.append(temp4.strip())
#     return temp2
#     # print(str(temp).split('\n')[3].split()[0:-2])
#
# print(getMovieData(temp))



movie_plus_10_ratings = avg_movie_rating.loc[avg_movie_rating['count']>=10]

filtered_ratings = pd.merge(movie_plus_10_ratings, ratings, on="movieId")

movie_wide = filtered_ratings.pivot(index = 'movieId', columns = 'userId', values = 'rating').fillna(0)
# movie_wide.head()

model_knn = NearestNeighbors(metric='cosine',algorithm='brute')
model_knn.fit(movie_wide)



movie_content_df_temp = movies_with_genres.copy()
movie_content_df_temp.set_index('movieId')
movie_content_df = movie_content_df_temp.drop(columns = ['movieId','title','genres'])
movie_content_df = movie_content_df.values
# movie_content_df




cosine_sim = linear_kernel(movie_content_df,movie_content_df)

# for i in range(len(cosine_sim)):
#     print(cosine_sim[i])

# indicies = pd.Series(movie_content_df_temp.index, movie_content_df_temp['title'])


#get ordered list of movieIds
item_indices = pd.DataFrame(sorted(list(set(ratings['movieId']))),columns=['movieId'])
#add in data frame index value to data frame
item_indices['movie_index']=item_indices.index
#inspect data frame
item_indices.head()


#get ordered list of movieIds
user_indices = pd.DataFrame(sorted(list(set(ratings['userId']))),columns=['userId'])
#add in data frame index value to data frame
user_indices['user_index']=user_indices.index
#inspect data frame
user_indices.head()



#join the movie indices
df_with_index = pd.merge(ratings,item_indices,on='movieId')
#join the user indices
df_with_index=pd.merge(df_with_index,user_indices,on='userId')
#inspec the data frame
df_with_index.head()




#import train_test_split module
#take 80% as the training set and 20% as the test set
df_train, df_test= train_test_split(df_with_index,test_size=0.2)
# print(len(df_train))
# print(len(df_test))


n_users = ratings.userId.unique().shape[0]
n_items = ratings.movieId.unique().shape[0]
# print(n_users)
# print(n_items)


#Create two user-item matrices, one for training and another for testing
train_data_matrix = np.zeros((n_users, n_items))
    #for every line in the data
for line in df_train.itertuples():
    #set the value in the column and row to
    #line[1] is userId, line[2] is movieId and line[3] is rating, line[4] is movie_index and line[5] is user_index
    train_data_matrix[line[5], line[4]] = line[3]
# train_data_matrix.shape

#Create two user-item matrices, one for training and another for testing
test_data_matrix = np.zeros((n_users, n_items))
    #for every line in the data
for line in df_test[:1].itertuples():
    #set the value in the column and row to
    #line[1] is userId, line[2] is movieId and line[3] is rating, line[4] is movie_index and line[5] is user_index
    #print(line[2])
    test_data_matrix[line[5], line[4]] = line[3]
    #train_data_matrix[line['movieId'], line['userId']] = line['rating']
# test_data_matrix.shape




def rmse(prediction, ground_truth):
    #select prediction values that are non-zero and flatten into 1 array
    prediction = prediction[ground_truth.nonzero()].flatten()
    #select test values that are non-zero and flatten into 1 array
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    #return RMSE between values
    return sqrt(mean_squared_error(prediction, ground_truth))


#Calculate the rmse sscore of SVD using different values of k (latent features)
rmse_list = []
for i in [1,2,5,20,40]:
    #apply svd to the test data
    u,s,vt = svds(train_data_matrix,k=i)
    #get diagonal matrix
    s_diag_matrix=np.diag(s)
    #predict x with dot product of u s_diag and vt
    X_pred = np.dot(np.dot(u,s_diag_matrix),vt)
    #calculate rmse score of matrix factorisation predictions
    rmse_score = rmse(X_pred,test_data_matrix)
    rmse_list.append(rmse_score)
    print("Matrix Factorisation with " + str(i) +" latent features has a RMSE of " + str(rmse_score))



#Convert predictions to a DataFrame
mf_pred = pd.DataFrame(X_pred)
# mf_pred.head()


df_names = pd.merge(ratings,movie_list,on='movieId')
# df_names.head()


#choose a user ID

def getRecommendedMovies(user_id):
    #get movies rated by this user id
    users_movies = df_names.loc[df_names["userId"]==user_id]
    #print how many ratings user has made
    print("User ID : " + str(user_id) + " has already rated " + str(len(users_movies)) + " movies")
    #list movies that have been rated
    # users_movies

    user_index = df_train.loc[df_train["userId"]==user_id]['user_index'][:1].values[0]
    #get movie ratings predicted for this user and sort by highest rating prediction
    sorted_user_predictions = pd.DataFrame(mf_pred.iloc[user_index].sort_values(ascending=False))
    #rename the columns
    sorted_user_predictions.columns=['ratings']
    #save the index values as movie id
    sorted_user_predictions['movieId']=sorted_user_predictions.index
    print("Top 10 predictions for User " + str(user_id))
    #display the top 10 predictions for this user
    return pd.merge(sorted_user_predictions,movie_list, on = 'movieId')[:10]

result = getRecommendedMovies(2)
print(result.movieId[0])