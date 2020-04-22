from builtins import print

from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.neighbors import NearestNeighbors
# import pickle
# from moviesRecommendationSystem import matrix_factorization_utilities
# import scipy.sparse as sp

ratings = pd.read_csv('../Dataset/ratings.csv')
movie_list = pd.read_csv('../Dataset/movies.csv')
tags = pd.read_csv('../Dataset/tags.csv')
ratings = ratings[['userId', 'movieId','rating']]
# print(ratings)

ratings_df = ratings.groupby(['userId','movieId']).aggregate(np.max)
# print(ratings_df)
# print(ratings.head())

count_ratings = ratings.groupby('rating').count()
count_ratings['perc_total']=round(count_ratings['userId']*100/count_ratings['userId'].sum(),1)

# print(count_ratings)

genres = movie_list['genres']

# print(str(genres[0]))

genre_list = ""
for index,row in movie_list.iterrows():
        genre_list += row.genres + "|"
#split the string into a list of values
genre_list_split = genre_list.split('|')
#de-duplicate values
new_list = list(set(genre_list_split))
#remove the value that is blank
new_list.remove('')
#inspect list of genres
# print(new_list)

movies_with_genres = movie_list.copy()

for genre in new_list :
    movies_with_genres[genre] = movies_with_genres.apply(lambda _:int(genre in _.genres), axis = 1)

# print(movies_with_genres.head())

no_of_users = len(ratings['userId'].unique())
no_of_movies = len(ratings['movieId'].unique())

sparsity = round(1.0 - len(ratings)/(1.0*(no_of_movies*no_of_users)),3)
# print(sparsity)

avg_movie_rating = pd.DataFrame(ratings.groupby('movieId')['rating'].agg(['mean','count']))
# print(avg_movie_rating.head())

#Get the average movie rating across all movies
avg_rating_all=ratings['rating'].mean()
# print(avg_rating_all)
#set a minimum threshold for number of reviews that the movie has to have
min_reviews=30
# print(min_reviews)
movie_score = avg_movie_rating.loc[avg_movie_rating['count']>min_reviews]
# print(movie_score.head())

def weighted_rating(x, m=min_reviews, C=avg_rating_all):
    v = x['count']
    R = x['mean']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)

movie_score['weighted_score'] = movie_score.apply(weighted_rating, axis=1)
# print(movie_score.head())

#join movie details to movie ratings
movie_score = pd.merge(movie_score,movies_with_genres,on='movieId')
#join movie links to movie ratings
#movie_score = pd.merge(movie_score,links,on='movieId')
# print(movie_score.head())

def best_movies_by_genre(genre,top_n):
    return pd.DataFrame(movie_score.loc[(movie_score[genre]==1)].sort_values(['weighted_score'],ascending=False)[['title','count','mean','weighted_score']][:top_n])

# Consider this one to get movies id.....
# print(str(best_movies_by_genre('Musical',10)).split('\n')[10].split(' ')[0])


# print(str(best_movies_by_genre('Musical',10).title).split('\n')[1].split(' ')[0])
# print(str(best_movies_by_genre('Musical',10).title[0]))

ratings_df = pd.pivot_table(ratings, index='userId', columns='movieId', aggfunc=np.max)
# print(ratings_df.head())

ratings_movies = pd.merge(ratings,movie_list, on = 'movieId')
# print(ratings_movies.head())

def get_other_movies(movie_name):
    #get all users who watched a specific movie
    df_movie_users_series = ratings_movies.loc[ratings_movies['title']==movie_name]['userId']
    #convert to a data frame
    df_movie_users = pd.DataFrame(df_movie_users_series,columns=['userId'])
    #get a list of all other movies watched by these users
    other_movies = pd.merge(df_movie_users,ratings_movies,on='userId')
    #get a list of the most commonly watched movies by these other user
    other_users_watched = pd.DataFrame(other_movies.groupby('title')['userId'].count()).sort_values('userId',ascending=False)
    other_users_watched['perc_who_watched'] = round(other_users_watched['userId']*100/other_users_watched['userId'][0],1)
    return other_users_watched[:10]

temp = get_other_movies('Gone Girl (2014)')
print(temp)
# ind = str(temp).index('userId')
# # print(str(temp).split('\n')[3][:ind].strip())
#
# movie_plus_10_ratings = avg_movie_rating.loc[avg_movie_rating['count']>=10]
# # print(len(movie_plus_10_ratings))
#
# filtered_ratings = pd.merge(movie_plus_10_ratings, ratings, on="movieId")
# # print(len(filtered_ratings))
#
# movie_wide = filtered_ratings.pivot(index = 'movieId', columns = 'userId', values = 'rating').fillna(0)
# # print(movie_wide.head())
#
# #specify model parameters
# model_knn = NearestNeighbors(metric='cosine',algorithm='brute')
# #fit model to the data set
# model_knn.fit(movie_wide)
#
# def print_similar_movies(query_index) :
#     get_movies = []
#     #get the list of user ratings for a specific userId
#     query_index_movie_ratings = movie_wide.loc[query_index,:].values.reshape(1,-1)
#     #get the closest 10 movies and their distances from the movie specified
#     distances,indices = model_knn.kneighbors(query_index_movie_ratings,n_neighbors = 11)
#     #write a lopp that prints the similar movies for a specified movie.
#     for i in range(0,len(distances.flatten())):
#         #get the title of the random movie that was chosen
# #         get_movie = movie_list.loc[movie_list['movieId']==query_index]['title']
#         #for the first movie in the list i.e closest print the title
# #         if i==0:
# #             print('Recommendations for {0}:\n'.format(get_movie))
# #         else :
#         #get the indiciees for the closest movies
#         indices_flat = indices.flatten()[i]
#         #get the title of the movie
#         get_movie = movie_list.loc[movie_list['movieId']==movie_wide.iloc[indices_flat,:].name]
#         #print the movie
#         get_movies.append(get_movie)
#     return get_movies
#
# # returns 10 similar movies..
# a = print_similar_movies(112552)[10]
# # print(str(a.title).split(' ')[0])
#
# movie_content_df_temp = movies_with_genres.copy()
# movie_content_df_temp.set_index('movieId')
# movie_content_df = movie_content_df_temp.drop(columns = ['movieId','title','genres'])
# movie_content_df = movie_content_df.values
# # print(movie_content_df)
#
# cosine_sim = linear_kernel(movie_content_df,movie_content_df)
# # for i in range(len(cosine_sim)):
# #     print(cosine_sim[i])
# indicies = pd.Series(movie_content_df_temp.index, movie_content_df_temp['title'])
# # print(indicies)
#
#
# def get_similar_movies_based_on_content(movie_index):
#     sim_scores = list(enumerate(cosine_sim[movie_index]))
#     # Sort the movies based on the similarity scores
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#
#     # Get the scores of the 10 most similar movies
#     sim_scores = sim_scores[0:11]
#     print(sim_scores)
#     # Get the movie indices
#     movie_indices = [i[0] for i in sim_scores]
#     print(movie_indices)
#     similar_movies = pd.DataFrame(movie_content_df_temp[['title', 'genres']].iloc[movie_indices])
#     return similar_movies
#
# # print("Similar movies : ",get_similar_movies_based_on_content(19338))
#
#
# ### for tomorrow ###
#
# #get ordered list of movieIds
# item_indices = pd.DataFrame(sorted(list(set(ratings['movieId']))),columns=['movieId'])
# #add in data frame index value to data frame
# item_indices['movie_index']=item_indices.index
# #inspect data frame
# print("item_indices.head() : ",item_indices.head())
#
# #get ordered list of movieIds
# user_indices = pd.DataFrame(sorted(list(set(ratings['userId']))),columns=['userId'])
# #add in data frame index value to data frame
# user_indices['user_index']=user_indices.index
# #inspect data frame
# print("user_indices.head() : ",user_indices.head())
#
# # join the movie indices
# df_with_index = pd.merge(ratings,item_indices,on='movieId')
# #join the user indices
# df_with_index=pd.merge(df_with_index,user_indices,on='userId')
# #inspec the data frame
# print("df_with_index.head() : ",df_with_index.head())
#
# #import train_test_split module
# #take 80% as the training set and 20% as the test set
# df_train, df_test= train_test_split(df_with_index,test_size=0.2)
# print(len(df_train))
# print(len(df_test))
#
# # 50
# n_users = ratings.userId.unique().shape[0]
# n_items = ratings.movieId.unique().shape[0]
# print(n_users)
# print(n_items)
#
# # 59
# train_data_matrix = np.zeros((n_users, n_items))
#     #for every line in the data
# for line in df_train.itertuples():
#     #set the value in the column and row to
#     #line[1] is userId, line[2] is movieId and line[3] is rating, line[4] is movie_index and line[5] is user_index
#     train_data_matrix[line[5], line[4]] = line[3]
# print("train_data_matrix.shape : ",train_data_matrix.shape)
#
# # 60
# #Create two user-item matrices, one for training and another for testing
# test_data_matrix = np.zeros((n_users, n_items))
#     #for every line in the data
# for line in df_test[:1].itertuples():
#     #set the value in the column and row to
#     #line[1] is userId, line[2] is movieId and line[3] is rating, line[4] is movie_index and line[5] is user_index
#     #print(line[2])
#     test_data_matrix[line[5], line[4]] = line[3]
#     #train_data_matrix[line['movieId'], line['userId']] = line['rating']
# print("test_data_matrix.shape : ",test_data_matrix.shape)
#
# from sklearn.metrics import mean_squared_error
# from math import sqrt
# def rmse(prediction, ground_truth):
#     #select prediction values that are non-zero and flatten into 1 array
#     prediction = prediction[ground_truth.nonzero()].flatten()
#     #select test values that are non-zero and flatten into 1 array
#     ground_truth = ground_truth[ground_truth.nonzero()].flatten()
#     #return RMSE between values
#     return sqrt(mean_squared_error(prediction, ground_truth))
#
# rmse_list = []
# for i in [1,2,5,20,40,60,100,200]:
#     #apply svd to the test data
#     u,s,vt = svds(train_data_matrix,k=i)
#     #get diagonal matrix
#     s_diag_matrix=np.diag(s)
#     #predict x with dot product of u s_diag and vt
#     X_pred = np.dot(np.dot(u,s_diag_matrix),vt)
#     #calculate rmse score of matrix factorisation predictions
#     rmse_score = rmse(X_pred,test_data_matrix)
#     rmse_list.append(rmse_score)
#     print("Matrix Factorisation with " + str(i) +" latent features has a RMSE of " + str(rmse_score))
#
# mf_pred = pd.DataFrame(X_pred)
# print("mf_pred.head() : ",mf_pred.head())
#
# df_names = pd.merge(ratings,movie_list,on='movieId')
# print("df_names.head() : ",df_names.head())
#
# #choose a user ID
#
# # def getMovies(user_id):
# #get movies rated by this user id
# user_id = 2
# users_movies = df_names.loc[df_names["userId"]==user_id]
# #print how many ratings user has made
# print("User ID : " + str(user_id) + " has already rated " + str(len(users_movies)) + " movies")
# #list movies that have been rated
# print("users_movies : ",users_movies)
#
# user_index = df_train.loc[df_train["userId"]==user_id]['user_index'][:1].values[0]
# #get movie ratings predicted for this user and sort by hig')
# print(str(temp).split('\n')[3][:ind].strip())

movie_plus_10_ratings = avg_movie_rating.loc[avg_movie_rating['count']>=10]
# print(len(movie_plus_10_ratings))

filtered_ratings = pd.merge(movie_plus_10_ratings, ratings, on="movieId")
# print(len(filtered_ratings))

movie_wide = filtered_ratings.pivot(index = 'movieId', columns = 'userId', values = 'rating').fillna(0)
# print(movie_wide.head())

#specify model parameters
model_knn = NearestNeighbors(metric='cosine',algorithm='brute')
#fit model to the data set
model_knn.fit(movie_wide)

def print_similar_movies(query_index) :
    get_movies = []
    #get the list of user ratings for a specific userId
    query_index_movie_ratings = movie_wide.loc[query_index,:].values.reshape(1,-1)
    #get the closest 10 movies and their distances from the movie specified
    distances,indices = model_knn.kneighbors(query_index_movie_ratings,n_neighbors = 11)
    #write a lopp that prints the similar movies for a specified movie.
    for i in range(0,len(distances.flatten())):
        #get the title of the random movie that was chosen
#         get_movie = movie_list.loc[movie_list['movieId']==query_index]['title']
        #for the first movie in the list i.e closest print the title
#         if i==0:
#             print('Recommendations for {0}:\n'.format(get_movie))
#         else :
        #get the indiciees for the closest movies
        indices_flat = indices.flatten()[i]
        #get the title of the movie
        get_movie = movie_list.loc[movie_list['movieId']==movie_wide.iloc[indices_flat,:].name]
        #print the movie
        get_movies.append(get_movie)
    return get_movies

# returns 10 similar movies..
a = print_similar_movies(112552)[10]
# print(str(a.title).split(' ')[0])

movie_content_df_temp = movies_with_genres.copy()
movie_content_df_temp.set_index('movieId')
movie_content_df = movie_content_df_temp.drop(columns = ['movieId','title','genres'])
movie_content_df = movie_content_df.values
# print(movie_content_df)

cosine_sim = linear_kernel(movie_content_df,movie_content_df)
# for i in range(len(cosine_sim)):
#     print(cosine_sim[i])
indicies = pd.Series(movie_content_df_temp.index, movie_content_df_temp['title'])
# print(indicies)


def get_similar_movies_based_on_content(movie_index):
    sim_scores = list(enumerate(cosine_sim[movie_index]))
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[0:11]
    print(sim_scores)
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    print(movie_indices)
    similar_movies = pd.DataFrame(movie_content_df_temp[['title', 'genres']].iloc[movie_indices])
    return similar_movies

# print("Similar movies : ",get_similar_movies_based_on_content(19338))


### for tomorrow ###

#get ordered list of movieIds
item_indices = pd.DataFrame(sorted(list(set(ratings['movieId']))),columns=['movieId'])
#add in data frame index value to data frame
item_indices['movie_index']=item_indices.index
#inspect data frame
print("item_indices.head() : ",item_indices.head())

#get ordered list of movieIds
user_indices = pd.DataFrame(sorted(list(set(ratings['userId']))),columns=['userId'])
#add in data frame index value to data frame
user_indices['user_index']=user_indices.index
#inspect data frame
print("user_indices.head() : ",user_indices.head())

# join the movie indices
df_with_index = pd.merge(ratings,item_indices,on='movieId')
#join the user indices
df_with_index=pd.merge(df_with_index,user_indices,on='userId')
#inspec the data frame
print("df_with_index.head() : ",df_with_index.head())

#import train_test_split module
#take 80% as the training set and 20% as the test set
df_train, df_test= train_test_split(df_with_index,test_size=0.2)
print(len(df_train))
print(len(df_test))

# 50
n_users = ratings.userId.unique().shape[0]
n_items = ratings.movieId.unique().shape[0]
print(n_users)
print(n_items)

# 59
train_data_matrix = np.zeros((n_users, n_items))
    #for every line in the data
for line in df_train.itertuples():
    #set the value in the column and row to
    #line[1] is userId, line[2] is movieId and line[3] is rating, line[4] is movie_index and line[5] is user_index
    train_data_matrix[line[5], line[4]] = line[3]
print("train_data_matrix.shape : ",train_data_matrix.shape)

# 60
#Create two user-item matrices, one for training and another for testing
test_data_matrix = np.zeros((n_users, n_items))
    #for every line in the data
for line in df_test[:1].itertuples():
    #set the value in the column and row to
    #line[1] is userId, line[2] is movieId and line[3] is rating, line[4] is movie_index and line[5] is user_index
    #print(line[2])
    test_data_matrix[line[5], line[4]] = line[3]
    #train_data_matrix[line['movieId'], line['userId']] = line['rating']
print("test_data_matrix.shape : ",test_data_matrix.shape)

from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction, ground_truth):
    #select prediction values that are non-zero and flatten into 1 array
    prediction = prediction[ground_truth.nonzero()].flatten()
    #select test values that are non-zero and flatten into 1 array
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    #return RMSE between values
    return sqrt(mean_squared_error(prediction, ground_truth))

rmse_list = []
for i in [1,2,5,20,40,60,100,200]:
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

mf_pred = pd.DataFrame(X_pred)
print("mf_pred.head() : ",mf_pred.head())

df_names = pd.merge(ratings,movie_list,on='movieId')
print("df_names.head() : ",df_names.head())

#choose a user ID

# def getMovies(user_id):
#get movies rated by this user id
user_id = 2
users_movies = df_names.loc[df_names["userId"]==user_id]
#print how many ratings user has made
print("User ID : " + str(user_id) + " has already rated " + str(len(users_movies)) + " movies")
#list movies that have been rated
print("users_movies : ",users_movies)

user_index = df_train.loc[df_train["userId"]==user_id]['user_index'][:1].values[0]
#get movie ratings predicted for this user and sort by highest rating prediction
sorted_user_predictions = pd.DataFrame(mf_pred.iloc[user_index].sort_values(ascending=False))
#rename the columns
sorted_user_predictions.columns=['ratings']
#save the index values as movie id
sorted_user_predictions['movieId']=sorted_user_predictions.index
print("Top 10 predictions for User " + str(user_id))
#display the top 10 predictions for this user
print(pd.merge(sorted_user_predictions,movie_list, on = 'movieId')[:10])
# print(str(getMovies(2)).split('\n')[0])

print("loaded")
# hest rating prediction
# sorted_user_predictions = pd.DataFrame(mf_pred.iloc[user_index].sort_values(ascending=False))
# #rename the columns
# sorted_user_predictions.columns=['ratings']
# #save the index values as movie id
# sorted_user_predictions['movieId']=sorted_user_predictions.index
# print("Top 10 predictions for User " + str(user_id))
# #display the top 10 predictions for this user
# print(pd.merge(sorted_user_predictions,movie_list, on = 'movieId')[:10])
# # print(str(getMovies(2)).split('\n')[0])
#
# print("loaded")
