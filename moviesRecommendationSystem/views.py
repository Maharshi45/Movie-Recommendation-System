from django.shortcuts import render
from django.shortcuts import redirect
from moviesRecommendationSystem.models import user, watchlist, requestMovieModel
# from builtins import print

from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.neighbors import NearestNeighbors
from math import sqrt
from sklearn.metrics import mean_squared_error

# from moviesRecommendationSystem.test import get_similar_movies_based_on_content
ratings = pd.read_csv('./Dataset/ratings.csv')
movie_list = pd.read_csv('./Dataset/movies.csv')
tags = pd.read_csv('./Dataset/tags.csv')

ratings = ratings[['userId', 'movieId', 'rating']]
movie_list1 = movie_list[['movieId', 'genres', 'title']]
ratings1 = ratings[['userId', 'movieId', 'rating']]

ratings_df = ratings.groupby(['userId', 'movieId']).aggregate(np.max)

count_ratings = ratings.groupby('rating').count()
count_ratings['perc_total'] = round(count_ratings['userId'] * 100 / count_ratings['userId'].sum(), 1)

genres = movie_list['genres']

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

movies_with_genres = movie_list.copy()

for genre in new_list:
    movies_with_genres[genre] = movies_with_genres.apply(lambda _: int(genre in _.genres), axis=1)

no_of_users = len(ratings['userId'].unique())
no_of_movies = len(ratings['movieId'].unique())

sparsity = round(1.0 - len(ratings) / (1.0 * (no_of_movies * no_of_users)), 3)

avg_movie_rating = pd.DataFrame(ratings.groupby('movieId')['rating'].agg(['mean', 'count']))

# Get the average movie rating across all movies
avg_rating_all = ratings['rating'].mean()

# set a minimum threshold for number of reviews that the movie has to have
min_reviews = 0

movie_score = avg_movie_rating.loc[avg_movie_rating['count'] > min_reviews]


def weighted_rating(x, m=min_reviews, C=avg_rating_all):
    v = x['count']
    R = x['mean']
    # Calculation based on the IMDB formula
    return (v / (v + m) * R) + (m / (m + v) * C)


movie_score['weighted_score'] = movie_score.apply(weighted_rating, axis=1)

# join movie details to movie ratings
movie_score = pd.merge(movie_score, movies_with_genres, on='movieId')

pd.DataFrame(movie_score.sort_values(['weighted_score'], ascending=False)[
                 ['title', 'count', 'mean', 'weighted_score', 'genres']][:10])


def best_movies_by_genre(genre, top_n):
    return pd.DataFrame(movie_score.loc[(movie_score[genre] == 1)].sort_values(['weighted_score'], ascending=False)[
                            ['title', 'count', 'mean', 'weighted_score']][:top_n])


ratings_df = pd.pivot_table(ratings, index='userId', columns='movieId', aggfunc=np.max)

ratings_movies = pd.merge(ratings, movie_list, on='movieId')


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
    return other_users_watched


movie_plus_10_ratings = avg_movie_rating.loc[avg_movie_rating['count'] >= 10]

filtered_ratings = pd.merge(movie_plus_10_ratings, ratings, on="movieId")

movie_wide = filtered_ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)

model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(movie_wide)

movie_content_df_temp = movies_with_genres.copy()
movie_content_df_temp.set_index('movieId')
movie_content_df = movie_content_df_temp.drop(columns=['movieId', 'title', 'genres'])
movie_content_df = movie_content_df.values

cosine_sim = linear_kernel(movie_content_df, movie_content_df)

# get ordered list of movieIds
item_indices = pd.DataFrame(sorted(list(set(ratings['movieId']))), columns=['movieId'])
# add in data frame index value to data frame
item_indices['movie_index'] = item_indices.index
# inspect data frame
item_indices.head()

# get ordered list of movieIds
user_indices = pd.DataFrame(sorted(list(set(ratings['userId']))), columns=['userId'])
# add in data frame index value to data frame
user_indices['user_index'] = user_indices.index
# inspect data frame
user_indices.head()

# join the movie indices
df_with_index = pd.merge(ratings, item_indices, on='movieId')
# join the user indices
df_with_index = pd.merge(df_with_index, user_indices, on='userId')
# inspec the data frame
df_with_index.head()

# import train_test_split module
# take 80% as the training set and 20% as the test set
df_train, df_test = train_test_split(df_with_index, test_size=0.2)

n_users = ratings.userId.unique().shape[0]
n_items = ratings.movieId.unique().shape[0]

# Create two user-item matrices, one for training and another for testing
train_data_matrix = np.zeros((n_users, n_items))
# for every line in the data
for line in df_train.itertuples():
    # set the value in the column and row to
    # line[1] is userId, line[2] is movieId and line[3] is rating, line[4] is movie_index and line[5] is user_index
    train_data_matrix[line[5], line[4]] = line[3]
# train_data_matrix.shape

# Create two user-item matrices, one for training and another for testing
test_data_matrix = np.zeros((n_users, n_items))
# for every line in the data
for line in df_test[:1].itertuples():
    # set the value in the column and row to
    # line[1] is userId, line[2] is movieId and line[3] is rating, line[4] is movie_index and line[5] is user_index
    test_data_matrix[line[5], line[4]] = line[3]
    # train_data_matrix[line['movieId'], line['userId']] = line['rating']


def rmse(prediction, ground_truth):
    # select prediction values that are non-zero and flatten into 1 array
    prediction = prediction[ground_truth.nonzero()].flatten()
    # select test values that are non-zero and flatten into 1 array
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    # return RMSE between values
    return sqrt(mean_squared_error(prediction, ground_truth))


# Calculate the rmse sscore of SVD using different values of k (latent features)
rmse_list = []
for i in [1, 2, 5, 20, 40]:
    # apply svd to the test data
    u, s, vt = svds(train_data_matrix, k=i)
    # get diagonal matrix
    s_diag_matrix = np.diag(s)
    # predict x with dot product of u s_diag and vt
    X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
    # calculate rmse score of matrix factorisation predictions
    rmse_score = rmse(X_pred, test_data_matrix)
    rmse_list.append(rmse_score)

# Convert predictions to a DataFrame
mf_pred = pd.DataFrame(X_pred)

df_names = pd.merge(ratings, movie_list, on='movieId')


# choose a user ID

def getRecommendedMovies(user_id):
    # get movies rated by this user id
    users_movies = df_names.loc[df_names["userId"] == user_id]

    user_index = df_train.loc[df_train["userId"] == user_id]['user_index'][:1].values[0]
    # get movie ratings predicted for this user and sort by highest rating prediction
    sorted_user_predictions = pd.DataFrame(mf_pred.iloc[user_index].sort_values(ascending=False))
    # rename the columns
    sorted_user_predictions.columns = ['ratings']
    # save the index values as movie id
    sorted_user_predictions['movieId'] = sorted_user_predictions.index
    # display the top 10 predictions for this user
    return pd.merge(sorted_user_predictions, movie_list, on='movieId')[:10]


def home(request):
    if (request.session.get('id')):
        return redirect('profile')
    else:
        tempMovieNames = getMovieByData(pd.DataFrame(movie_score.sort_values(['weighted_score'], ascending=False)[
                                                         ['title', 'count', 'mean', 'weighted_score', 'genres']][:10]))
        movieNames = []

        for movies in tempMovieNames:
            movieNames.append(movies[1:])

        movieNames1 = []
        movies_index = []
        genres = []
        ratings = []
        for movie in movieNames:
            for i in range(len(movie_list1)):
                if movie_list1.title[i] == movie:
                    movies_index.append(movie_list1.movieId[i])
                    genres.append(movie_list1.genres[i])
                    movieNames1.append(movie_list1.title[i])
                    break

        for i in range(len(movies_index)):
            count = 0
            sum = 0
            for j in range(len(ratings1)):
                if ratings1.movieId[j] == movies_index[i]:
                    sum = sum + ratings1.rating[j]
                    count = count + 1

            if sum == 0 or count == 0:
                ratings.append(0)
            else:
                ratings.append(round(sum / count, 1))

        data = []
        for i in range(len(ratings)):
            Dict = {}
            Dict['rating'] = ratings[i]
            Dict['genre'] = genres[i]
            Dict['movie'] = movieNames[i]
            Dict['index'] = movies_index[i]
            data.append(Dict)
        return render(request, 'home.html', {'data': data})


def doLogin(request):
    if (request.session.get('id')):
        return redirect(profile)

    else:
        return redirect(login)


def profile(request):
    if (request.session.get('id')):
        id = request.session['id']
        users = user.objects.filter(id=id)
        tempMovieNames = getMovieByData(pd.DataFrame(movie_score.sort_values(['weighted_score'], ascending=False)[
                                                         ['title', 'count', 'mean', 'weighted_score', 'genres']][:10]))

        movieNames = []

        for movies in tempMovieNames:
            movieNames.append(movies[1:])

        movieNames1 = []
        movies_index = []
        genres = []
        ratings = []
        for movie in movieNames:
            for i in range(len(movie_list1)):
                if movie_list1.title[i] == movie:
                    movies_index.append(movie_list1.movieId[i])
                    genres.append(movie_list1.genres[i])
                    movieNames1.append(movie_list1.title[i])
                    break

        for i in range(len(movies_index)):
            count = 0
            sum = 0
            for j in range(len(ratings1)):
                if ratings1.movieId[j] == movies_index[i]:
                    sum = sum + ratings1.rating[j]
                    count = count + 1

            if sum == 0 or count == 0:
                ratings.append(0)
            else:
                ratings.append(round(sum / count, 1))

        data = []
        for i in range(5):
            Dict = {}
            Dict['rating'] = ratings[i]
            Dict['genre'] = genres[i]
            Dict['movie'] = movieNames[i]
            Dict['index'] = movies_index[i]
            data.append(Dict)

        recommendedMoviesIndices = getRecommendedMovies(id).movieId

        recommendedMoviesNames = []
        recommendedMoviesGenres = []
        recommendedMoviesRatings = []

        for i in range(len(recommendedMoviesIndices)):
            for j in range(len(movie_list1)):
                if movie_list1.movieId[j] == recommendedMoviesIndices[i]:
                    recommendedMoviesNames.append(movie_list1.title[j])
                    recommendedMoviesGenres.append(movie_list1.genres[j])
                    break

        for i in range(len(recommendedMoviesIndices)):
            count1 = 0
            sum1 = 0
            for j in range(len(ratings1)):
                if ratings1.movieId[j] == recommendedMoviesIndices[i]:
                    sum1 = sum1 + ratings1.rating[j]
                    count1 = count1 + 1

            if sum1 == 0 or count1 == 0:
                recommendedMoviesRatings.append(0)
            else:
                recommendedMoviesRatings.append(round(sum1 / count1, 1))

        recommendedData = []

        for k in range(5):
            recommendedDict = {}
            recommendedDict['rating'] = recommendedMoviesRatings[k]
            recommendedDict['genre'] = recommendedMoviesGenres[k]
            recommendedDict['movie'] = recommendedMoviesNames[k]
            recommendedDict['index'] = recommendedMoviesIndices[k]
            recommendedData.append(recommendedDict)

        return render(request, 'profile.html', {'user': users[0], 'data': data, 'recommendedData': recommendedData})

    else:
        return redirect(login)


def logout(request):
    if (request.session.get('id')):
        del request.session['id']
    return render(request, 'LogoutSuccessful.html')


def login(request):
    if request.method == 'POST':
        email = request.POST['email']
        password = request.POST['password']
        users = user.objects.filter(email=email, password=password)
        if (users):
            request.session['id'] = users[0].id
            return redirect(profile)
        else:
            return render(request, 'InvalidCredentials.html')

    elif request.method == "GET":
        return render(request, 'login.html')


def doRegister(request):
    if request.method == 'POST':
        firstName = request.POST.get('firstName')
        lastName = request.POST.get('lastName')
        email = request.POST['email']
        password = request.POST['password']
        users = user.objects.filter(email=email, password=password)
        for i in users:
            print(i.email)
        u = user()
        u.firstName = firstName
        u.lastName = lastName
        u.email = email
        u.password = password
        u.status = True
        u.save()
        if (request.session.get('id')):
            del request.session['id']
        return redirect(registered)


def registered(request):
    return render(request, 'registered.html')


def register(request):
    return render(request, 'register.html')


def getMovieData(temp):
    temp = str(temp).split('\n')

    temp2 = []
    for i in range(3, len(temp)):
        temp3 = temp[i].split()[0:-2]
        temp4 = ""
        for j in temp3:
            temp4 = temp4 + " " + j
        temp2.append(temp4.strip())
    return temp2


def getMovie(request):
    sum = 0
    count = 0
    index_movie = int(request.GET['movieId'])

    for i in range(len(movie_list1)):
        if movie_list1.movieId[i] == index_movie:
            genres1 = movie_list1.genres[i]
            movieName = movie_list1.title[i]
            break

    # tempSimilarMovies = get_other_movies(movieName)[:10]
    #
    # similarMovies = getMovieData(tempSimilarMovies)
    # indexesOfSimilarMovies = []
    #
    # for i in range(len(similarMovies)):
    #     for j in range(len(movie_list1)):
    #         if similarMovies[i] == movie_list1.title[j]:
    #             indexesOfSimilarMovies.append(movie_list1.movieId[j])
    #
    # data = []
    # for i in range(4):
    #     Dict = {}
    #     Dict['movie'] = similarMovies[i]
    #     Dict['index'] = indexesOfSimilarMovies[i]
    #     data.append(Dict)

    for i in range(len(ratings1)):
        if ratings1.movieId[i] == index_movie:
            sum = sum + ratings1.rating[i]
            count = count + 1

    if sum == 0 or count == 0:
        avg_rating = 0
    else:
        avg_rating = sum / count
    if (request.session.get('id')):
        return render(request, 'getMovieWithLogin.html',
                      {'movie_name': movieName, 'rating': round(avg_rating, 1), 'genres': genres1,
                       'movie_id': index_movie})

    else:
        return render(request, 'getMovieWithoutLogin.html',
                      {'movie_name': movieName, 'rating': round(avg_rating, 1), 'genres': genres1,
                       'movie_id': index_movie})


def addToWatchList(request):
    if (request.session.get('id')):
        id = int(request.session['id'])
        movieId = int(request.GET['movieId'])
        getMovieData = watchlist.objects.filter(movieId=movieId)

        flag = 0
        for movieData in getMovieData:
            if movieData.movieId == movieId:
                flag = 1
                break

        if flag == 0:
            w = watchlist()
            w.userId = id
            w.movieId = movieId
            w.save()
            return render(request, 'addedToWatchList.html')
        else:
            return render(request, 'alreadyAddedTOWatchlist.html')

    else:
        return redirect(login)


def getWatchList(request):
    if (request.session.get('id')):
        id = request.session['id']
        movies = watchlist.objects.filter(userId=id)

        avg_rating = []
        genre = []
        movie_name = []
        index_movie = []
        for movie in movies:
            for i in range(len(movie_list1)):
                if movie.movieId == movie_list1.movieId[i]:
                    index_movie.append(movie_list1.movieId[i])
                    movie_name.append(movie_list1.title[i])
                    genre.append(movie_list1.genres[i])

        for i in index_movie:
            count = 0
            sum = 0
            for j in range(len(ratings1)):
                if i == ratings1.movieId[j]:
                    count = count + 1
                    sum = sum + ratings1.rating[j]
            if (sum == 0):
                avg_rating.append(0)
            else:
                avg_rating.append(round(sum / count, 1))

        data = []
        for i in range(len(avg_rating)):
            Dict = {}
            Dict['rating'] = avg_rating[i]
            Dict['genre'] = genre[i]
            Dict['movie'] = movie_name[i]
            Dict['index'] = index_movie[i]
            data.append(Dict)

        return render(request, 'getWatchList.html', {'data': data})

    else:
        return redirect(login)


def requestMovie(request):
    if (request.session.get('id')):
        return render(request, 'requestMovie.html')
    else:
        return redirect(login)


def doRequestMovie(request):
    if request.method == 'POST':
        id = request.session['id']
        movie = request.POST['movie']

        rm = requestMovieModel()
        rm.userId = id
        rm.movieName = movie
        rm.save()
        return render(request, 'requestSubmitted.html')


def search(request):
    avg_rating = 0
    movieName = request.GET['movieName']
    flag = 0
    sum = 0
    count = 0
    for i in range(len(movie_list1)):
        if movie_list1.title[i] == movieName:
            index_movie = movie_list1.movieId[i]
            genre = movie_list1.genres[i]
            flag = 1
            break

    if flag == 0 and request.session.get('id'):
        return render(request, 'resultNotFoundWithLogin.html')

    elif flag == 0:
        return render(request, 'resultNotFound.html')
    elif flag == 1 and request.session.get('id'):
        for i in range(len(ratings1)):
            if ratings1.movieId[i] == index_movie:
                sum = sum + ratings1.rating[i]
                count = count + 1
        if sum == 0 or count == 0:
            avg_rating == 0
        else:
            avg_rating = sum / count

        return render(request, 'resultWithLogin.html',
                      {'movie_name': movieName, 'rate': round(avg_rating, 1), 'genre': genre, 'index': index_movie})
    else:
        for i in range(len(ratings1)):
            if ratings1.movieId[i] == index_movie:
                sum = sum + ratings1.rating[i]
                count = count + 1
        if sum == 0 or count == 0:
            avg_rating == 0
        else:
            avg_rating = sum / count

        return render(request, 'result.html',
                      {'movie_name': movieName, 'rate': round(avg_rating, 1), 'genre': genre, 'index': index_movie})


def genres(request):
    data = ['Adventure', 'Musical', 'Drama', 'Romance', 'Animation', 'Sci-Fi', 'Thriller', 'Crime', 'Fantasy',
            'Mystery', 'Action']
    if (request.session.get('id')):

        return render(request, 'genreWithLogin.html', {'data': data})
    else:
        return render(request, 'genreWithoutLogin.html', {'data': data})


def getMovieByData(data):
    temp = str(data.title)
    arr_data = []
    temp = temp.split('\n')
    for i in temp:
        temp2 = i.split()
        temp3 = ""
        for j in range(1, len(temp2)):
            temp3 = temp3 + ' ' + temp2[j]
        temp3.strip()
        temp3.lstrip()
        temp3.rstrip()
        arr_data.append(temp3)
    return arr_data


def getMoviesByGenre(request):
    genre = request.GET['genre']
    tempMovieNames = getMovieByData(best_movies_by_genre(genre, 10))
    movieNames = []

    for movies in tempMovieNames:
        movieNames.append(movies[1:])

    movieNames1 = []
    movies_index = []
    genres = []
    ratings = []
    for movie in movieNames:
        for i in range(len(movie_list1)):
            if movie_list1.title[i] == movie:
                movies_index.append(movie_list1.movieId[i])
                genres.append(movie_list1.genres[i])
                movieNames1.append(movie_list1.title[i])
                break

    for i in range(len(movies_index)):
        count = 0
        sum = 0
        for j in range(len(ratings1)):
            if ratings1.movieId[j] == movies_index[i]:
                sum = sum + ratings1.rating[j]
                count = count + 1

        if sum == 0 or count == 0:
            ratings.append(0)
        else:
            ratings.append(round(sum / count, 1))

    data = []
    for i in range(len(ratings)):
        Dict = {}
        Dict['rating'] = ratings[i]
        Dict['genre'] = genres[i]
        Dict['movie'] = movieNames[i]
        Dict['index'] = movies_index[i]
        data.append(Dict)

    if request.session.get('id'):
        return render(request, 'getMoviesByGenreWithLogin.html', {'data': data, 'genre': genre})
    else:
        return render(request, 'getMoviesByGenreWithoutLogin.html', {'data': data, 'genre': genre})


def topRated(request):
    tempMovieNames = getMovieByData(pd.DataFrame(movie_score.sort_values(['weighted_score'], ascending=False)[
                                                     ['title', 'count', 'mean', 'weighted_score', 'genres']][:10]))

    movieNames = []

    for movies in tempMovieNames:
        movieNames.append(movies[1:])

    movieNames1 = []
    movies_index = []
    genres = []
    ratings = []
    for movie in movieNames:
        for i in range(len(movie_list1)):
            if movie_list1.title[i] == movie:
                movies_index.append(movie_list1.movieId[i])
                genres.append(movie_list1.genres[i])
                movieNames1.append(movie_list1.title[i])
                break

    for i in range(len(movies_index)):
        count = 0
        sum = 0
        for j in range(len(ratings1)):
            if ratings1.movieId[j] == movies_index[i]:
                sum = sum + ratings1.rating[j]
                count = count + 1

        if sum == 0 or count == 0:
            ratings.append(0)
        else:
            ratings.append(round(sum / count, 1))

    data = []
    for i in range(len(ratings)):
        Dict = {}
        Dict['rating'] = ratings[i]
        Dict['genre'] = genres[i]
        Dict['movie'] = movieNames[i]
        Dict['index'] = movies_index[i]
        data.append(Dict)

    if (request.session.get('id')):
        return render(request, 'topRatedWithLogin.html', {'data': data})
    else:
        return render(request, 'topRatedWithoutLogin.html', {'data': data})


def removeFromWatchList(request):
    movieId = int(request.GET['movieId'])
    id = int(request.session.get('id'))

    movie = watchlist.objects.get(userId=id, movieId=movieId)
    movie.delete()
    return render(request, 'removedFromWatchList.html')
