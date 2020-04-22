import pandas as pd

ratings = pd.read_csv('../Dataset/ratings.csv')
movie_list = pd.read_csv('../Dataset/movies.csv')
tags = pd.read_csv('../Dataset/tags.csv')

movie_list1 = movie_list[['movieId', 'genres', 'title']]
ratings1 = ratings[['userId', 'movieId', 'rating']]

print(movie_list1.title[12])

arr = set()

for i in range(len(ratings1)):
    if ratings1.userId[i] <= 100:
        arr.add(ratings1.movieId[i])

print('Arr', arr)

movie_index = []

for i in range(len(movie_list1)):
    if movie_list1.movieId[i] not in arr:
        movie_index.append(i)

print('Movie index', movie_index)

# print(max(movie_index))

# for i in range(len(movie_index) - 1, 1, -1):
#     if (movie_index[i + 1] != movie_index[i]):
#         break_index = movie_index[i]
#         break
#
# print(break_index)

movie_list1.drop(movie_index)

print(movie_list1.title[12])

print("Deleted")
