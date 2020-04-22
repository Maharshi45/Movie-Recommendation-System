from django.db import models


class user(models.Model):
    firstName = models.CharField(max_length=20)
    lastName = models.CharField(max_length=20)
    email = models.CharField(max_length=50)
    password = models.CharField(max_length=20)
    status = models.BooleanField()


class watchlist(models.Model):
    userId = models.IntegerField()
    movieId = models.IntegerField()


class requestMovieModel(models.Model):
    userId = models.IntegerField()
    movieName = models.CharField(max_length=50)
