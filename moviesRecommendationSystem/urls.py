"""moviesRecommendationSystem URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
  path('admin/', admin.site.urls),
  path('home', views.home, name = 'home'),
  path('', views.home, name = 'home'),
  path('registered', views.registered, name = 'registered'),
  path('profile', views.profile, name = 'profile'),
  path('logout', views.logout, name = 'logout'),
  path('doLogin', views.doLogin, name = 'doLogin'),
  path('login', views.login, name = 'login'),
  path('register', views.register, name = 'register'),
  path('doRegister', views.doRegister, name = 'registered'),
  path('getMovie', views.getMovie, name = 'getMovie'),
  path('addToWatchList', views.addToWatchList, name = 'addToWatchList'),
  path('getWatchList', views.getWatchList, name = 'getWatchList'),
  path('requestMovie', views.requestMovie, name = 'getWatchList'),
  path('doRequestMovie', views.doRequestMovie, name = 'doRequestMovie'),
  path('search', views.search, name = 'search'),
  path('genres', views.genres, name = 'genres'),
  path('getMoviesByGenre', views.getMoviesByGenre, name = 'getMoviesByGenre'),
  path('topRated', views.topRated, name = 'topRated'),
  path('removeFromWatchList', views.removeFromWatchList, name = 'removeFromWatchList'),
]
