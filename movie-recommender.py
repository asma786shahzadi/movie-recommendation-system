import pandas as pd
import numpy as np 
# include the dataset csv files in the system
movies = pd.read_csv('d:/Internships/AI Projects/Project1/movie-recommendation-system/tmdb_5000_movies.csv')
credits = pd.read_csv('d:/Internships/AI Projects/Project1/movie-recommendation-system/tmdb_5000_credits.csv')
print(movies.head()) # it prints the entire movie header with entities
print(credits.head()) # it prints entire credit header with entities

#print(credits.head(1)['cast'].values) it shows the values of the cast
#print(credits.head(1)['crew'].values) it shows the values of crew both are important in our dataset and used further...abs
# merge datasets because it hacktic to used individual in our model training on the base of movie title

print(movies.merge(credits,on='title').shape) # shape shows the merged entities quantity of rows and column
print(movies.head().shape)
print(credits.head().shape)  # difference is bcz of merge on the base of title so, it is included twice
movies=movies.merge(credits,on='title')
print(movies.head()) # it shows the dataset of merged entities with credit in movies csv file and also included the title onceS in it


