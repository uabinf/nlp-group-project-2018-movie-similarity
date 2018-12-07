import pandas
from imdb import IMDb

df = pandas.read_csv('movie_dataset.csv')
ia = IMDb()
count_row = df.shape[0]


for x in range(153, 154):
	movie_title = df['Title'][x]
	movie_id = df['imdbID'][x][2:]
	curr_movie = ia.get_movie(movie_id)
	fo = open(movie_title, "w")
	fo.write(curr_movie['synopsis'][0])
	print("Working on:" + df['Title'][x])
	fo.close()

print("....Completed.......")