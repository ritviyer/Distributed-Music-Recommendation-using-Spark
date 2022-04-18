#script to merge the triplet and metadata file
import pandas


dataPath = '../dataset/Million Song/'
triplets_file = dataPath + "triplets.csv"
songs_metadata_file = dataPath + "unique_tracks.csv"

song_df_1 = pandas.read_csv(triplets_file)
song_df_2 =  pandas.read_csv(songs_metadata_file)

song_df = pandas.merge(song_df_1, song_df_2.drop_duplicates(['SongID']), on="SongID", how="inner")

song_df.to_csv(dataPath + 'songCSV.csv', index = False)