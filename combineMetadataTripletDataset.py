#script to merge the triplet and metadata file
import pandas

triplets_file = r"train_triplets\train_triplets.txt"
songs_metadata_file = r"SongCSV.csv"

song_df_1 = pandas.read_table(triplets_file,header=None)
song_df_1.columns = ['user_id', 'song_id', 'listen_count']
#print(song_df_1[1:10])

song_df_2 =  pandas.read_csv(songs_metadata_file)
song_df_2.rename(columns = {'SongID':'song_id'}, inplace = True)

song_df = pandas.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left")

song_df.to_csv('out.csv')