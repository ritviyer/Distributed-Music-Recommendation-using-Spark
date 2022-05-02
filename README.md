# Distributed-Music-Recommendation-using-Spark

Steps to run code:

Pre-requisites:
1) Install Spark
2) Install Pyspark
3) Install the necessary missing libraries like findspark 
4) Download all required dataset files from: <http://millionsongdataset.com/>

*Run locally*

Run MapReduce-Based CF by:
1) spark-submit map_reduce.py

Run ALS-Based CF by:
1) spark-submit finalALSModel.py

Run Friend-Based CF by:
1) python friendBasedCF.py

You can use additional options while submitting the spark job such as --master local[8] where 8 can be replaced with the number of cores you want to use.

*Run on Cloud*
1. Sign up on AWS services and opt for EMR and S3 services.
2. Create a S3 bucket to store input/output/log data files.
3. Go to EMR and create a new cluster (You can choose any number of nodes in your cluster). Additional configs related to driver/executor memory, session timeout etc. can be added as a JSON config while creating the cluster. (This can be set manually even after the cluster is started using %configure signature)
4. Create a new Jupyter Notebook and link it to the newly created cluster.
5. Wait for the cluster to complete set up and then start/open the Jupyter Notebook.
6. Select a pySpark Kernel once it starts.
7. Finally, run the Jupyter Notebook.

### cleanlasFMdata.py
Cleaning the data and converting the tab separated dat file into csv files.

### combineMetadataTripletDataset.py
Merging the triplets dataset which consists of the userID, songID and listen count, with metadata such as song name, artist name, etc.

### spark-ALS.py
Uses cross validator and grid search to run the ALS model multiple times with different parameters to find the optimal hyperparameters.

### finalALSModel.py
"Alternating least squares (ALS)" is a distributed matrix factorization method that allows for faster and more efficient computations. The algorithm predicts number of listens using Matrix Factorization. It is an iterative algorithm that alternates back and forth between user and song vectors for solving and providing the recommendations.

Uses the best hyperparameters obtained for ALS to run the model on the entire dataset and make predictions for users.

### friendBasedCF.py
Friend-Based CF is based on the assumption that an individuals taste/liking is strongly influence by the people around him. It is more likely that an individuals taste in music is more similar to his friend rather than a stranger in a different country. Hence, if we can define these relations and form smaller cluster then it is possible to use CF to compare an individual only to his friends and connections in order to determine similarity. Hence, a friend based collaborative filtering would be more efficient, accurate and scalable.

Creates a cluster of 2nd degree friends and calculates the cosine similarity between users of that cluster and makes recommendation. It calculates the cosine similarity of all users as well so that the results can be compared.

### map_reduce.py
This approach uses item-item collaborative filtering to provide the recommendations for a user. We started with the data set consisting of user, song and rating information. We calculated all the pairs of songs listened by the users and the corresponding ratings of songs. This is done for all the users and all the songs they have listened to. Once we have this list consisting of song pairs and rating pairs, for each song pair we form a vector of ratings pairs collected by a number of users. Next the cosine similarity algorithm is applied on this vector to find the similarity score of the song pair. While providing the recommendation for a user, we consider the user’s top songs and recommend other songs which are similar to his listening history.

### normalizeData.py
Create a normalized version of the listen count to check how it affects the performance of the model.

### tripletsToCSV.py
Converts the triplets file in text format into a csv format for the model to access.


Please refer to the report for additional information and experimental results.
