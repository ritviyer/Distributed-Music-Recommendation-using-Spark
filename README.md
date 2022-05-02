# Distributed-Music-Recommendation-using-Spark

Steps to run code:

Pre-requisites:
1) Install Spark
2) Install Pyspark
3) Download all required dataset files from: <>

*Run locally*

Run MapReduce-Based CF by:
1) spark-submit map_reduce.py

Run ALS-Based CF by:
1) spark-submit finalALSModel.py

Run Friend-Based CF by:
1) python friendBasedCF.py

*Run on Cloud*
1. Sign up on AWS services and opt for EMR and S3 services.
2. Create a S3 bucket to store input/output/log data files.
3. Go to EMR and create a new cluster (You can choose any number of nodes in your cluster). Additional configs related to driver/executor memory, session timeout etc. can be added as a JSON config while creating the cluster. (This can be set manually even after the cluster is started using %configure signature)
4. Create a new Jupyter Notebook and link it to the newly created cluster.
5. Wait for the cluster to complete set up and then start/open the Jupyter Notebook.
6. Select a pySpark Kernel once it starts.
7. Finally, run the Jupyter Notebook.
