import pymongo
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark import SparkContext

'''
specify collection name, here:
    ratings_temp: original ml-20m ratings
    ratings_sort: sort ratings_temp by userId (ascend) and timestamp (ascend)
'''
def read_mongo(collection):
    client = pymongo.MongoClient("39.98.136.173", 9099)
    client.movie.authenticate('user','cloud',mechanism='SCRAM-SHA-1')
    database = client['movie']
    ratings = database[collection]
    cursor = ratings.find()
    df = pd.DataFrame(list(cursor))
    df = df[['userId', 'movieId', 'rating', 'timestamp']]
    spark = SparkSession.builder.appName('readMongo').getOrCreate()
    sqlContest = SQLContext(spark)
    spark_df = sqlContest.createDataFrame(df)
    return spark_df