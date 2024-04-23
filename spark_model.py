import sys
import json
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, when
from pyspark.sql.types import FloatType, StringType, StructField, StructType, IntegerType
from pyspark.ml import PipelineModel
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression

# Accept JSON input from the command line
if __name__ == '__main__':
    input_json = sys.argv[1]
    input_data = json.loads(input_json)

# Initialize Spark Session
conf = SparkConf().setMaster("local[*]").setAppName("PredictionApp")
sc = SparkContext.getOrCreate(conf=conf)
spark = SparkSession(sc)
spark.sparkContext.setLogLevel("ERROR")

# Schema definition for incoming data
schema = StructType([
    StructField("text", StringType(), True),
    StructField("total_hours_week", FloatType(), True),
    StructField("is_weekend_open", IntegerType(), True),
    StructField("state", StringType(), True),
    StructField("categories", StringType(), True),
])

# Assuming input_data is already defined; for example:

# Create a DataFrame using the input data
df = spark.createDataFrame([input_data], schema=schema)

# Load your trained model

model = PipelineModel.load("hdfs://compute-14:9000/user/kokai1/model2")

# Make predictions
prediction = model.transform(df)

# Convert prediction result to JSON
prediction_result = prediction.select("prediction", "probability").collect()[0]
result_dict = {
    "prediction": prediction_result['prediction'],
    "probability": str(prediction_result['probability'])
}

# Print the output as JSON
print(json.dumps(result_dict))

