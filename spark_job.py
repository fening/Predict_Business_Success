from pyspark import SparkConf, SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import col, udf, when
from pyspark.sql.types import FloatType
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline,PipelineModel
from functools import reduce
from operator import add
from pyspark.ml.feature import StringIndexer, OneHotEncoder
import json
import sys
import json
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, when
from pyspark.sql.types import FloatType, StringType, StructField, StructType, IntegerType
from pyspark.ml import PipelineModel
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression

if __name__ == '__main__':
    input_json = sys.argv[1]
    input_data = json.loads(input_json)

# Initialize Spark Session
conf = SparkConf().setMaster("local[*]").set("spark.executer.memory", "2g")
sc = SparkContext(conf=conf)
spark = SparkSession.builder \
    .appName("AppName") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()
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

# Business hours
# Define the UDF to parse hours from strings
def parse_hours(hours_str):
    try:
        if hours_str is None or '-' not in hours_str:
            return 0
        open_time, close_time = hours_str.split('-')
        open_hour, open_minute = map(int, open_time.split(':'))
        close_hour, close_minute = map(int, close_time.split(':'))
        open_minutes = open_hour * 60 + open_minute
        close_minutes = close_hour * 60 + close_minute
        if close_minutes < open_minutes:
            close_minutes += 1440  # handle closing times past midnight
        return (close_minutes - open_minutes) / 60
    except ValueError:
        return 0  # Return 0 hours if there is any error in parsing the hours

# Register the UDF
parse_hours_udf = udf(parse_hours, FloatType())

# Load the business hours dataset
yelp_business_hours_df = spark.read.format("csv").option("header", "true").option("multiline","true").load("yelp_business_hours.csv")
# Apply the UDF to calculate total hours for each day
days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
for day in days:
    yelp_business_hours_df = yelp_business_hours_df.withColumn(f'{day}_hours', parse_hours_udf(col(day)))

# Calculate total hours open per week
total_hours_cols = [col(f'{day}_hours') for day in days]
yelp_business_hours_df = yelp_business_hours_df.withColumn('total_hours_week', reduce(add, total_hours_cols))

# Determine if open on weekends
yelp_business_hours_df = yelp_business_hours_df.withColumn(
    'is_weekend_open',
    when((col('saturday_hours') > 0) | (col('sunday_hours') > 0), 1).otherwise(0)
)

# Select only the necessary columns
yelp_business_hours_df = yelp_business_hours_df.select('business_id', 'total_hours_week', 'is_weekend_open')



# Business details
yelp_business_df = spark.read.format("csv").option("header", "true").option("multiline","true").load("yelp_business.csv")
yelp_business_df = yelp_business_df.select("business_id", "state", "stars", "review_count", "is_open", "categories").withColumnRenamed('stars', 'review_stars')



# reviews dataset
df_reviews = spark.read.format("csv").option("header", "true").option("multiline","true").load("yelp_review.csv").select('business_id', 'text', 'stars').withColumn("label", col("stars").cast("double"))

df_joined = df_reviews.join(yelp_business_hours_df, 'business_id', 'inner')
df_joined = df_joined.join(yelp_business_df, on='business_id', how='inner')

df_joined = df_joined.na.drop()



df_joined =  df_joined.filter(df_joined.label.isin(1.0, 2.0, 3.0, 4.0, 5.0)).limit(10000)

# Handle Nulls
df_joined = df_joined.na.fill({"state": "unknown", "categories": "unknown"})

# Pipeline components
stateIndexer = StringIndexer(inputCol="state", outputCol="stateIndex", handleInvalid="keep")
categoryIndexer = StringIndexer(inputCol="categories", outputCol="categoryIndex", handleInvalid="keep")
stateEncoder = OneHotEncoder(inputCol="stateIndex", outputCol="stateVec")
categoryEncoder = OneHotEncoder(inputCol="categoryIndex", outputCol="categoryVec")
tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol="words", outputCol="tfFeatures")
idf = IDF(inputCol="tfFeatures", outputCol="features")
assembler = VectorAssembler(
    inputCols=["features", "total_hours_week", "is_weekend_open", "stateVec", "categoryVec"],
    outputCol="final_features",
    handleInvalid="keep"
)
lr = LogisticRegression(maxIter=10, regParam=0.001, featuresCol="final_features", labelCol="label")

# Define and fit the pipeline
pipeline = Pipeline(stages=[
    stateIndexer, categoryIndexer, stateEncoder, categoryEncoder,
    tokenizer, hashingTF, idf, assembler, lr
])
model = pipeline.fit(df_joined)
model.write().overwrite().save("hdfs://compute-14:9000/user/kokai1/model2")

# Prepare and predict
prediction_df = spark.createDataFrame([input_data], schema=schema)
prediction = model.transform(prediction_df)

# Display results
selected = prediction.select("state", "probability", "prediction")
for row in selected.collect():
    print(f"({row.state}) --> prediction = {row.prediction}, probability = {row.probability}")

# Convert prediction result to JSON and print
prediction_result = prediction.select("prediction", "probability").collect()[0]
result_dict = {
    "prediction": prediction_result['prediction'],
    "probability": str(prediction_result['probability'])
}
print(json.dumps(result_dict))

# Clean up resources
spark.stop()