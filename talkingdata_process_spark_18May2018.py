
# Author: David Modjeska
# Project: Kaggle Competition - TalkingData AdTracking Fraud Detection Challenge
# File: process the raw data

# In[24]:


from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator

file_suffix = ''



# In[29]:


#----- Get data from Amazon S3

import os, zipfile
import boto
import boto.s3.connection

def s3_get_file(bucket_name, filename):
    conn = boto.connect_s3(
            host = 's3.amazonaws.com',
            is_secure = False,
            calling_format = boto.s3.connection.OrdinaryCallingFormat(),
            )

    bucket = conn.get_bucket(bucket_name)
    key = bucket.get_key(filename)
    key.get_contents_to_filename(filename)

    home_dir = os.path.expanduser("~")
    os.chdir(home_dir)
    my_zipfile = zipfile.ZipFile(filename)
    my_zipfile.extractall()

bucket_name = 'talking-data-raw-dkm'

s3_get_file(bucket_name, 'train_sample.parquet.zip')
s3_get_file(bucket_name, 'test_sample.parquet.zip')
s3_get_file(bucket_name, 'train.parquet.zip')
s3_get_file(bucket_name, 'test.parquet.zip')

# In[30]:


## create Spark session, with Spark context and SQL context

spark = SparkSession.builder \
    .config('spark.executor.instances', "30") \
    .config('spark.executor.memoryOverhead', "3072") \
    .config('spark.executor.memory', "21G") \
    .config('spark.driver.memoryOverhead', "1034") \
    .config('spark.driver.memory', "6G") \
    .config('spark.executor.cores', "3") \
    .config('spark.driver.cores', "1") \
    .config('spark.default.parallelism', "180") \
    .config('spark.dynamicAllocation.enabled', "false") \
    .getOrCreate() 
   
sc = spark.sparkContext    
sqlContext = SQLContext(sc)

sc.setLogLevel('ERROR')

# In[31]


# Helper function to load and display data
def load_and_display_data(in_file_prefix):

    # read data file
    data = sqlContext.read.parquet(in_file_prefix + '.parquet')
        
    print()
    print('RAW DATA')
    
    print()
    print('Shape of data:', data.count(), ',', len(data.columns))

    print()
    print('Head of data:')
    print()
    data.show(5)

    print()
    print('Schema:')
    print()
    data.printSchema()
    
    # drop 'attributed_time' column from training data because not used
    if in_file_prefix != 'test':
        data = data.drop('attributed_time')
    
    return(data)


# helper function to clean data
def clean_data(data):

    # check for duplicate records
    data = data.dropDuplicates() # FIX?

    print()
    print('Shape of deduplicated data:', data.count(), ',', len(data.columns))

    # check for null values
    print()
    print('Columns containing nulls:')
    data.select(*(sum(col(c).isNull().cast("int")).alias(c) for c in data.columns)) \
        .show()
    
    return(data)


# helper function to prep date and time
def prep_dates_and_times(data):
    
    # get first and last time stamps
    first_timestamp = data.agg(min('click_time'))
    last_timestamp = data.agg(max('click_time'))    
    first_timestamp_string = first_timestamp         .first()[0]         .strftime("%Y-%m-%d %H:%M:%S")  
    
    # show the date limits
    print()
    first_timestamp.show()
    last_timestamp.show()
      
    # create date and time features
    data = data \
        .withColumn("time_index", datediff(col('click_time'), lit(first_timestamp_string)).cast('Int')) \
        .withColumn("second_of_minute", second(col('click_time')).cast('Int')) \
        .withColumn("minute_of_hour", minute(col('click_time')).cast('Int'))   \
        .withColumn("hour_of_day", hour(col('click_time')).cast('Int'))         \
        .withColumn("day_of_month", dayofmonth(col('click_time')).cast('Int'))         \
        .drop('click_time')

    print()
    print('Head of data with date and time features:')
    data.show(5)
    
    return(data)


# helper function for multi-column click analysis
def prep_click_analysis(data):
    
    # group data and count clicks
    click_count_df = data \
        .groupby('ip', 'device', 'app') \
        .count() \
        .withColumn('count', log('count'))

    # rename columns for join
    click_count_df = click_count_df \
        .withColumnRenamed("count", "click_count")  \
        .withColumnRenamed("ip", "ip_click") \
        .withColumnRenamed("device", "device_click") \
        .withColumnRenamed("app", "app_click") 

    # join data with click counts
    data = data.join(click_count_df, 
                       (data.ip == click_count_df.ip_click) & 
                       (data.device == click_count_df.device_click) &
                       (data.app == click_count_df.app_click), 
                       how = 'left') \
                       .drop('ip_click', 'device_click', 'app_click')

    return(data)

# helper function for single-column click analysis
def prep_click_analysis2(data, col_name):
    
    # group data and count clicks
    click_count_df = data \
        .groupby(col_name) \
        .count() 

    # rename columns for join
    join_col_name = col_name + "_click"
    click_count_df = click_count_df \
        .withColumnRenamed("count", col_name + "_click_count")  \
        .withColumnRenamed(col_name, join_col_name) 

    # join data with click counts
    data = data.join(click_count_df, 
                     data[col_name]  == click_count_df[join_col_name], how = 'left') \
        .drop(join_col_name)

    return(data)

# helper function for all click analysis
def prep_all_click_analysis(data):
    data = prep_click_analysis(data)

    data = prep_click_analysis2(data, 'ip')
#    data = prep_click_analysis2(data, 'channel')
#    data = prep_click_analysis2(data, 'os')
#    data = prep_click_analysis2(data, 'app')

    print()
    print('Shape of data including click analysis:', 
          data.count(), 
          ',', 
          len(data.columns))
    print()
    data.show(5)

    print()
    print('DATA MERGED WITH CLICK ANALYSIS')
    print()
    print('Shape of data:', data.count(), ',', len(data.columns))
    print('Head of data:')
    data.show(5)
    print()
    print('Schema:')
    data.printSchema()
    
    return(data)

# helper function to label encode one column
def label_encode_one_col(data, col_name):
    new_col_name = col_name + '_encoded'
    print("Column: ", col_name)
    
    indexer = StringIndexer(inputCol = col_name, outputCol = new_col_name, handleInvalid = 'error')   

    new_data = indexer  \
       .fit(data) \
        .transform(data) \
        .drop(col_name)
        
    return(new_data)


# helper function to label encode all columns 
def label_encode_cols(data):  
    print()
    print('INDEXING STRINGS')
    print()
    new_data = label_encode_one_col(data, 'app')
    new_data = label_encode_one_col(new_data, 'device')
    new_data = label_encode_one_col(new_data, 'os')
    new_data = label_encode_one_col(new_data, 'channel')

    print()
    print('WITH LABEL-ENCODED COLUMNS')
    print()    
    print('Head of data')
    data.show(5)

    return(new_data)


# helper function to one-hot encode all columns
def one_hot_encode_cols(data):
    encoder = OneHotEncoderEstimator()         .setInputCols(["app_encoded", "device_encoded", "os_encoded", "channel_encoded"])         .setOutputCols(["app_vector", "device_vector", "os_vector", "channel_vector"])
        
    data = encoder \
        .fit(data) \
        .transform(data) \
        .drop('app_encoded', 'device_encoded', 'os_encoded', 'channel_encoded')
    
    print()
    print('WITH ONE-HOT-ENCODED COLUMNS')
    print()    
    print('Head of data')
    data.show(5)
    
    return(data)


# helper function to start processing of one raw data file
def process_one_data_file(in_file_prefix):
    data = load_and_display_data(in_file_prefix)
    data = clean_data(data)
    data = prep_dates_and_times(data)
    data = prep_all_click_analysis(data)

    return(data)


# helper function to save one processed data file
def save_one_processed_data_file(data, out_file_prefix):
    data.write \
        .format("parquet") \
        .mode('overwrite') \
        .option("header", "true") \
        .save('processed_' + out_file_prefix + '.parquet')


# In[40]:


## process all raw data files
        
train_prefix = 'train' + file_suffix
test_prefix = 'test' + file_suffix

print('=========================================================')
print('TRAIN DATA')
print()
train_data = process_one_data_file(train_prefix)

print('=========================================================')
print('TEST DATA')
print()
test_data = process_one_data_file(test_prefix)


# In[42]:


select_list = ['ip', 'app', 'device', 'os', 'channel', 'time_index', 'second_of_minute', 
    'minute_of_hour', 'hour_of_day', 'day_of_month', 'click_count', 'source', 
    'is_attributed', 'click_id', 'ip_click_count']

train_data = train_data \
    .withColumn("source", lit('train')) \
    .withColumn("click_id", lit(0)) \
    .select(*select_list)
    
test_data = test_data \
    .withColumn("source", lit('test')) \
    .withColumn("is_attributed", lit(0)) \
    .select(*select_list)    


# In[43]:


all_data = train_data.union(test_data)

all_data = label_encode_cols(all_data)    
all_data = one_hot_encode_cols(all_data)

train_data = all_data \
    .filter("source = 'train'") \
    .drop('source', 'click_id')
    
test_data = all_data \
    .filter("source = 'test'") \
    .drop('source', 'is_attributed')



# save all processed data files
save_one_processed_data_file(train_data, train_prefix)
save_one_processed_data_file(test_data, test_prefix)
print('Saved processed data files to disk')


# In[46]:


 ##  create 10% training parquet file

#data = sqlContext.read.parquet('test.parquet')
#
#use_data, discard_data = data.randomSplit([.10,.90], seed = 1234)
#
#use_data.write \
#     .format("parquet") \
#     .mode('overwrite') \
#     .option("header", "true") \
#     .save('test_10_percent.parquet')

