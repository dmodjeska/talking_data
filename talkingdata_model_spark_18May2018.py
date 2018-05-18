
# Author: David Modjeska
# Project: Kaggle Competition - TalkingData AdTracking Fraud Detection Challenge
# File: model and predict using processed data

# In[7]:


import numpy as np

from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import when, sum
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler

file_suffix = ''


# In[11]:


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

bucket_name = 'talking-data-proc-dkm'

s3_get_file(bucket_name, 'processed_train.parquet.zip')
s3_get_file(bucket_name, 'processed_test.parquet.zip')


# In[12]:


#----- PREP TRAINING DATA

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

# helper function to load and show data
def load_and_show_data(filename):
    data = sqlContext.read.parquet(filename)
    num_rows = data.count()

    print()
    print('Shape of data:')
    print()
    print(num_rows, len(data.columns))

    print()
    print('Head of data:')
    print()
    data.show(5)

    print()
    print('Data schema:')
    print()
    data.printSchema()
    
    return(data, num_rows)


# In[14]:


in_filename = 'processed_train' + file_suffix + '.parquet'
ads_df, num_rows = load_and_show_data(in_filename)


## helper function to create label column and vectorized features column
def create_modeling_cols(data):
    numericCols = ['ip', 'time_index', 'second_of_minute', 'minute_of_hour', 'hour_of_day', 
        'day_of_month', 'click_count', 'ip_click_count']
    categoricalCols = ['app_vector', 'device_vector', 'os_vector', 'channel_vector']

    assemblerInputs = numericCols + categoricalCols
    assembler = VectorAssembler(inputCols = assemblerInputs, outputCol = "features")
    data_vector = assembler \
        .transform(data) \
        .drop(*numericCols) \
        .drop(*categoricalCols)
        
    return(data_vector)
    
ads_df_vector = create_modeling_cols(ads_df)



# In[16]:


## create weights column

num_attributions = ads_df_vector.agg(sum("is_attributed")).first()[0]
my_scale_pos_weight = (num_rows - num_attributions) / num_attributions
print('my_scale_pos_weight:', my_scale_pos_weight)

ads_df_vector = ads_df_vector \
    .withColumn("attrib_weights", when(ads_df_vector["is_attributed"] == 1, my_scale_pos_weight).otherwise(1)) \
    .withColumnRenamed("is_attributed", "label")

# divide training data into training and validation sets
train_data, validate_data = ads_df_vector.randomSplit([.66,.34], seed = 1234)



# In[32]:


# HELPER FUNCTIONS

# adapted from http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
def display_roc_curve(y, predict_y_proba):
   
    fp_rate1, tp_rate1, _ = roc_curve(y, predict_y_proba[:, 1])
    roc_auc1 = auc(fp_rate1, tp_rate1)
    print()
    print('\tROC AUC score:', roc_auc1)


def describe_predictions(probabilities):
    print()
    print('PREDICTIONS SUMMARY')
    print()
    print('\tMin:', np.min(predictions_ar[:, 1]))
    print('\tMax:', np.max(predictions_ar[:, 1]))
    print('\tMean:', np.mean(predictions_ar[:, 1]))
    


# In[146]:


#----- LOGISTIC REGRESSION

print()
print()
print('LOGISTIC REGRESSION')

log_reg = LogisticRegression(featuresCol = 'features', labelCol = 'label', weightCol = 'attrib_weights',
                             maxIter = 10, regParam = 0.00, elasticNetParam = 0.0, standardization = True)
logModel = log_reg.fit(train_data)

# make predictions
predicted = logModel.transform(validate_data) 
evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
print('\tROC AUC score = ', evaluator.evaluate(predicted))



# In[ ]:


#----- RANDOM FOREST
#
#print()
#print()
#print('RANDOM FOREST')
#
#rfc = RandomForestClassifier(featuresCol = 'features', labelCol = 'label', numTrees = 200, maxDepth = 20)
#rfcModel = rfc.fit(train_data)
#
## make predictions
#predicted = rfcModel.transform(validate_data)
#evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
#print('\tROC AUC score = ', evaluator.evaluate(predicted))



# In[149]:


#----- PREP TEST DATA

in_filename = 'processed_test' + file_suffix + '.parquet'
test_df, num_rows = load_and_show_data(in_filename)

test_df_vector = create_modeling_cols(test_df)
test_predicted = logModel.transform(test_df_vector)


print()
print('Shape of data:')
print()
print(test_predicted.count(), len(test_predicted.columns))

print()
print('Head of data:')
print()
test_predicted.show(5)

print()
print('Data schema:')
print()
test_predicted.printSchema()


# In[152]:


#----- PREDICT WITH TEST DATA

predictions = test_predicted.select("probability").rdd.map(lambda x: x[0]).collect()
is_attributed = np.array([x[1] for x in predictions]).reshape(-1, 1)

ids = test_predicted.select("click_id").rdd.map(lambda x: x[0]).collect()
click_id = np.array(ids).reshape(-1, 1)

predictions_ar = np.concatenate([click_id, is_attributed], axis = 1)
describe_predictions(predictions_ar)

np.savetxt(fname = 'submission_data_ar.csv', X = predictions_ar, delimiter = ',',
           fmt = ['%d', '%0.10f'])
print()
print('Saved submission data to disk')

