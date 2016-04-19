#Goal: given the time and location
#you must predict the category of the crime
#Start using python3: PYSPARK_PYTHON=python3 /usr/bin/pyspark --driver-memory 2g

#make sure hadoop is started: hdfs dfs -l
#Take a sample of the training csv, and name it train2.csv - otherwise its too big
#put the train2.csv into the HDFS main directory
import pyspark.mllib.regression as mllib_reg
import pyspark.mllib.linalg as mllib_lalg
import pyspark.mllib.classification as mllib_class
import pyspark.mllib.tree as mllib_tree
from pyspark.ml.feature import StringIndexer
import numpy as np
from pyspark.sql import SQLContext
import pandas as pd
import csv
import io
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import col
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.mllib.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SQLContext
sqlContext=SQLContext(sc)

#create the framework for the training set dataframe
def loadRecord(line):
    input = io.StringIO(line)
    reader = csv.DictReader(input, fieldnames=['Dates', 'Category', 'Descript', 'DayOfWeek', 'PdDistrict', 'Resolution', 'Address', 'X', 'Y'])
    return next(reader)

#load the training set
input_file = sc.textFile('train.csv').map(loadRecord)

#remove the 1st row b/c it has a duplicate data header
header = input_file.first()
rows = input_file.filter(lambda line: line != header)
#rows.take(5) #test to see if the data is in


#create dataframe

data_df = rows.toDF()

#create pandas dataframe
pandas_df = data_df.toPandas()

#clean up the data
#make a new column to track hour


pandas_df['Dates'] = pd.to_datetime(pandas_df['Dates'])
pandas_df['day'] = pandas_df['Dates'].dt.day
pandas_df['month'] = pandas_df['Dates'].dt.month
pandas_df['year'] = pandas_df['Dates'].dt.year
pandas_df['hour'] = pandas_df['Dates'].dt.hour
pandas_df['dayofweek'] = pandas_df['Dates'].dt.dayofweek
pandas_df['week'] = pandas_df['Dates'].dt.weekofyear
pandas_df['x_sim'] = pandas_df['X'].str[1:8]
pandas_df['X'] = pandas_df['X'].str[1:8]
pandas_df['y_sim'] = pandas_df['Y'].str[0:6]
pandas_df['X'] = pd.to_numeric(pandas_df['X'])
pandas_df['Y'] = pd.to_numeric(pandas_df['Y'])
pandas_df['x_sim'] = pd.to_numeric(pandas_df['x_sim'])
pandas_df['y_sim'] = pd.to_numeric(pandas_df['y_sim'])

#send back to the RDD
data_df = sqlContext.createDataFrame(pandas_df)

#encode the police dept as a feature

stringIndexer = StringIndexer(inputCol="PdDistrict", outputCol="PdDistrict_Index")
model = stringIndexer.fit(data_df)
indexed = model.transform(data_df)
encoder = OneHotEncoder(dropLast=False, inputCol="PdDistrict_Index", outputCol="pd")
encoded = encoder.transform(indexed)

#remove data_df from memory
data_df.unpersist() 

#encode the dependent variable - category_predict
classifyIndexer = StringIndexer(inputCol="Category", outputCol="Category_Index")
classifymodel = classifyIndexer.fit(encoded)
encoded2 = classifymodel.transform(encoded)



#keep the following columns: x, y, hour, day, month, year, dayofweek, week, x_sim, y_sim
#drop the following
cleaned = encoded2.select([c for c in encoded2.columns if c not in{'DayOfWeek','Category','Address','Dates','Descript','PdDistrict','Resolution','PdDistrict_Index'}])

ignore = ['Category_Index']
assembler = VectorAssembler(inputCols=[x for x in cleaned.columns if x not in ignore],outputCol='features')

transformed = assembler.transform(cleaned)


data_transformed = transformed.select(col("Category_Index").alias("label"), col("features")).map(lambda row: LabeledPoint(row.label, row.features))

#********************************************************************************
# split the training set
train, test = data_transformed.randomSplit([0.7, 0.3], seed = 2)

#naivebayes classifier
#lambda = 1.0
# initialize classifier:
nb_model = mllib_class.NaiveBayes.train(train, 1.0)
#this step will take 50 seconds

# Make prediction and test accuracy.
# Evaluating the model on training data
labelsAndPreds = test.map(lambda p: (p.label, nb_model.predict(p.features)))
trainErr = labelsAndPreds.filter(lambda vp: vp[0] != vp[1]).count() / float(train.count())
print("Training Error = " + str(trainErr))
#this comes out to be .339

#save model
nb_model.save(sc, "nb_model")


# Evaluate model on test instances and compute test error
predictions = model.predict(test.map(lambda x: x.features))
labelsAndPredictions = test.map(lambda lp: lp.label).zip(predictions)
testMSE = labelsAndPredictions.map(lambda vp: (vp[0] - vp[1]) * (vp[0] - vp[1])).sum() / float(test.count())
print('Test Mean Squared Error = ' + str(testMSE))
#test MSE = 54.207

#*********************************************************************
#load the test set
#create the framework for the training set dataframe
def loadTestRecord(line):
    input = io.StringIO(line)
    reader = csv.DictReader(input, fieldnames=['Id','Dates','DayOfWeek','PdDistrict','Address','X','Y'])
    return next(reader)

#load the training set
test_input_file = sc.textFile('test.csv').map(loadTestRecord)

#remove the 1st row b/c it has a duplicate data header
test_header = test_input_file.first()
test_rows = test_input_file.filter(lambda line: line != test_header)
test_rows.take(5) #test to see if the data is in


#create dataframe
test_data_df = test_rows.toDF()

#create pandas dataframe
test_pandas_df = test_data_df.toPandas()

#clean up the data
#make a new column to track hour

test_pandas_df['Dates'] = pd.to_datetime(test_pandas_df['Dates'])
test_pandas_df['day'] = test_pandas_df['Dates'].dt.day
test_pandas_df['month'] = test_pandas_df['Dates'].dt.month
test_pandas_df['year'] = test_pandas_df['Dates'].dt.year
test_pandas_df['hour'] = test_pandas_df['Dates'].dt.hour
test_pandas_df['dayofweek'] = test_pandas_df['Dates'].dt.dayofweek
test_pandas_df['week'] = test_pandas_df['Dates'].dt.weekofyear

#truncate the X and Y for similiar locations
#1st decimal is an area of 11.1km
#2nd decimal is an area of 1.1km
#3rd decimal is 110 meters - USING three decimal places
#4th is 11 meters
test_pandas_df['x_sim'] = test_pandas_df['X'].str[1:8] #non-negative data
test_pandas_df['X'] = test_pandas_df['X'].str[1:8] #non-negative
test_pandas_df['y_sim'] = test_pandas_df['Y'].str[0:6]


test_pandas_df['X'] = pd.to_numeric(test_pandas_df['X'])
test_pandas_df['Y'] = pd.to_numeric(test_pandas_df['Y'])
test_pandas_df['x_sim'] = pd.to_numeric(test_pandas_df['x_sim'])
test_pandas_df['y_sim'] = pd.to_numeric(test_pandas_df['y_sim'])

#send back to the RDD
#
test_data_df = sqlContext.createDataFrame(test_pandas_df)

#encode the police dept as a feature
#stringIndexer = StringIndexer(inputCol="PdDistrict", outputCol="PdDistrict_Index")
model = stringIndexer.fit(test_data_df)
indexed = model.transform(test_data_df)
encoder = OneHotEncoder(dropLast=False, inputCol="PdDistrict_Index", outputCol="pd")
encoded = encoder.transform(indexed)


#encode the dependent variable - category_predict
#test_classifyIndexer = StringIndexer(inputCol="category_predict", outputCol="category")
#classifymodel = classifyIndexer.fit(encoded)
#encoded2 = classifymodel.transform(encoded)

#keep the following columns: x, y, hour, day, month, year, dayofweek, week, x_sim, y_sim
#drop the following

test_cleaned = encoded.select([c for c in encoded.columns if c not in{' day_of_week','Address','Dates','DayOfWeek','PdDistrict','category_predict','address','date','description_ignore','pd_district','resolution','pd_district_Index'}])

test_ignore = ['Address','Dates','DayOfWeek','Id','PdDistrict_Index']
test_assembler = VectorAssembler(inputCols=[x for x in test_cleaned.columns if x not in test_ignore],outputCol='features')

test_transformed = test_assembler.transform(test_cleaned)

#test_rdd = test_transformed.map(lambda data: Vectors.dense([float(c) for c in data]))


data_transformed = test_transformed.select(col("Id").alias("label"), col("features")).map(lambda row: LabeledPoint(row.label, row.features))

#Evaluate the model on the training data - output "ID", "prediction"
realTest_labelsAndPreds = data_transformed.map(lambda p: (p.label, (float(nb_model.predict(p.features)))))

output = sqlContext.createDataFrame(realTest_labelsAndPreds,['id','Category_Index'])

#convert back to Categories
#you need SPARK1.6 for this
#in cmd prompt,type in: sudo yum install spark-core spark-master spark-worker spark-python
from pyspark.ml.feature import IndexToString
converter = IndexToString(inputCol="Category_Index", outputCol="originalCategory", labels=classifymodel.labels)
converted = converter.transform(output)

#converted.write.format('com.databricks.spark.csv').save('submission1.csv')

def toCSVLine(data):
  return ','.join(str(d) for d in data)

lines = converted.map(toCSVLine)
lines.saveAsTextFile('submission1.csv')


#view Error rates
#realTest_trainErr = realTest_labelsAndPreds.filter(lambda vp: vp[0] != vp[1]).count() / float(test_transformed.count())
#print("Training Error = " + str(realTest_trainErr))

#model.predict(test_rdd)

# Make prediction and test accuracy.
# Evaluating the model on training data

#test_trainErr = test_labelsAndPreds.filter(lambda vp: vp[0] != vp[1]).count() / float(train.count())
#print("Training Error = " + str(trainErr))

