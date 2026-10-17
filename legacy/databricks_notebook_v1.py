# Databricks notebook source
# MAGIC %pip install langchain

# COMMAND ----------

import langchain

# COMMAND ----------

import tensorflow

# COMMAND ----------

# MAGIC %md
# MAGIC #Lấy dữ liệu từ mongodb và tiền xử lí (nội suy tuyến tính)

# COMMAND ----------

# MAGIC %md
# MAGIC Kéo dữ liệu và lấy các trường quan trọng

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import *

# COMMAND ----------

# Thông tin kết nối lưu trong Databricks secret scope, không để trong code
connectionString = dbutils.secrets.get(scope="air-quality", key="mongo-uri")
data_85 = spark.read \
                .format("mongodb") \
                .option("spark.mongodb.read.connection.uri", connectionString) \
                .option("database", "sic") \
                .option("collection", "3409385") \
                .load()

# COMMAND ----------

df_85 = data_85 \
  .withColumn("result", F.explode(F.col("results")))\
  .select( \
    F.to_timestamp(F.col("result.period.datetimeFrom.local").substr(0, 19)).alias("local"), \
    F.to_timestamp(F.col("result.period.datetimeFrom.utc")).alias("datetime_utc"), \
    F.col("result.period.interval"), \
    F.col("result.parameter.name").alias("parameter"), \
    F.col("result.parameter.units").alias("unit"), \
    F.col("result.value") \
  ) 

display(df_85)

# COMMAND ----------

# MAGIC %md
# MAGIC Nội suy

# COMMAND ----------

temp_85 = df_85.filter(F.col("parameter") == "temperature") \
                  .withColumn("true_value", F.when(F.col("value") <= 0, None).otherwise(F.col("value"))) \
                  .withColumn("null_local", F.when(F.col("value") <= 0, None).otherwise(F.col("local"))) \
                  .orderBy(F.col("local"))
display(temp_85)

# COMMAND ----------

windowForward = Window.orderBy("local").rowsBetween(Window.unboundedPreceding, -1)
windowBackward = Window.orderBy("local").rowsBetween(1, Window.unboundedFollowing)

temp_85_withBounds = temp_85 \
  .withColumn("prev_value", F.last("true_value", True).over(windowForward)) \
  .withColumn("prev_timestamp", F.last("null_local", True).over(windowForward)) \
  .withColumn("next_value", F.first("true_value", True).over(windowBackward)) \
  .withColumn("next_timestamp", F.first("null_local", True).over(windowBackward))

display(temp_85_withBounds)

# COMMAND ----------

filled_temp_85 = temp_85_withBounds \
.withColumn("interpolated_value", F.when(F.col("true_value").isNotNull(), F.col("value")) \
  .otherwise( \
    # Công thức nội suy: v1 + (v2 - v1) * (t - t1) / (t2 - t1)
    F.col("prev_value") + (F.col("next_value") - F.col("prev_value")) * \
    (F.unix_timestamp(F.col("local")) - F.unix_timestamp(F.col("prev_timestamp"))) / \
    (F.unix_timestamp(F.col("next_timestamp")) - F.unix_timestamp(F.col("prev_timestamp"))) \
  ) \
) \
.select("local", "interpolated_value") \
# .withColumn("hour", F.col("local").substr(12,2))

display(filled_temp_85.limit(10))

# COMMAND ----------

display(filled_temp_85.withColumn("hour", F.col("local").substr(12,2)))

# COMMAND ----------

# MAGIC %md
# MAGIC #LSTM Model

# COMMAND ----------

# MAGIC %md
# MAGIC ##Training

# COMMAND ----------

import tensorflow as tf
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# COMMAND ----------

input = spark.read \
                .option("header", True) \
                .option("inferschema", True) \
                .csv("dbfs:/FileStore/AQ_temprature.csv") 

# COMMAND ----------

df = input.select("T (degC)").toPandas()
# df = df[5::6]

# COMMAND ----------

temp = df['T (degC)']
temp.plot()

# COMMAND ----------

def df_to_X_y(df, window_size=5):
  df_as_np = df.to_numpy()
  X = []
  y = []
  for i in range(len(df_as_np)-window_size):
    row = [[a] for a in df_as_np[i:i+window_size]]
    X.append(row)
    label = df_as_np[i+window_size]
    y.append(label)
  return np.array(X), np.array(y)

# COMMAND ----------

temp

# COMMAND ----------

WINDOW_SIZE = 24
X1, y1 = df_to_X_y(temp, WINDOW_SIZE)
X1.shape, y1.shape

# COMMAND ----------

X_train1, y_train1 = X1[:410000], y1[:410000]
X_val1, y_val1 = X1[410000:415000], y1[410000:415000]
X_test1, y_test1 = X1[415000:], y1[415000:]
X_train1.shape, y_train1.shape, X_val1.shape, y_val1.shape, X_test1.shape, y_test1.shape

# COMMAND ----------

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

model1 = Sequential()
model1.add(InputLayer((WINDOW_SIZE, 1)))
model1.add(LSTM(64))
model1.add(Dense(8, 'relu'))
model1.add(Dense(1, 'linear'))

model1.summary()

# COMMAND ----------

cp1 = ModelCheckpoint('model1.keras', save_best_only=True)
model1.compile(loss=MeanSquaredError(),
               optimizer=Adam(learning_rate=0.0001),
               metrics=[RootMeanSquaredError()])

# COMMAND ----------

X_train1

# COMMAND ----------

model1.fit(X_train1, y_train1, validation_data=(X_val1, y_val1), epochs=2, callbacks=[cp1])

# COMMAND ----------

loss_per_epoch = model1.history.history['loss']
plt.plot(range(len(loss_per_epoch)),loss_per_epoch)

# COMMAND ----------

from tensorflow.keras.models import load_model
model1 = load_model('model1.keras')

# COMMAND ----------

train_predictions = model1.predict(X_train1).flatten()
train_results = pd.DataFrame(data={'Train Predictions':train_predictions, 'Actuals':y_train1})
train_results

# COMMAND ----------

from sklearn.metrics import mean_squared_error
mean_squared_error(train_results['Train Predictions'], train_results['Actuals'])

# COMMAND ----------

import matplotlib.pyplot as plt
plt.plot(train_results['Train Predictions'][50:100])
plt.plot(train_results['Actuals'][50:100])

# COMMAND ----------

val_predictions = model1.predict(X_val1).flatten()
val_results = pd.DataFrame(data={'Val Predictions':val_predictions, 'Actuals':y_val1})
val_results

# COMMAND ----------

plt.plot(val_results['Val Predictions'][:100])
plt.plot(val_results['Actuals'][:100])

# COMMAND ----------

test_predictions = model1.predict(X_test1).flatten()
test_results = pd.DataFrame(data={'Test Predictions':test_predictions, 'Actuals':y_test1})
test_results

# COMMAND ----------

plt.plot(test_results['Test Predictions'][:100])
plt.plot(test_results['Actuals'][:100])

# COMMAND ----------

# MAGIC %md
# MAGIC ##Kiểm thử với dữ liệu trong trạm id: 3409385

# COMMAND ----------

real_test = filled_temp_85.toPandas()
# real_test = real_test[2::4]
real_test.head()

# COMMAND ----------

X2, y2 = df_to_X_y(real_test['interpolated_value'], WINDOW_SIZE)
X2.shape, y2.shape

# COMMAND ----------

test_predictions2 = model1.predict(X2).flatten()
test_results2 = pd.DataFrame(data={'Test Predictions':test_predictions2, 'Actuals':y2})
test_results2

# COMMAND ----------

test_results2.plot()

# COMMAND ----------

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse=sqrt(mean_squared_error(test_results2['Test Predictions'],test_results2['Actuals']))
print(rmse)

# COMMAND ----------

mean_squared_error(test_results2['Test Predictions'],test_results2['Actuals'])

# COMMAND ----------

# MAGIC %md
# MAGIC #Tích hợp mô hình

# COMMAND ----------

lastRecord = filled_temp_85.orderBy(F.desc("local")).first()
display(lastRecord)

# COMMAND ----------

inputData = filled_temp_85.orderBy(F.desc("local")) \
                                .limit(WINDOW_SIZE + 1) \
                                .select("interpolated_value") \
                                .toPandas()

display(inputData)

# COMMAND ----------

def predict():
    global filled_temp_85
    lastRecord = filled_temp_85.orderBy(F.desc("local")).first()
    if lastRecord[1] <= 0:
        filled_temp_85 = filled_temp_85.filter(F.col("local") != lastRecord[0])
        inputData = filled_temp_85.orderBy(F.desc("local")) \
                                .limit(WINDOW_SIZE + 1) \
                                .orderBy("local") \
                                .select("interpolated_value") \
                                .toPandas()[1:]
        inputData = np.array(inputData).reshape(1,24,1)
        pred = model1.predict(inputData)

        columns = ['local', 'interpolated_value']
        data = [(lastRecord[0], float(pred[0][0]))]
        predictedRecord = spark.createDataFrame(data, columns)

        filled_temp_85 = filled_temp_85.unionByName(predictedRecord)

# COMMAND ----------

import time
from datetime import datetime, timedelta, timezone

# COMMAND ----------

def run_job_every_quarter():
    now = datetime.now(timezone.utc)
    next_minute = ((now.minute // 15) + 1) * 15 + 1
    if next_minute == 60:
        next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        next_time = now.replace(minute=next_minute, second=0, microsecond=0)
    sleep_sec = (next_time - now).total_seconds()
    print(f"Truy vấn tiếp theo lúc {next_time} UTC (còn {int(sleep_sec)} giây)")
    time.sleep(sleep_sec)

while True:
    predict()
    run_job_every_quarter()


# COMMAND ----------

# MAGIC %md
# MAGIC #Trực quan hóa BI

# COMMAND ----------

df_all = spark.read.option("header", True).option("ifnerschema", True).csv("dbfs:/FileStore/zOut.csv").filter(F.col("interpolated_value") != 'null')
display(df_all)

# COMMAND ----------

q = [0.98]

# COMMAND ----------

co = df_all.filter(F.col("parameter") == "co").select("local", (F.col("value").cast("Double")).alias("co"))
# avg_co = co.agg(F.avg(F.col("co"))).first()["avg(co)"]
# display(co.limit(10))
q3_co = co.approxQuantile("co", q, 0.01)[0]
print(q3_co)

# COMMAND ----------

pm25 = df_all.filter(F.col("parameter") == "pm25").select("local", (F.col("value").cast("Double")).alias("pm25"))
# display(pm25.limit(10))
# avg_pm25 = pm25.agg(F.avg(F.col("pm25"))).first()["avg(pm25)"]
q3_pm25 = pm25.approxQuantile("pm25", q, 0.01)[0]
print(q3_pm25)

# COMMAND ----------

so2 = df_all.filter(F.col("parameter") == "so2").select("local", (F.col("value").cast("Double")).alias("so2"))
# display(so2.limit(10))
# avg_so2 = so2.agg(F.avg(F.col("so2"))).first()["avg(so2)"]
q3_so2 = so2.approxQuantile("so2", q, 0.01)[0]
print(q3_so2)

# COMMAND ----------

o3 = df_all.filter(F.col("parameter") == "o3").select("local", (F.col("value").cast("Double")).alias("o3"))
# display(o3.limit(10))
# avg_o3 = o3.agg(F.avg(F.col("o3"))).first()["avg(o3)"]
q3_o3 = o3.approxQuantile("o3", q, 0.01)[0]
print(q3_o3)

# COMMAND ----------

no2 = df_all.filter(F.col("parameter") == "no2").select("local",( F.col("value").cast("Double")).alias("no2"))
# display(no2.limit(10))
# avg_no2 = no2.agg(F.avg(F.col("no2"))).first()["avg(no2)"]
q3_no2 = no2.approxQuantile("no2", q, 0.01)[0]
print(q3_no2)

# COMMAND ----------

temperature = df_all.filter(F.col("parameter") == "temperature").select("local", (F.col("value").cast("Double")).alias("temperature"))
# avg_temperature = temperature.agg(F.avg(F.col("temperature"))).first()["avg(temperature)"]
q3_temperature = temperature.approxQuantile("temperature", q, 0.01)[0]
print(q3_temperature)

# COMMAND ----------

no = df_all.filter(F.col("parameter") == "no").select("local", (F.col("value").cast("Double")).alias("no"))
# avg_no = no.agg(F.avg(F.col("no"))).first()["avg(no)"]
q3_no = no.approxQuantile("no", q, 0.01)[0]
print(q3_no)

# COMMAND ----------

pm10 = df_all.filter(F.col("parameter") == "pm10").select("local",( F.col("value").cast("Double")).alias("pm10"))
# avg_pm10 = pm10.agg(F.avg(F.col("pm10"))).first()["avg(pm10)"]
q3_pm10 = pm10.approxQuantile("pm10", q, 0.01)[0]
print(q3_pm10)

# COMMAND ----------

relativehumidity = df_all.filter(F.col("parameter") == "relativehumidity").select("local", (F.col("value").cast("Double")).alias("relativehumidity"))
# avg_relativehumidity = relativehumidity.agg(F.avg(F.col("relativehumidity"))).first()["avg(relativehumidity)"]
q3_relativehumidity = relativehumidity.approxQuantile("relativehumidity", q, 0.01)[0]
print(q3_relativehumidity)

# COMMAND ----------

merge = co.join(pm25, "local", "full") \
            .join(no, "local", "full") \
            .join(no2, "local", "full") \
            .join(o3, "local", "full") \
            .join(pm10, "local", "full") \
            .join(so2, "local", "full") \
            .filter((F.col("no2") > 0) & (F.col("no") > 0))

display(merge)

# COMMAND ----------

merge_scaled = merge.withColumn("date", F.col("local").substr(0, 10)) \
.groupBy("date").agg( \
        F.avg(F.col("co")).alias("co"), \
        F.avg(F.col("no")).alias("no"), \
        F.avg(F.col("no2")).alias("no2"), \
        F.avg(F.col("so2")).alias("so2"), \
        F.avg(F.col("o3")).alias("o3"), \
        F.avg(F.col("pm25")).alias("pm25"), \
        F.avg(F.col("pm10")).alias("pm10"), \
    ) 

# COMMAND ----------

def scale(x, bound):
    return x / bound
scale_col = F.udf(scale, DoubleType())

merge_scaled = merge.withColumn("date", F.col("local").substr(0, 10)) \
    .groupBy("date").agg( \
        F.avg(F.col("co")).alias("co"), \
        F.avg(F.col("no")).alias("no"), \
        F.avg(F.col("no2")).alias("no2"), \
        F.avg(F.col("so2")).alias("so2"), \
        F.avg(F.col("o3")).alias("o3"), \
        F.avg(F.col("pm25")).alias("pm25"), \
        F.avg(F.col("pm10")).alias("pm10"), \
    ) \
    .select( \
        "date", \
        scale_col(F.col("co"), F.lit(1747)).alias("co"), \
        scale_col(F.col("no"), F.lit(13.3)).alias("no"), \
        scale_col(F.col("no2"), F.lit(13.3)).alias("no2"), \
        scale_col(F.col("so2"), F.lit(15.3)).alias("so2"), \
        scale_col(F.col("o3"), F.lit(81.5)).alias("o3"), \
        scale_col(F.col("pm25"), F.lit(15)).alias("pm25"), \
        scale_col(F.col("pm10"), F.lit(45)).alias("pm10") \
    ) \
    .orderBy("date")

display(merge_scaled)

# COMMAND ----------

# trung bình gấp bao nhiêu lần mức cho phép


# COMMAND ----------

merge_hour_nox = merge.withColumn("date", F.col("local").substr(0, 10)) \
    .groupBy("date").agg( \
        # F.round(F.avg(F.col("co")), 4).alias("co"), \
        F.round(F.avg(F.col("no")), 4).alias("no"), \
        F.round(F.avg(F.col("no2")), 4).alias("no2"), \
        # F.round(F.avg(F.col("so2")), 4).alias("so2"), \
        # F.round(F.avg(F.col("o3")), 4).alias("o3"), \
        # F.round(F.avg(F.col("pm25")), 4).alias("pm25"), \
        # F.round(F.avg(F.col("pm10")), 4).alias("pm10"), \
    ) \
                
display(merge_hour_nox)

# COMMAND ----------

merge_hour_pmx = merge.withColumn("date", F.col("local").substr(0, 10)) \
    .groupBy("date").agg( \
        # F.round(F.avg(F.col("co")), 4).alias("co"), \
        # F.round(F.avg(F.col("no")), 4).alias("no"), \
        # F.round(F.avg(F.col("no2")), 4).alias("no2"), \
        # F.round(F.avg(F.col("so2")), 4).alias("so2"), \
        # F.round(F.avg(F.col("o3")), 4).alias("o3"), \
        F.round(F.avg(F.col("pm25")), 4).alias("pm25"), \
        F.round(F.avg(F.col("pm10")), 4).alias("pm10"), \
    ) \
                
display(merge_hour_pmx)