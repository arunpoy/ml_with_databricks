# Databricks notebook source
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

# COMMAND ----------

store_revenue = pd.read_csv('/dbfs/FileStore/datasets/store.csv')

store_revenue.sample(10)

# COMMAND ----------

store_revenue.shape

# COMMAND ----------

store_revenue.isnull().sum()

# COMMAND ----------

store_revenue = store_revenue.dropna()

sum(store_revenue.isnull().sum())

# COMMAND ----------

# MAGIC %md
# MAGIC After deleting NULL values or unwanted columns again checking the number rows and columns present in the datatset

# COMMAND ----------

store_revenue.shape

# COMMAND ----------

# MAGIC %md
# MAGIC Summary statistics of whole dataset is obtained

# COMMAND ----------

store_revenue.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC Correlation matrix is obtained using corr() method

# COMMAND ----------

store_revenue.corr()

# COMMAND ----------

# MAGIC %md
# MAGIC Checking the same correlation matrix  above  using heatmap

# COMMAND ----------

fig1, ax = plt.subplots(figsize = (12, 10))

sns.heatmap(store_revenue.corr(), annot = True)

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC We know that event values are not numeric, so we need to convert the event columns in to numeric column which we will do later.

# COMMAND ----------

store_revenue['event'].unique()

# COMMAND ----------

# MAGIC %md
# MAGIC Obtaining  revenue under these four types of event
# MAGIC 
# MAGIC special event earned the most revenue averagely
# MAGIC cobranding is the most frequent event

# COMMAND ----------

store_revenue.groupby(['event'])['revenue'].describe()

# COMMAND ----------

# MAGIC %md
# MAGIC Event wise revenue breakdown using barplot

# COMMAND ----------

fig2, ax = plt.subplots(figsize = (12, 8))

sns.barplot(x = 'event', y = 'revenue', data = store_revenue)

# COMMAND ----------

# MAGIC %md
# MAGIC Visualising Relationships using Scatter plot. As observed above..Positive correlation is observed between investment in local ads and Revenue

# COMMAND ----------

fig3, ax = plt.subplots(figsize = (12, 8))

sns.scatterplot(x = 'local_tv', y = 'revenue', data = store_revenue )

plt.xlabel('LocalTV_Ads_Investment')
plt.ylabel('Overall_Revenue')

# COMMAND ----------

# MAGIC %md
# MAGIC Converting catergorical values in to Numerical values

# COMMAND ----------

store_revenue = pd.get_dummies(store_revenue, columns = ['event'])

store_revenue.head()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC We can introduce mlflow by setting our created workspace experiment to log all runs therafter.Experiments can also be seen by clicking on the Flask in the header menu.

# COMMAND ----------

mlflow.set_experiment(experiment_name = '/Users/abc@email.com/store_revenue_prediction')

# COMMAND ----------

# MAGIC %md
# MAGIC - We can log our figures as artifacts in our run for  Mlflow experiment.
# MAGIC - Click on view run detail boxed arrow on right side and click on Artifacts section. 
# MAGIC - Click on the artifacts and show that they are our plots
# MAGIC - Note that some random name is being given to this run.

# COMMAND ----------

mlflow.start_run()
mlflow.log_figure(fig1, 'corrplot_heatmap.png')
mlflow.log_figure(fig2, 'barplot_revenue_event_wise.png')
mlflow.log_figure(fig3, 'localtvads_revenue.png')
mlflow.end_run()

# COMMAND ----------

# MAGIC %md
# MAGIC Splitting the data for test and train.Note that We are converting Predictors df as Float type to avoid warnings related to missing data

# COMMAND ----------

# DBTITLE 1,Regression model
X = store_revenue.drop('revenue', axis = 1)
y = store_revenue['revenue']
X = X.astype(np.float64)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

# COMMAND ----------

X_train.shape, X_test.shape, y_train.shape, y_test.shape

# COMMAND ----------

# MAGIC %md
# MAGIC By the below box plot we can  see that our numeric columns are in very different ranges of values before scaling

# COMMAND ----------

store_revenue.boxplot(column = ['local_tv', 'reach', 'online', 'instore', 'person'], figsize = (12, 8))

plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC Separate datasets are made for numerical and categorical columns before performing Scaling

# COMMAND ----------

numerical_data = ['reach', 'local_tv', 'online', 'instore', 'person']
categorical_data = ['event_cobranding', 'event_holiday', 'event_non_event', 'event_special']

X_train_numerical = X_train[numerical_data]
X_test_numerical = X_test[numerical_data]

X_train_numerical.head()

# COMMAND ----------

X_train_categorical = X_train[categorical_data]
X_test_categorical = X_test[categorical_data]

X_train_categorical.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Here we are scaling our  columns of separate dataframe which is  made up of  numerical columns from training set and scaling columns of  separate dataframe which is made up of numerical columns from test set(using the same mean, stddev from the training set)

# COMMAND ----------

scaler = StandardScaler()

X_train_numerical = pd.DataFrame(scaler.fit_transform(X_train_numerical), 
                                 columns = X_train_numerical.columns)

X_test_numerical = pd.DataFrame(scaler.transform(X_test_numerical),
                                columns = X_test_numerical.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC Combining our scaled numeric columns with our one-hot-encoded categorical columns( only the training dataset)

# COMMAND ----------

X_train_categorical.reset_index(drop = True, inplace = True)
X_train_numerical.reset_index(drop = True, inplace = True)

X_train = pd.concat([X_train_numerical, X_train_categorical], axis = 1)

X_train.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Combining our scaled numeric columns with our one-hot-encoded categorical columns( only the testing dataset)

# COMMAND ----------

X_test_categorical.reset_index(drop = True, inplace = True)
X_test_numerical.reset_index(drop = True, inplace = True)

X_test = pd.concat([X_test_numerical, X_test_categorical], axis = 1)

X_test.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Post scaling, The numerical variables distributions can be seen in more or less same range

# COMMAND ----------

X_train.boxplot(column = ['local_tv', 'reach', 'online', 'instore', 'person'], figsize = (12, 8))

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC We are enabling autologging of parameters and metrics. All training parameters and training metrics are autologged.
# MAGIC 
# MAGIC Fitting the model into training data. Here The run remains open throughout the with statement, and is automatically closed when the statement exits, even if it exits due to an exception.We are specifying run name for linear_regression_model so that random name is not given.
# MAGIC 
# MAGIC https://learn.microsoft.com/en-gb/azure/databricks/mlflow/databricks-autologging
# MAGIC 
# MAGIC Understanding the artifacts in the ML model
# MAGIC https://www.mlflow.org/docs/latest/models.html

# COMMAND ----------

mlflow.sklearn.autolog()

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

with mlflow.start_run(run_name = 'linear_regression_model') as run1:

      lr = LinearRegression()
      lr.fit(X_train, y_train)

      y_pred =  lr.predict(X_test)
      
      testing_score = r2_score(y_test, y_pred)
      mean_absolute_score = mean_absolute_error(y_test, y_pred)
      mean_sq_error = mean_squared_error(y_test, y_pred)
      
      run1 = mlflow.active_run()
      
      print('Active run_id: {}'.format(run1.info.run_id))

# COMMAND ----------

# MAGIC %md
# MAGIC Running one more run this time using Randomforest Regressor on same training data.

# COMMAND ----------

with mlflow.start_run(run_name = 'randomforest_regression_model') as run2:
      rf = RandomForestRegressor()
      rf.fit(X_train, y_train)

      y_pred =  rf.predict(X_test)
      
      testing_score = r2_score(y_test, y_pred)
      mean_absolute_score = mean_absolute_error(y_test, y_pred)
      mean_sq_error = mean_squared_error(y_test, y_pred)
      
      run2 = mlflow.active_run()
      
      print('Active run_id: {}'.format(run2.info.run_id))

# COMMAND ----------

with mlflow.start_run(run_name = 'knn_regression_model') as run3:
      knn = KNeighborsRegressor()
      knn.fit(X_train, y_train)
      
      y_pred =  knn.predict(X_test)
      
      testing_score = r2_score(y_test, y_pred)
      mean_absolute_score = mean_absolute_error(y_test, y_pred)
      mean_sq_error = mean_squared_error(y_test, y_pred)
      
      run3 = mlflow.active_run()
      
      print('Active run_id: {}'.format(run3.info.run_id))

# COMMAND ----------

run1.info.run_id

# COMMAND ----------

run_id1 = run1.info.run_id
model_uri = 'runs:/' + run_id1 + '/model'

# COMMAND ----------

import mlflow.sklearn

model = mlflow.sklearn.load_model(model_uri = model_uri)

model.coef_

# COMMAND ----------

# MAGIC %md
# MAGIC Obtaining predictions for test data

# COMMAND ----------

y_pred = model.predict(X_test)

y_pred

# COMMAND ----------

# MAGIC %md
# MAGIC The loaded model should match the original

# COMMAND ----------

predictions_loaded = model.predict(X_test)
predictions_original = lr.predict(X_test)
 
assert(np.array_equal(predictions_loaded, predictions_original))

# COMMAND ----------

# MAGIC %md
# MAGIC Creating  the PySpark UDF.

# COMMAND ----------

import mlflow.pyfunc

pyfunc_udf = mlflow.pyfunc.spark_udf(spark, model_uri = model_uri, env_manager = 'conda')

# COMMAND ----------

# MAGIC %md
# MAGIC Spark dataframe is created for Test data

# COMMAND ----------

X_test_sp = spark.createDataFrame(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC Predictions are obtained and are same as predictions made earlier

# COMMAND ----------

from pyspark.sql.functions import struct
 
predicted_df = X_test_sp.withColumn('prediction', pyfunc_udf(struct(*(X_test.columns.tolist()))))
display(predicted_df)

# COMMAND ----------


