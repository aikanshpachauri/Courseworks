
# Part 1: Decision Trees with Categorical Attributes

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics

# Return a pandas dataframe with data set to be mined.
# data_file will be populated with a string 
# corresponding to a path to the adult.csv file.
def read_csv_1(data_file):
	df = pd.read_csv(data_file)
	df = df.drop('fnlwgt', axis = 1)
	return df

# Return the number of rows in the pandas dataframe df.
def num_rows(df):
	return len(df)

# Return a list with the column names in the pandas dataframe df.
def column_names(df):
	return list(df.columns)

# Return the number of missing values in the pandas dataframe df.
def missing_values(df):
	return df.isna().sum().sum()

# Return a list with the columns names containing at least one missing value in the pandas dataframe df.
def columns_with_missing_values(df):
	list_of_req_attributes = []
	for column in df:
		if df[column].isna().sum() > 0:
			list_of_req_attributes.append(column)
	return list_of_req_attributes


# Return the percentage of instances corresponding to persons whose education level is 
# Bachelors or Masters (by rounding to the first decimal digit)
# in the pandas dataframe df containing the data set in the adult.csv file.
# For example, if the percentage is 21.547%, then the function should return 21.6.
def bachelors_masters_percentage(df):
	num_of_ind = 0
	for edu in df['education']:
		if edu == 'Bachelors' or edu == 'Masters':
			num_of_ind = num_of_ind + 1

	return np.ceil((num_of_ind/len(df))*1000)/10

# Return a pandas dataframe (new copy) obtained from the pandas dataframe df 
# by removing all instances with at least one missing value.
def data_frame_without_missing_values(df):
	df = df.dropna(axis = 0)
	return df

# Return a pandas dataframe (new copy) from the pandas dataframe df 
# by converting the df categorical attributes to numeric using one-hot encoding.
# The function's output should not contain the target attribute.
def one_hot_encoding(df):
	df_encoded = df.copy(deep = True)
	df_encoded = df_encoded.drop('class', axis = 1)
	required_attributes = list(df_encoded.select_dtypes(include='object').columns)
	df_encoded = pd.get_dummies(df_encoded, columns = required_attributes) 
	return df_encoded

# Return a pandas series (new copy), from the pandas dataframe df, 
# containing only one column with the labels of the df instances
# converted to numeric using label encoding. 
def label_encoding(df):
	df_target = df['class']
	le = LabelEncoder()
	le.fit(df_target)
	df_target = le.transform(df_target)
	return pd.Series(df_target, name='class')

# Given a training set X_train containing the input attribute values 
# and labels y_train for the training instances,
# build a decision tree and use it to predict labels for X_train. 
# Return a pandas series with the predicted values. 
def dt_predict(X_train,y_train):
	DTC = DecisionTreeClassifier(criterion = 'entropy', splitter = 'random', random_state = 2)
	DTC = DTC.fit(X_train,y_train)

	y_pred = DTC.predict(X_train)
	return pd.Series(y_pred)

# Given a pandas series y_pred with the predicted labels and a pandas series y_true with the true labels,
# compute the error rate of the classifier that produced y_pred.  
def dt_error_rate(y_pred, y_true):
	return round(1 - metrics.accuracy_score(y_pred, y_true), 3)



