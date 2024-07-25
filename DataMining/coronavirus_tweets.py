# Part 3: Text mining.
import pandas as pd
import numpy as np
import requests
import nltk
from nltk.stem import PorterStemmer
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from nltk.tokenize import word_tokenize

url = "https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt"

# Return a pandas dataframe containing the data set.
# Specify a 'latin-1' encoding when reading the data.
# data_file will be populated with a string 
# corresponding to a path containing the wholesale_customers.csv file.
def read_csv_3(data_file):
	df = pd.read_csv(data_file, encoding = 'latin-1')
	return df

# Return a list with the possible sentiments that a tweet might have.
def get_sentiments(df):
	return list(df["Sentiment"].unique())

# Return a string containing the second most popular sentiment among the tweets.
def second_most_popular_sentiment(df):
	sentiment_count = {}
	for sentiment in list(df["Sentiment"].unique()):
		sentiment_count[sentiment] = df["Sentiment"].value_counts()[sentiment]

	sentiment_count_sorted = sorted(sentiment_count.items(), key = lambda item: item[1])
	return sentiment_count_sorted[-2][0]

# Return the date (string as it appears in the data) with the greatest number of extremely positive tweets.
def date_most_popular_tweets(df):
	max_pos_count = 0
	required_date = 0
	grouped_data = df.groupby(df["TweetAt"])["Sentiment"].value_counts()

	for date in list(df["TweetAt"].unique()):
		if grouped_data[date]["Extremely Positive"] > max_pos_count:
			max_pos_count = grouped_data[date]["Extremely Positive"]
			required_date = date

	return required_date

# Modify the dataframe df by converting all tweets to lower case. 
def lower_case(df):
	df["OriginalTweet"] = df["OriginalTweet"].str.lower()
	return df

# Modify the dataframe df by replacing each characters which is not alphabetic or whitespace with a whitespace.
def remove_non_alphabetic_chars(df):
	df["OriginalTweet"] = df["OriginalTweet"].apply(lambda x : re.sub(r'[^A-Za-z]',' ', str(x)))
	return df

# Modify the dataframe df with tweets after removing characters which are not alphabetic or whitespaces.
def remove_multiple_consecutive_whitespaces(df):
	df["OriginalTweet"] = df["OriginalTweet"].apply(lambda x : re.sub(r'\s+', ' ', str(x)))
	return df

# Given a dataframe where each tweet is one string with words separated by single whitespaces,
# tokenize every tweet by converting it into a list of words (strings).
def tokenize(df):
	df['OriginalTweet'] = df['OriginalTweet'].apply(lambda x : word_tokenize(x))
	return df

# Given dataframe tdf with the tweets tokenized, return the number of words in all tweets including repetitions.
def count_words_with_repetitions(tdf):
	return len(tdf['OriginalTweet'].explode())

# Given dataframe tdf with the tweets tokenized, return the number of distinct words in all tweets.
def count_words_without_repetitions(tdf):
	return tdf['OriginalTweet'].explode().unique()

# Given dataframe tdf with the tweets tokenized, return a list with the k distinct words that are most frequent in the tweets.
def frequent_words(tdf,k):
	word_freq = {}
	for tweet_words in tdf['OriginalTweet']:
		for word in tweet_words:
			if word in word_freq:
				word_freq[word] += 1
			else:
				word_freq[word] = 1
    
	most_common_words = sorted(word_freq, key=word_freq.get, reverse=True)[:k]
	return most_common_words

# Given dataframe tdf with the tweets tokenized, remove stop words and words with <=2 characters from each tweet.
# The function should download the list of stop words via:
# https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt
def remove_stop_words(tdf):
	nltk.download('punkt')
	response = requests.get(url)
	
	def clean_tokens(tokens):
		new_token = []
		for token in tokens:
			if token not in response.text.split() and len(token) > 2:
				new_token.append(token)
		return new_token
	
	tdf['OriginalTweet'] = tdf['OriginalTweet'].apply(clean_tokens)
	return tdf

# Given dataframe tdf with the tweets tokenized, reduce each word in every tweet to its stem.
def stemming(tdf):
	ps = PorterStemmer()
	
	def porter_stemmer(list_of_words):
		clean_list = []
		for word in list_of_words:
			clean_list.append(ps.stem(word))
		return clean_list
    
	tdf['OriginalTweet'] = tdf['OriginalTweet'].apply(porter_stemmer)
	return tdf

# Given a pandas dataframe df with the original coronavirus_tweets.csv data set,
# build a Multinomial Naive Bayes classifier. 
# Return predicted sentiments (e.g. 'Neutral', 'Positive') for the training set
# as a 1d array (numpy.ndarray). 
def mnb_predict(df):
	X = df['OriginalTweet'].to_numpy()
	label_encoder = LabelEncoder()
	y = label_encoder.fit_transform(df['Sentiment'])
	
	count_vec = CountVectorizer(min_df=1, max_df=10, ngram_range=(1, 2))
	X_token = count_vec.fit_transform(X)
	
	classifier = MultinomialNB(alpha = 0.001)
	classifier.fit(X_token, y)

	predicted_sentiments = label_encoder.inverse_transform(classifier.predict(X_token))
	return predicted_sentiments


# Given a 1d array (numpy.ndarray) y_pred with predicted labels (e.g. 'Neutral', 'Positive') 
# by a classifier and another 1d array y_true with the true labels, 
# return the classification accuracy rounded in the 3rd decimal digit.
def mnb_accuracy(y_pred,y_true):
	return round(metrics.accuracy_score(y_true, y_pred), 3)






