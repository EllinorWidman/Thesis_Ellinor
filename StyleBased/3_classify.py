#https://stackabuse.com/text-classification-with-python-and-scikit-learn/
#import numpy as np
import re
import nltk
from sklearn.datasets import load_files
nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier

news_data = load_files(r"C:\Users\HP\Documents\Skola\C-uppsats\FakeNewsNet\code\nltk_prep_result")
X, y = news_data.data, news_data.target

documents = []

#from nltk.stem import WordNetLemmatizer

#stemmer = WordNetLemmatizer()

for sen in range(0, len(X)):
 #   # Remove all the special characters
  #  document = re.sub(r'\W', ' ', str(X[sen]))
	document = str(X[sen])

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(documents).toarray()

from sklearn.feature_extraction.text import TfidfTransformer

tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()

documents.append(document)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(documents).toarray()

from sklearn.feature_extraction.text import TfidfTransformer
tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()


#You can also directly convert text documents into TFIDF feature values (without first converting documents to bag of words features) using the following script:

#from sklearn.feature_extraction.text import TfidfVectorizer
#tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
#X = tfidfconverter.fit_transform(documents).toarray()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train) 


y_pred = classifier.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

with open('text_classifier', 'wb') as picklefile:
    pickle.dump(classifier,picklefile)


with open('text_classifier', 'rb') as training_model:
    model = pickle.load(training_model)


y_pred2 = model.predict(X_test)

print(confusion_matrix(y_test, y_pred2))
print(classification_report(y_test, y_pred2))
print(accuracy_score(y_test, y_pred2)) 

