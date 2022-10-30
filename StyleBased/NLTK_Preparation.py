import numpy as np
import re
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from sklearn.datasets import load_files
nltk.download('stopwords')
import pickle
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
import os

news_data = '/mnt/c/users/EllinorWidman/OneDrive - Tromb/Dokument/Knowledge Graph/C-uppsats/StyleBased/Resultat'
result_root_dir = '/mnt/c/users/EllinorWidman/OneDrive - Tromb/Dokument/Knowledge Graph/C-uppsats/StyleBased/nltk_prep_result'

#X, y = news_data.data, news_data.target
stemmer = WordNetLemmatizer()


def word_count(string):
	return (len(string.strip().split(" ")))

def nltk_preparation():
	for subdir, dirs, files in os.walk(news_data):
		for file in files:
			newFileName = os.path.join(file)
			if newFileName.find('real') >= 0:
				resultdir = result_root_dir + "/Real"
			elif newFileName.find('fake') >= 0:
				resultdir = result_root_dir +  "/Fake"
			resultdir = result_root_dir
			newFileNameWithPath = os.path.join(resultdir, newFileName)
			#print (newFileNameWithPath)
			if os.path.exists(newFileNameWithPath):
				os.remove(newFileNameWithPath)
			f = open (os.path.join(subdir, file), "r")
			OrigTextContent = f.read()

			textContent=''

	#https://stackabuse.com/text-classification-with-python-and-scikit-learn/
			for sen in range(0, len(OrigTextContent)):
				# Remove all the special characters
				CurrentChar = re.sub(r'\W', ' ', str(OrigTextContent[sen]))
				textContent = textContent+CurrentChar

			# remove all single characters
			textContent = re.sub(r'\s+[a-zA-Z]\s+', ' ', textContent)

			# Remove single characters from the start
			textContent = re.sub(r'\^[a-zA-Z]\s+', ' ', textContent)

			# Substituting multiple spaces with single space
			textContent = re.sub(r'\s+', ' ', textContent, flags=re.I)

			# Removing prefixed 'b'
			textContent = re.sub(r'^b\s+', '', textContent)

			# Converting to Lowercase
			textContent = textContent.lower()

			# Lemmatization
			textContent = textContent.split()

			textContent = [stemmer.lemmatize(word) for word in textContent]
			textContent = ' '.join(textContent)



			if word_count(str(textContent))>10:
				newTestFile = open(newFileNameWithPath, "x")
				newTestFile.write(str(textContent))
				#newTestFile.close()
				#print("clean...")
				newTestFile.close()



