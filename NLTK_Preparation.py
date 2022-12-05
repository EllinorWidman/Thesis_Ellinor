import numpy as np
import re
import nltk
from tqdm import tqdm
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

#news_data = '/mnt/c/users/EllinorWidman/OneDrive - Tromb/Dokument/Knowledge Graph/C-uppsats/StyleBased/Resultat'
#result_root_dir = '/mnt/c/users/EllinorWidman/OneDrive - Tromb/Dokument/Knowledge Graph/C-uppsats/StyleBased/nltk_prep_result'
DATA_FOLDER = "data"


#X, y = news_data.data, news_data.target
stemmer = WordNetLemmatizer()


def word_count(string):
	return (len(string.strip().split(" ")))

def nltk_preparation(dataset_name):

	#data_name =  dataset_name
	workload_folder = "{}/{}/workload".format(DATA_FOLDER, dataset_name)
	nltk_folder = "{}/{}/result_nltk_prep".format(DATA_FOLDER, dataset_name)

	print(dataset_name)
	print(workload_folder)
	print(nltk_folder)
	counter = 0

	for subdir, dirs, files in os.walk(workload_folder):
	#for subdir, dirs, files in tqdm(enumerate(os.walk(workload_folder))):
		for file in files:
			counter = counter + 1
			if counter % 100 == 0:
				print("Processed: ",counter)
			newFileName = os.path.join(file)
			if newFileName.lower().find('real') >= 0:
				resultdir = nltk_folder + "/Real"
			elif newFileName.lower().find('fake') >= 0:
				resultdir = nltk_folder +  "/Fake"
			else:
				resultdir = nltk_folder

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

	print("Processed: ",counter)


