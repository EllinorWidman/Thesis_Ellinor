# -*- coding: utf-8 -*-

import os
import sys
import csv
#import tarfile
import shutil
import hashlib
from tqdm import tqdm
#from urllib.request import urlretrieve
#from urllib.error import URLError
#from urllib.error import HTTPError
from NLTK_Preparation import nltk_preparation


csv.field_size_limit(100000000)
DATA_FOLDER = "data"


class FakeNewsNet(object):

    def __init__(self):

        self.data_name = 'FakeNewsNet'
        self.data_folder = "{}/{}/input".format(DATA_FOLDER, self.data_name)
        self.workload_folder = "{}/{}/workload".format(DATA_FOLDER, self.data_name)
        print(self.data_folder)
        print(self.workload_folder)

        for subdir, dirs, files in os.walk(self.data_folder):
            for file in files:
                newFileName = os.path.join(subdir, file).replace("/", "_").replace("_news content.json", "")
                if newFileName.find('real') >= 0:
                    resultdir = self.workload_folder + "/Real"
                elif newFileName.find('fake') >= 0:
                    resultdir = self.workload_folder + "/Fake"
                newFileNameWithPath = os.path.join(resultdir, newFileName + ".txt")
                if os.path.exists(newFileNameWithPath):
                    os.remove(newFileNameWithPath)
                newTestFile = open(newFileNameWithPath, "x")
                f = open(os.path.join(subdir, file), "r")
                textContent = f.read()
                posText = textContent.find('"text":')
                # print(posText)
                posImg = textContent.find('"images":')
                # print(posImg)
                posTitle = textContent.find('"title":')
                # print(posTitle)
                posMeta = textContent.find('"meta_data":')
                # print(posMeta)
                textContent = (textContent[posTitle:posMeta]) + (textContent[posText:posImg])
                newTestFile.write(str(textContent))
                newTestFile.close()

class ISOT_FakeNewsDataset(object):

    def __init__(self):

        self.data_name = 'ISOT_FakeNewsDataset'
        self.data_folder = "{}/{}/input".format(DATA_FOLDER, self.data_name)
        self.workload_folder = "{}/{}/workload".format(DATA_FOLDER, self.data_name)
        print(self.data_folder)
        print(self.workload_folder)

        for subdir, dirs, files in os.walk(self.data_folder):
            for filename in files:

                if filename.find('True') >= 0:
                    resultdir = self.workload_folder + "/Real"
                    file_name_prefix = "Real_"
                elif filename.find('Fake') >= 0:
                    resultdir = self.workload_folder + "/Fake"
                    file_name_prefix = "Fake_"
                print(filename)
                with open(self.data_folder + "/" + filename, "r") as f:
                    readCSV = csv.reader(f, delimiter=',')
                    csv.field_size_limit(100000000)
                    sents = []
                    for idx, (title, text, subject, date) in tqdm(enumerate(readCSV)):
                        # print ([idx, title, text,subject,date])
                        newFileName = file_name_prefix + str(idx) + ".txt"
                        newFileNameWithPath = os.path.join(resultdir,newFileName)
                        textContent = dict()
                        textContent["title"] = title
                        textContent["text"] = text
                        if os.path.exists(newFileNameWithPath):
                            os.remove(newFileNameWithPath)

                        newTestFile = open(newFileNameWithPath, "x")
                        newTestFile.write(str(textContent))
                        newTestFile.close()

class Kaggle_Ruchi7798(object):

    def __init__(self):

        self.data_name = 'kaggle_ruchi7798'
        self.data_folder = "{}/{}/input".format(DATA_FOLDER, self.data_name)
        self.workload_folder = "{}/{}/workload".format(DATA_FOLDER, self.data_name)
        print(self.data_folder)
        print(self.workload_folder)

        for subdir, dirs, files in os.walk(self.data_folder):
            for filename in files:

                print(filename)
                with open(self.data_folder + "/" + filename, "r") as f:
                    readCSV = csv.reader(f, delimiter=',')
                    csv.field_size_limit(100000000)
                    sents = []
                    for idx, (author, published, title, text, language, site_url, main_img_url, type, label, title_without_stopwords, text_without_stopwords, hasImage) in tqdm(enumerate(readCSV)):
                        if label.strip() == 'Real':
                            resultdir = self.workload_folder + "/Real"
#                            print("realdir=",resultdir)
                            file_name_prefix = "Real_"
                        elif label.strip() == 'Fake':
                            resultdir = self.workload_folder + "/Fake"
 #                           print("fakedir=",resultdir)
                            file_name_prefix = "Fake_"
                        else:
                            file_name_prefix = label
                            resultdir = self.workload_folder
  #                          print("label=",label)

                        # print ([idx, title, text,subject,date])
                        newFileName = file_name_prefix + str(idx) + ".txt"
                        newFileNameWithPath = os.path.join(resultdir,newFileName)
                        textContent = dict()
                        textContent["title"] = title
                        textContent["text"] = text
                        if os.path.exists(newFileNameWithPath):
                            os.remove(newFileNameWithPath)

                        newTestFile = open(newFileNameWithPath, "x")
                        newTestFile.write(str(textContent))
                        newTestFile.close()


class Kaggle_Jruvika(object):

    def __init__(self):

        self.data_name = 'kaggle_jruvika'
        self.data_folder = "{}/{}/input".format(DATA_FOLDER, self.data_name)
        self.workload_folder = "{}/{}/workload".format(DATA_FOLDER, self.data_name)
        print(self.data_folder)
        print(self.workload_folder)

        for subdir, dirs, files in os.walk(self.data_folder):
            for filename in files:

                print(filename)
                with open(self.data_folder + "/" + filename, "r") as f:
                    readCSV = csv.reader(f, delimiter=',')
                    csv.field_size_limit(100000000)
                    sents = []
                    for idx, (URLs,Headline,Body,Label) in tqdm(enumerate(readCSV)):
                        if Label.strip() == '1':
                            resultdir = self.workload_folder + "/Real"
#                            print("realdir=",resultdir)
                            file_name_prefix = "Real_"
                        elif Label.strip() == '0':
                            resultdir = self.workload_folder + "/Fake"
 #                           print("fakedir=",resultdir)
                            file_name_prefix = "Fake_"
                        else:
                            file_name_prefix = Label
                            resultdir = self.workload_folder
  #                          print("label=",label)

                        # print ([idx, title, text,subject,date])
                        newFileName = file_name_prefix + str(idx) + ".txt"
                        newFileNameWithPath = os.path.join(resultdir,newFileName)
                        textContent = dict()
                        textContent["title"] = Headline
                        textContent["text"] = Body
                        if os.path.exists(newFileNameWithPath):
                            os.remove(newFileNameWithPath)

                        newTestFile = open(newFileNameWithPath, "x")
                        newTestFile.write(str(textContent))
                        newTestFile.close()

class Kaggle_Mohamadalhasan(object):

    def __init__(self):

        self.data_name = 'kaggle_mohamadalhasan'
        self.data_folder = "{}/{}/input".format(DATA_FOLDER, self.data_name)
        self.workload_folder = "{}/{}/workload".format(DATA_FOLDER, self.data_name)
        print(self.data_folder)
        print(self.workload_folder)

        for subdir, dirs, files in os.walk(self.data_folder):
            for filename in files:

                print(filename)
                with open(self.data_folder + "/" + filename, "r",errors='ignore') as f:
                    readCSV = csv.reader(f, delimiter=',')
                    csv.field_size_limit(100000000)
                    sents = []
                    for idx, (unit_id,article_title,article_content,source,date,location,labels) in tqdm(enumerate(readCSV)):
                        if labels.strip() == '1':
                            resultdir = self.workload_folder + "/Real"
#                            print("realdir=",resultdir)
                            file_name_prefix = "Real_"
                        elif labels.strip() == '0':
                            resultdir = self.workload_folder + "/Fake"
 #                           print("fakedir=",resultdir)
                            file_name_prefix = "Fake_"
                        else:
                            file_name_prefix = labels
                            resultdir = self.workload_folder
  #                          print("label=",label)

                        # print ([idx, title, text,subject,date])
                        newFileName = file_name_prefix + str(idx) + ".txt"
                        newFileNameWithPath = os.path.join(resultdir,newFileName)
                        textContent = dict()
                        textContent["title"] = article_title
                        textContent["text"] = article_content
                        if os.path.exists(newFileNameWithPath):
                            os.remove(newFileNameWithPath)

                        newTestFile = open(newFileNameWithPath, "x")
                        newTestFile.write(str(textContent))
                        newTestFile.close()


def load_datasets(names):

    datasets = []

    if 'FakeNewsNet' in names:
        FakeNewsNet()
    if 'ISOT_FakeNewsDataset' in names:
        ISOT_FakeNewsDataset()
    if 'kaggle_ruchi7798' in names:
        Kaggle_Ruchi7798()
    if 'kaggle_jruvika' in names:
        Kaggle_Jruvika()
    if 'kaggle_mohamadalhasan' in names:
        Kaggle_Mohamadalhasan()

    #return datasets


if __name__ == "__main__":

    load_data = False
    nltk_prep = True

    names = [
        #'FakeNewsNet',
        #'kaggle_ruchi7798',
        #'kaggle_jruvika',
        #'kaggle_mohamadalhasan',
        'ISOT_FakeNewsDataset'

    ]

    for name in names:
        print("name: {}".format(name))
        if load_data:
            load_datasets(name)
        if nltk_prep:
            nltk_preparation(name)

