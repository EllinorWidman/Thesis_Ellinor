import os, sys, json
import argparse, datetime, time
import random, numpy as np
#from trainer import Trainer
from timeit import default_timer as timer
#from News_File_Fetcher_csv import get_news_files_from_CSV
#from News_File_Fetcher import get_news_files
#from NLTK_Preparation import nltk_preparation
from Classifier import classifier

def main():


    names = [
        'FakeNewsNet',
        'ISOT_FakeNewsDataset',
        'kaggle_ruchi7798',
        'kaggle_jruvika',
        'kaggle_mohamadalhasan'
    ]

    for name in names:
        print("name: {}".format(name))
        classifier(name)


if __name__ == '__main__':
    main()

