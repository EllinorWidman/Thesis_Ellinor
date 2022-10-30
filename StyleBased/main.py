import os, sys, json
import argparse, datetime, time
import random, numpy as np
from trainer import Trainer
from timeit import default_timer as timer
from News_File_Fetcher import get_news_files
from NLTK_Preparation import nltk_preparation
from Classifier import classifier
'''
node_type:
    '3 represents three types: Document&Entity&Topic; \n'
    '2 represents two types: Document&Entiy; \n'
    '1 represents two types: Document&Topic; \n'
    '0 represents only one type: Document. '
'''
# CUDA_VISIBLE_DEVICES_DICT = {0: '4',    1: '3',     2: '4',     3: '5'}
# MEMORY_DICT =               {0: 4000,   1: 9500,    2: 7600,    3: 8000}

def main():
    get_news_files()
    nltk_preparation()
    classifier()

if __name__ == '__main__':
    main()

