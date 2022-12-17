# @file: preprocess.py
# @description: Preprocess the data return the data in the format of numpy array
# @datasets: 1.Short -  2.Medium - 3.Long - 20newsgroup 


import os
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
from nltk.corpus import wordnet
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import gensim
from gensim.utils import simple_preprocess
from gensim import corpora, models
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import datetime
import time
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from gensim import corpora
import gensim
import numpy as np
from datetime import datetime
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import re
import nltk
from nltk.tokenize import word_tokenize
from language_detector import detect_language
from sklearn.datasets import fetch_20newsgroups



import pkg_resources
from symspellpy import SymSpell, Verbosity

sym_spell = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
if sym_spell.word_count:
    pass
else:
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1) 

class Preprocess():
    def __init__(self, dataset, data_path=None, label_path=None, max_len=None):
        self.dataset = dataset
        self.data_path = data_path
        self.label_path = label_path
        self.max_len = max_len
        self.stop_words = (list(
        set(get_stop_words('en'))
        |set(get_stop_words('es'))
        |set(get_stop_words('de'))
        |set(get_stop_words('it'))
        |set(get_stop_words('ca'))
        |set(get_stop_words('pt'))
        |set(get_stop_words('pl'))
        |set(get_stop_words('da'))
        |set(get_stop_words('ru'))
        |set(get_stop_words('sv'))
        |set(get_stop_words('sk'))
        |set(get_stop_words('nl'))))
        self.stemmer = SnowballStemmer('english')
        self.lemmatizer = WordNetLemmatizer()

    def load_data(self):
        if self.dataset == 'short':
            # data = pd.read_csv(self.data_path, header=None)
            # labels = pd.read_csv(self.label_path, header=None)
            # data = data.values
            # labels = labels.values
            
            # Kritika
            folder_path = "/home/jugaad/CMSC723/723/data/bbc"
            folders = ['entertainment','business','sport','politics','tech']
            
            news=[]
            label=[]
            for folder in folders:
                new_path = os.path.join(folder_path, folder)
                for file in os.listdir(new_path):
                    with open(os.path.join(new_path,file), 'r') as file:
                        data = file.read()
                        news.append(data)
                        label.append(folder)
            data={'News':news,'category':label}
            df = pd.DataFrame(data)
            
            data = df.News
            labels = df.category
            
            # return data, labels
            print(len(data.to_list()))
            return data.to_list()
        elif self.dataset == 'medium':
            data = pd.read_csv(self.data_path, header=None)
            labels = pd.read_csv(self.label_path, header=None)
            data = data.values
            labels = labels.values
            return data, labels
        elif self.dataset == 'long':
            data = pd.read_csv(self.data_path, header=None)
            labels = pd.read_csv(self.label_path, header=None)
            data = data.values
            labels = labels.values
            return data, labels
        elif self.dataset == '20newsgroup':
            # data = pd.read_csv(self.data_path, header=None)
            # labels = pd.read_csv(self.label_path, header=None)
            # data = data.values
            # labels = labels.values
            data = fetch_20newsgroups(subset='all')['data']
            print(len(data))
            return data
        else:
            print('Invalid dataset name')

    
    def preprocess_string(self, s):
       
        s = re.sub(r'([a-z])([A-Z])', r'\1\. \2', s)  # split camel case words
        s = s.lower() # lowercase
        s = re.sub(r'&gt|&lt', ' ', s) # remove html tags
        s = re.sub(r'([a-z])\1{2,}', r'\1', s) # remove repeating characters
        s = re.sub(r'([\W+])\1{1,}', r'\1', s) # remove repeating punctuation
        s = re.sub(r'\*|\W\*|\*\W', '. ', s) # replace * with .
        s = re.sub(r'\(.*?\)', '. ', s) # replace text in brackets with .
        s = re.sub(r'\W+?\.', '.', s) # remove punctuation before .
        s = re.sub(r'(\.|\?|!)(\w)', r'\1 \2', s) # add space after .?!
        s = re.sub(r' ing ', ' ', s) # remove ing
        s = re.sub(r'product received for free[.| ]', ' ', s) # remove product received for free
        s = re.sub(r'(.{2,}?)\1{1,}', r'\1', s) # remove repeating words

        return s.strip()

    def det_language(self, text):
        try:
            return detect_language(text) in {'English','French','German','Italian','Spanish','Chinese','Japanese','Korean'}
        except:
            return 'English'
        
    def filter_punctuation(self, word_list):
        result = []
        for word in word_list:
            if word.isalpha():
                result.append(word)
        return result
    
    def filter_nouns(self, word_list):
       return [word for (word, pos) in nltk.pos_tag(word_list) if pos.startswith('NN')]
   
    def filter_typo(self, word_list):
        
        result = []
        for word in word_list:
            suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True)
            if suggestions:
                result.append(suggestions[0].term)
            else:
                pass
        return result
    
    def filter_stem(self, word_list):
        return [self.stemmer.stem(word) for word in word_list]
    
    def remnove_stopwords(self, word_list):
        return [word for word in word_list if word not in self.stop_words]
    
    def preprocess_sentence(self, sentence):
        
        result = self.preprocess_string(sentence)
        if not self.det_language(result):
            return None
        return result
    
    
    
    def preprocess_words(self,data):
        
        if not data:
            return None
        
        word_list = word_tokenize(data)
        word_list = self.filter_punctuation(word_list)
        word_list = self.filter_nouns(word_list)
        word_list = self.filter_typo(word_list)
        word_list = self.filter_stem(word_list)
        word_list = self.remnove_stopwords(word_list)
        
        return word_list
    
    def preprocess(self,sample_size=100):
        
        
        data = self.load_data()
        
        if not sample_size or sample_size > len(data):
            sample_size = len(data)
            
        print('Preprocessing {} samples'.format(sample_size))
        
        number_docs = len(data)
        
        sentences = []
        token_lists = []
        idx_sample_list = []
        
        sample = np.random.choice(number_docs, sample_size, replace=False)
        
        for i, idx in enumerate(sample):
            sentence = self.preprocess_sentence(data[idx])
            if sentence:
                sentences.append(sentence)
                token_lists.append(self.preprocess_words(sentence))
                idx_sample_list.append(idx)
                
            if i % 10000 == 0:
                print('Preprocessed {} samples'.format(i))
                
                
            
        return sentences, token_lists, idx_sample_list
        
        # data, labels = self.load_data()
        # data = [self.preprocess_sentence(sentence) for sentence in data]
        # data = [self.preprocess_words(sentence) for sentence in data]
        # data = [sentence for sentence in data if sentence]
        # labels = [label for label in labels if label]
        # return data, labels