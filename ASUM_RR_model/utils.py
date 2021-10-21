import csv
import os
import re
import pickle as pk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


def read_csv_data(path):

    reviews = []
    ratings = []

    with open(path, 'r') as f:
        reader = csv.reader(f)

        for i,row in enumerate(reader):
            doc = []
            if i>0:
                sentences = row[0].split('[')
                for st in sentences:
                    doc.append(st)
                ratings.append([int(float(row[1]))])
                reviews.append(doc)

    return reviews,ratings



def preprocessing(documents):
    '''
    对文档进行预处理，
    1.先去除掉标点符号，数字，和非英文
    2.使用Porter stemmer取词根
    3.去除stop words
    :param documents:
    :return: cleaned_documents
    '''

    '''nltk库中的Porter提取器'''
    porter_stemmer = PorterStemmer()

    '''使用nltk中的停用词'''
    stop_words = stopwords.words('english')

    cleaned_documents = []
    '''每一篇文档'''
    for d in documents:
        doc = []
        for st in d:
            cleaned_st = []
            '''每一个句子'''
            '''都转换为小写'''
            '''提取词根并且判断是否是停用词'''
            words = st.split(',')
            for word in words:
                letters_only = re.sub("[^a-zA-Z]", "", word)
                word = letters_only.lower()
                if word != '' and word not in stop_words:
                    cleaned_st.append(porter_stemmer.stem(word))

            if len(cleaned_st)!=0:
                doc.append(cleaned_st)

        cleaned_documents.append(doc)

    return cleaned_documents



def read_test_csv_data(path):

    reviews = []
    ratings = []

    with open(path, 'r') as f:
        reader = csv.reader(f)
        for i,row in enumerate(reader):
            if i>0:
                reviews.append(row[0])
                ratings.append([int(float(row[1]))])


    return reviews,ratings


def test_preprocessing(documents):
    '''
    对文档进行预处理，
    1.先去除掉标点符号，数字，和非英文
    2.使用Porter stemmer取词根
    3.去除stop words
    :param documents:
    :return: cleaned_documents
    '''

    '''nltk库中的Porter提取器'''
    porter_stemmer = PorterStemmer()

    '''使用nltk中的停用词'''
    stop_words = stopwords.words('english')

    cleaned_documents = []
    '''每一篇文档'''
    for d in documents:
        cleaned_document = []
        '''每一个句子'''
        '''都转换为小写'''
        '''提取词根并且判断是否是停用词'''
        words = d.split(',')
        for word in words:
            letters_only = re.sub("[^a-zA-Z]", "", word)
            word = letters_only.lower()
            if word != '' and word not in stop_words:
                cleaned_document.append(porter_stemmer.stem(word))

        cleaned_documents.append(cleaned_document)

    return cleaned_documents