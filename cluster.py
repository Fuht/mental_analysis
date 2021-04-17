# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 18:56:38 2021

@author:  F H T
"""
import numpy as np
import pandas as pd
import gensim
import jieba
from gensim.models.doc2vec import Doc2Vec
from sklearn.cluster import KMeans 
from wordcloud import WordCloud
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import jieba.analyse
TaggededDocument = gensim.models.doc2vec.TaggedDocument
import os
def open_file(path,k=3):
    data = pd.read_csv(path).iloc[:,1:]
    data = data.dropna(subset = ['text'])
    data.index = np.arange(len(data))
    if os.path.exists('res_title_news_vector.txt'):
        os.remove('res_title_news_vector.txt')
        print('update')
    with open('stopwords.txt',"r",encoding = 'utf-8') as f:    #设置文件对象
        stopword = f.read()    #可以是随便对文件的操作

        b = []
        for line in range(len(data['text'])):
            words = jieba.cut(data['text'][line])
            try:                         
                a = ''
                for word in words:
                # print(word)
                    if word not in stopword:
                        a = a + word + " "
                document = TaggededDocument(a,tags = [line])
                b.append(document)
#                        f.write(word + " ")
#                f.write('\n')
            except:
                print (word)

    model_dm = Doc2Vec(b,min_count=1, window = 3, vector_size = 300, sample=1e-3, negative=5, workers=4)
    model_dm.train(b, total_examples=model_dm.corpus_count, epochs=100)    
    try:
        out = open('res_title_news_vector.txt', 'w')
        for idx, docvec in enumerate(model_dm.docvecs):        
            for value in docvec:
                out.write(str(value) + ' ')
            out.write('\n')     
    except:
        pass
        
    out.close()

           

    f = open('res_title_news_vector.txt',"r")   
    lines = f.readlines()      #读取全部内容 ，并以列表方式返回  
    corpus = []  
    for line in lines:   
        t = line.strip().split(' ')
        corpus.append(t)  


    kmean_model = KMeans(n_clusters=k)
    kmean_model.fit(corpus)
    labels= kmean_model.predict(corpus)
    cluster_centers = kmean_model.cluster_centers_
    data['label'] = labels
    return cluster_centers,data


path = 'data_final.csv'
clus,data = open_file(path,5)

s = WordCloud(font_path='c:\windows\fonts\simsun.ttc',
              background_color='white',
#              mask=plt.imread('F:\\QD\\研究生就读期间心理状态分析\\1.jpg')
              )

def cloud(data):
    with open('stopwords.txt',"r",encoding = 'utf-8') as f:    #设置文件对象
        stopword = f.read()
    stopword = stopword + '大连理工'+'展开'+'大连理工大学'+'研究生'+'全文'+'展开'+'没有'+'觉得'+\
      '时候'+'看到'+'这个'+'我们' 
    gjc = pd.DataFrame()
    c = []
    for i in np.unique(data['label']):
        
        data_class = data[data['label'] == i]
        print(len(data_class))
        a = ''
        words = []
        for j in range(len(data_class)):            
            text = data_class.iloc[j,0]
#            word_list = jieba.cut(text,cut_all=False)
            words = jieba.cut(text)
            b = ''
            for word in words:
            # print(word)
                if word not in stopword:
                    b = b + word + " "   
            a = a + b
        c.append(a)
        tags = jieba.analyse.extract_tags(a, topK=60, withWeight=True)
        d = pd.DataFrame(tags)
        gjc = pd.concat([gjc,d],axis =1)
        s.generate_from_frequencies(dict(tags))
        plt.imshow(s)
        plt.axis("off")
        plt.show()
    return gjc
gjc = cloud(data)
print(gjc)
#tags = jieba.analyse.extract_tags(a, topK=80, withWeight=True)        
#        print (i)










