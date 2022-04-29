# 
# 
# 
# 
# 
# 


import pandas as pd
import numpy as np
import sklearn as skl
from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk import PorterStemmer
from nltk.util import ngrams
from datetime import datetime
from datetime import timedelta
import string
from wordcloud import WordCloud

import nltk
nltk.download('punkt')
nltk.download('stopwords')

today = datetime.today().strftime('%d%b%Y')



# Preprocessing helper functions

def remove_stop(text):
    stpwrds = stopwords.words('english')
    result = []
    for word in text:
        if word not in stpwrds:
            result.append(word)
    return result

def remove_punct(text):
    result = []
    for word in text:
        if word.isalpha():
            result.append(word)
    return result

def stem(text):
    stemmer = PorterStemmer()
    result = []
    for word in text:
        result.append(stemmer.stem(word))
    return result

def add_ngrams(word_list, max_n=3):
    grams = []
    for n in range(2,max_n+1):
        grams += [' '.join(gram) for gram in ngrams(word_list, n=n)]
    return word_list + grams


# Preprocessing funtion

def preprocess(df, lower=True, token=True, rempunct=True, dostem=True, ngrams=False, remstops=True, max_n=3):
    if lower:
        df['Text'] = df['Text'].str.lower()
        print('Converted to lowercase!\n')
    if token:
        df['Text'] = df['Text'].apply(word_tokenize)
        print('Tokenized articles!\n')
    if rempunct:
        df['Text'] = df['Text'].apply(remove_punct)
        print('Removed punctuation!\n')
    if dostem:
        df['Text'] = df['Text'].apply(stem)
        print('Stemmed words!\n')
    if ngrams:
        df['Text'] = df['Text'].apply(lambda x: add_ngrams(x,max_n=max_n))
        print('Added 2 to '+str(max_n)+'-grams!\n')
    if remstops:
        df['Text'] = df['Text'].apply(remove_stop)
        print('Removed stop words!\n')
    df.to_pickle('Data/2_Preprocessed/'+today+'.csv')


# Processing function

def process(dataIn, source=False, rollAvLen=5):
    data = dataIn
    if source:
        data = data[data['Source']==source]
    
    allWords = set([word for art in data['Text'] for word in art])
    numWords = len(allWords)
    days = list(set(data['Date']))
    days.sort()
    df = pd.DataFrame(columns=days, index=allWords).fillna(0)

    # calculate daily frequencies
    for day in days:
        dailyWords = [word for art in data['Text'][data['Date']==day] for word in art]
        uniq = list(set(dailyWords))
        for i, word in enumerate(uniq):
            df.loc[word, day] = dailyWords.count(word)
        df.loc[uniq, day] /= sum(df[day])
    print('Calculated daily frequencies!\n')

    # calculate rolling averages
    for i, day in enumerate(days):
        first = max(i-rollAvLen, 0)
        rollingAvDays = days[first:i+1]
        df[day] = df[rollingAvDays].apply(np.mean, axis=1)
    print('Calculated '+str(rollAvLen)+' day rolling averages!\n')

    # normalize
    df = df.apply(lambda x: x/max(x), axis=1)
    print('Normalized frequencies!\n')
    df.to_csv('Data/3_Embedded/'+today+'.csv')


# Clustering 

def cluster(df, percent=0, max_n=3, rollAvLen=2, opticsParams=(5, 0.00001)):
    indexer = df.apply(sum, axis=1)>=np.percentile(df.apply(sum, axis=1), percent)
    df_mostfre = df[indexer]

    df_mostfre['labels'] = OPTICS(min_samples=opticsParams[0], xi=opticsParams[1]).fit_predict(df_mostfre)
    print('Done clustering!\n')
    df_mostfre.to_csv('Data/4_Clustered/'+today+'.csv')
    return df_mostfre


# Helper functions

def find_clusters(df, word):
    print('Found', str(len(set(df['labels']))-1),'clusters\n')
    for i in set(df['labels']):
        words = ', '.join(list(df[df['labels']==i].index))
        if word in words:
            print(i, words, '\n')


# Analysis Functions

def plot_cluster(df, clstnum, topic_name=False, split_by_source=False, include_words=True, include_av=True, save=False):
    
    if not topic_name:
        topic_name = str(clstnum)

    include_words = include_words & (len(list(df[df['labels']==clstnum].index))<=15)
        
    #with plt.xkcd():
    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(15)
    wdlist = []
    

    if include_words:
        for i in df[df['labels']==clstnum].index:
            ax.plot(df.loc[i, df.columns!='labels'], linewidth=1.0, color='grey') 
        wdlist += list(df[df['labels']==clstnum].index)
    if include_av:            
        ax.plot(df.loc[df['labels']==clstnum, df.columns!='labels'].apply(np.mean), linewidth=3.0, color='black')
        a = df.loc[df['labels']==clstnum, df.columns!='labels']
        wdlist.append('Average')
    if split_by_source:
        for s in set(articles['Source']):
            df_source = process(articles, source=s, rollAvLen=2)
            indexer = [i in list(df[df['labels']==clstnum].index) for i in df_source.index]
            ax.plot(df_source.loc[indexer].apply(np.mean), linewidth=1.5)
        wdlist += list(set(articles['Source']))
    
    plt.xticks(rotation = 45)
    plt.xlabel('Date', fontsize=20)
    plt.ylabel('Relative Word Frequency', fontsize=20)
    plt.title('Frequency of Topic '+topic_name+' Words', fontsize=24)
    
    plt.legend(wdlist, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16)
    
    plt.grid()
    
    plt.savefig('Data/5_Analysis/'+today+'_WordFreqPlot.png', pad_inches=10)
        
    plt.show()

def generate_word_cloud(df, cluster, ax, wordcloud):
    d = {}
    temp = df.loc[df['labels']==cluster, df.columns!='labels'].apply(np.mean, axis=1)
    for i,x in enumerate(temp):
        d[temp.index[i]] = x
    wordcloud.generate_from_frequencies(frequencies=d)
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.set_title(' '.join([str(cluster), "av freq", str(round(np.mean(df.loc[df['labels']==cluster, df.columns!='labels'].apply(np.mean, axis=1)), 3))]))
    ax.axis('off')

def plot_all_word_clouds(df, width=15):
    total = len(set(df['labels']))-1
    fig, axs = plt.subplots(round(total/5+.5), 5)
    fig.set_figheight(width*60/15*35/60*round(total/5+.5)/18)
    fig.set_figwidth(width)
    wc = WordCloud(background_color="white")
    for i in range(total):
        generate_word_cloud(df, i, axs[int(i/5), i%5], wc)
    fig.show()


def help():

    print("Usage:\n Open a database of articles into a pandas dataframe (df) with columns 'Source', 'Date', and 'text'\n Use `clusters = NewsPipeline.cluster(df, method)`\n method should be a string such as 'ngrams', 'words' describing the type of embedding method")

