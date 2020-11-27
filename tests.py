'''import csv
    with open('file.tsv', 'r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
        for item in reader:
            ...'''

import pandas as pd
import numpy as np

'''
df = pd.read_csv('C:\\Users\\Steven_Verkest\\Documents\\INF_STUDY\\BECODE\\PROJECT_FAKTION\\kleister-charity-master\\train\\expected-original.tsv',delimiter='\t', header=None)
print(df.head())
print(df.shape)
'''

'''
df = pd.read_csv('C:\\Users\\Steven_Verkest\\Documents\\INF_STUDY\\BECODE\\PROJECT_FAKTION\\kleister-charity-master\\train\\expected.tsv',delimiter='\t', header=None)
print(df.head())
print(df.shape)
'''

#GOAL 1 -> vectorize,transform, cluster similar pdf-files together (same company,with same labels -> easier for labeler to label)
#GOAL 2 -> automize clustering,

#row 1-2-3-4 contains 4 different way data is extracted out of the pdf files
df = pd.read_csv('C:\\Users\\Steven_Verkest\\Documents\\INF_STUDY\\BECODE\\PROJECT_FAKTION\\kleister-charity-master\\train\\in.tsv',delimiter='\t', header=None)
#print(df.head())
df.dropna()
print(df.shape)
print(df.columns)

#X_train = df[2]
#X_train = df[3]
#X_train = df[4]
X_train = df[5]

print('##################################################################################')
#print(df[0])  #PDF FILE NAME
print('##################################################################################')
#print(df[1]) #ADRESS / NUMBER / ...
print('##################################################################################')
#print(df[2]) # TEXT
print('##################################################################################')
#print(df[3])
print('##################################################################################')
#print(df[4])
print('##################################################################################')

##########################################################################################################
##########################################################################################################
##########################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
import re #regular expressions
from bs4 import BeautifulSoup
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',1000)
pd.set_option('display.max_colwidth',150)


# Text preprocessing

def text_preprocessing(text, language, minWordSize):
    # remove html
    text_no_html = BeautifulSoup(str(text), "html.parser").get_text()

    # remove non-letters
    text_alpha_chars = re.sub("[^a-zA-Z']", " ", str(text_no_html))

    # convert to lower-case
    text_lower = text_alpha_chars.lower()

    # remove stop words
    stops = set(stopwords.words(language))
    text_no_stop_words = ' '

    for w in text_lower.split():
        if w not in stops:
            text_no_stop_words = text_no_stop_words + w + ' '

    # do stemming
    text_stemmer = ' '
    stemmer = SnowballStemmer(language)
    for w in text_no_stop_words.split():
        text_stemmer = text_stemmer + stemmer.stem(w) + ' '

    # remove short words
    text_no_short_words = ' '
    for w in text_stemmer.split():
        if len(w) >= minWordSize:
            text_no_short_words = text_no_short_words + w + ' '
    #print(text_no_short_words)
    return text_no_short_words

###################################################################################
# Convert training and test set to bag of words
language = 'english'
minWordLength = 2

for i in range(len(X_train)):
    X_train[i] = text_preprocessing(X_train[i], language, minWordLength)

#for i in range(len(X_test)):
#    X_test[i] = text_preprocessing(X_test[i], language, minWordLength)


# Make sparse features vectors
# Bag of words

count_vect = CountVectorizer()
X_train_bag_of_words = count_vect.fit(X_train)
X_train_bag_of_words = count_vect.transform(X_train)
#X_test_bag_of_words = count_vect.transform(X_test)

print('#########################')
print('X_train_bag_of_words')
print(X_train_bag_of_words)
print('#########################')


tfidf_transformer = TfidfTransformer()
tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_bag_of_words)
X_train_tf = tf_transformer.transform(X_train_bag_of_words)
#X_test_tf = tf_transformer.transform(X_test_bag_of_words)

print(X_train_tf)

############################################################################################################
print('############################################################################################################')

#CLUSTERING
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans

'''
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_train_tf)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
'''

kmeans = KMeans(n_clusters=9, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X_train_tf)

x_trans = kmeans.fit_transform(X_train_tf) #plot
'''
import matplotlib.pyplot as plt
plt.scatter(x_trans[:0],x_trans[:,1], c=pred_y,cmap='viridis')
print(pred_y) #Index of the cluster each sample belongs to.'''

# SAVE pred_y into dataframe
df_clusters = pd.DataFrame(pred_y, columns=["cluster"])
print(df_clusters.head(10))
dataframe = df_clusters.to_csv('df_clusters.csv', header = True)

#plt.scatter(X_train[:,0], X_train[:,1])
#plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
#plt.show()

#############################################################################################
'''
#DBSCAN

import matplotlib
from sklearn.cluster import DBSCAN
dbscan=DBSCAN()
dbscan = dbscan.fit(X_train_tf)
print(dbscan)


dbscan_opt=DBSCAN(eps=30,min_samples=6)
dbscan_min  = dbscan_opt.fit(X_train_tf)
print(dbscan_min)'''

##########################################################################################
'''
# HIERARCHICAL CLUSTERING
from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(n_clusters=8, affinity='euclidean')
hier_clust = model.fit(X_train_tf.toarray())
print(hier_clust)'''

###########################################################################################
'''
# NEAREST NEIGHBOUR
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(X_train_tf)
distances, indices = nbrs.kneighbors(X_train_tf)

# Plotting K-distance Graph
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.figure(figsize=(20,10))
plt.plot(distances)
plt.title('K-distance Graph',fontsize=20)
plt.xlabel('Data Points sorted by distance',fontsize=14)
plt.ylabel('Epsilon',fontsize=14)
plt.show()
'''

###################################################################################################
'''
df_cluster = pd.DataFrame()
df_cluster['bow_ie'] = bow_ie_list
df_cluster['bow_ns'] = bow_ns_list
df_cluster['bow_tf'] = bow_tf_list
df_cluster['bow_pj'] = bow_pj_list

df_cluster['tf_ie'] = tf_ie_list
df_cluster['tf_ns'] = tf_ns_list
df_cluster['tf_tf'] = tf_tf_list
df_cluster['tf_pj'] = tf_pj_list

mbti = mbti[:10]

df_cluster['IE'] = mbti['posts'].swifter.apply(lambda x: mbtipredict_IE(x))
df_cluster['NS'] = mbti['posts'].swifter.apply(lambda x: mbtipredict_NS(x))
df_cluster['TF'] = mbti['posts'].swifter.apply(lambda x: mbtipredict_TF(x))
df_cluster['PJ'] = mbti['posts'].swifter.apply(lambda x: mbtipredict_PJ(x))
'''








