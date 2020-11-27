import pandas as pd
from sklearn.cluster import KMeans
import nltk
nltk.download('stopwords')
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from bs4 import BeautifulSoup
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

#row 2-3-4-5 of dataframe contains 4 different ways data is extracted out of the pdf files
df = pd.read_csv('C:\\Users\\Steven_Verkest\\Documents\\INF_STUDY\\BECODE\\PROJECT_FAKTION\\kleister-charity-master\\train\\in.tsv',delimiter='\t', header=None)
#print(df.head())
df.dropna()
print(df.shape)
print(df.columns)

#X_train = df[2]
#X_train = df[3]
#X_train = df[4]
X_train = df[5]


##########################################################################################################

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

language = 'english'
minWordLength = 2

for i in range(len(X_train)):
    X_train[i] = text_preprocessing(X_train[i], language, minWordLength)

df['preprocessed_text'] = X_train

print(df.head())
print(df.columns)
###############################################################################################################

#TF IDF

tfidf_vectorizer=TfidfVectorizer(use_idf=True)
x = tfidf_vectorizer=tfidf_vectorizer.fit_transform(X_train)
print(x)
print(x.shape)
#print(x.get_feature_names())


######################################################################

#trying to find the optimal amount of clusters with the elbow method

kmeans = KMeans(n_clusters=9, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(x)

x_trans = kmeans.fit_transform(x) #plot
#plt.scatter(x_trans[:0],x_trans[:,1], c=pred_y,cmap='viridis')
#print(pred_y) #Index of the cluster each sample belongs to.

# SAVE pred_y into dataframe
df_idf_clusters = pd.DataFrame(pred_y, columns=["cluster"])
print(df_idf_clusters.head(10))
dataframe = df_idf_clusters.to_csv('df_idf_clusters.csv', header = True)


#############################################################################################################################
##############################################################################################################################
############################################################################################################################


df = pd.read_csv('df_idf_clusters.csv')
#print(df['cluster'].value_counts())

############################################
#add charity numbers to the dataframe

df_labels = pd.read_csv('C:\\Users\\Steven_Verkest\\Documents\\INF_STUDY\\BECODE\\PROJECT_FAKTION\\kleister-charity-master\\train\\expected.tsv',sep='\t')

#search for the label numbers in the 
charity_numbers = []
for element in df_labels['number']:
    regex="charity_number=([0-9]){6}"
    x = re.findall(regex,element)
    charity_numbers.append(x)

#print(charity_numbers[])

df['charity_number'] = charity_numbers

###################################################


#make index column in df
index_list = []
for i in range(0,1729):
    index_list.append(i)
df['index'] = index_list
#print(df.head())

'''
1    551
2    428
0    349
4    127
7     76
5     76
6     48
8     45
3     29
'''

#########################################################################################
#########################################################################################

booleans = []
for elements in df['cluster']:
    if elements == 1:
        booleans.append(True)
    if elements != 1:
        booleans.append(False)

list_1 = pd.Series(booleans)
#print(list_1)

cluster_1 = df[booleans]
print(cluster_1.head(10))
print(cluster_1.shape)

index_cluster_1 = []
for element in cluster_1['index']:
    index_cluster_1.append(element)
print(index_cluster_1)

print(cluster_1['charity_number'].value_counts())
''''
[7]    69
[2]    65
[5]    60
[9]    59
[4]    56
[6]    54
[0]    53
[8]    53
[1]    47
[3]    35
'''

###################################

booleans = []
for elements in df['cluster']:
    if elements == 2:
        booleans.append(True)
    if elements != 2:
        booleans.append(False)

list_2 = pd.Series(booleans)

cluster_2 = df[booleans]
print(cluster_2.head(10))
print(cluster_2.shape)

index_cluster_2 = []
for element in cluster_2['index']:
    index_cluster_2.append(element)
print(index_cluster_2)

print(cluster_2['charity_number'].value_counts())

###################################

booleans = []
for elements in df['cluster']:
    if elements == 3:
        booleans.append(True)
    if elements != 3:
        booleans.append(False)

list_3 = pd.Series(booleans)

cluster_3 = df[booleans]
print(cluster_3.head(10))
print(cluster_3.shape)

index_cluster_3 = []
for element in cluster_3['index']:
    index_cluster_3.append(element)
print(index_cluster_3)

print(cluster_3['charity_number'].value_counts())

###################################

booleans = []
for elements in df['cluster']:
    if elements == 4:
        booleans.append(True)
    if elements != 4:
        booleans.append(False)

list_4 = pd.Series(booleans)

cluster_4 = df[booleans]
print(cluster_4.head(10))
print(cluster_4.shape)

index_cluster_4 = []
for element in cluster_4['index']:
    index_cluster_4.append(element)
print(index_cluster_4)

print(cluster_4['charity_number'].value_counts())

###################################

booleans = []
for elements in df['cluster']:
    if elements == 5:
        booleans.append(True)
    if elements != 5:
        booleans.append(False)

list_5 = pd.Series(booleans)

cluster_5 = df[booleans]
print(cluster_5.head(10))
print(cluster_5.shape)

index_cluster_5 = []
for element in cluster_5['index']:
    index_cluster_5.append(element)
print(index_cluster_5)

print(cluster_5['charity_number'].value_counts())

####################################

###################################

booleans = []
for elements in df['cluster']:
    if elements == 6:
        booleans.append(True)
    if elements != 6:
        booleans.append(False)

list_6 = pd.Series(booleans)

cluster_6 = df[booleans]
print(cluster_6.head(10))
print(cluster_6.shape)

index_cluster_6 = []
for element in cluster_6['index']:
    index_cluster_6.append(element)
print(index_cluster_6)

print(cluster_6['charity_number'].value_counts())

###################################

booleans = []
for elements in df['cluster']:
    if elements == 7:
        booleans.append(True)
    if elements != 7:
        booleans.append(False)

list_7 = pd.Series(booleans)

cluster_7 = df[booleans]
print(cluster_7.head(10))
print(cluster_7.shape)

index_cluster_7 = []
for element in cluster_7['index']:
    index_cluster_7.append(element)
print(index_cluster_7)

print(cluster_1['charity_number'].value_counts())
###################################

booleans = []
for elements in df['cluster']:
    if elements == 8:
        booleans.append(True)
    if elements != 8:
        booleans.append(False)

list_8 = pd.Series(booleans)

cluster_8 = df[booleans]
print(cluster_8.head(10))
print(cluster_8.shape)

index_cluster_8 = []
for element in cluster_8['index']:
    index_cluster_8.append(element)
print(index_cluster_8)

print(cluster_1['charity_number'].value_counts())
#########################################

booleans = []
for elements in df['cluster']:
    if elements == 0:
        booleans.append(True)
    if elements != 0:
        booleans.append(False)

list_0 = pd.Series(booleans)

cluster_0 = df[booleans]
print(cluster_0.head(10))
print(cluster_0.shape)

index_cluster_0 = []
for element in cluster_0['index']:
    index_cluster_0.append(element)
print(index_cluster_0)

print(cluster_1['charity_number'].value_counts())


######################################################






