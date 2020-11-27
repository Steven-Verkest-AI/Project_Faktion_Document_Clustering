import pandas as pd
import re

df_clusters = pd.read_csv('df_clusters.csv')
#print(df_clusters.head())
#print(df_clusters['cluster'].value_counts())

df_expected = pd.read_csv('C:\\Users\\Steven_Verkest\\Documents\\INF_STUDY\\BECODE\\PROJECT_FAKTION\\kleister-charity-master\\train\\expected.tsv',delimiter='\t',header=None)


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


booleans = []
for elements in df_clusters['cluster']:
    if elements == 1:
        booleans.append(True)
    if elements != 1:
        booleans.append(False)

list = pd.Series(booleans)

cluster_1 = df_clusters[booleans]
print(cluster_1.head(10))
print(cluster_1.shape)

