import pandas as pd

df = pd.read_csv('df_idf_clusters.csv')
#print(df['cluster'].value_counts())

############################################
#add charity numbers to the dataframe

import re

df_labels = pd.read_csv('C:\\Users\\Steven_Verkest\\Documents\\INF_STUDY\\BECODE\\PROJECT_FAKTION\\kleister-charity-master\\train\\expected.tsv',sep='\t')

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



