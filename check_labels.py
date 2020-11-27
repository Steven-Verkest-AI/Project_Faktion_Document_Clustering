import pandas as pd
import re

df_labels = pd.read_csv('C:\\Users\\Steven_Verkest\\Documents\\INF_STUDY\\BECODE\\PROJECT_FAKTION\\kleister-charity-master\\train\\expected.tsv',sep='\t')

charity_numbers = []
for element in df_labels['number']:
    regex="charity_number=([0-9]){6}"
    x = re.findall(regex,element)
    charity_numbers.append(x)

print(charity_numbers[0])
