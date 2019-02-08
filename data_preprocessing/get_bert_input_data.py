import os
import pandas as pd
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
from nltk import tokenize
from tqdm import tqdm

input_data_path = 'data/8000_cp_product.pkl'
output_data_path = 'data/bert_pretrain_input.txt'
try:
    f= open(output_data_path,"w+")
    f.close()
except:
    print('file already exists.')

input_data = pd.read_pickle(input_data_path)
headers =['name','description_text']

def convert_column_to_string():
    for i in headers:
        input_data[i] = input_data[i].astype(str)
def clean_string(sentences_list):
    sentences_list = [i.strip().replace('\n','').replace('\t','') for i in sentences_list]
    sentences_list = [i for i in sentences_list if i!='nan']
    sentences_list = [i+'.' if i[-1]!='.' else i for i in sentences_list ]
    return sentences_list

def write_to_file(clean_doc):
    one_doc = ''
    for i in clean_doc:
        one_doc = one_doc+i+'\n'
    one_doc = one_doc+'\n'
    f = open(output_data_path, "a")
    f.write(one_doc)

convert_column_to_string()
for i in tqdm(range(len(input_data))):
    one_doc_data=[]
    for j in headers:
        one_doc_data = one_doc_data + tokenize.sent_tokenize(input_data[j][i])
    clean_doc_data = clean_string(one_doc_data)
    write_to_file(clean_doc_data)
