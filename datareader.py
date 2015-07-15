# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 10:29:13 2015

@author: gsubramanian
"""
from sklearn.feature_extraction import DictVectorizer
import os
import numpy as np

train_data = os.getcwd() +   '\\data\\train.csv'
test_data  = os.getcwd() +   '\\data\\test.csv'
submission_file = os.getcwd() +   '\\data\\submit.csv'
model_path = os.getcwd() + "\\model\\"
model_file = "tree_stump_ada.pkl"




def convert_dict(entries):
    heading = entries[0]
    data_dict = []
    for i, entry in enumerate(entries[1:]):
        dict_entry = {}
        contents = entry
        for j, cell in enumerate(contents):
            if cell.isdigit():
                dict_entry[heading[j]] = float(cell)
            else:
                dict_entry[heading[j]] = cell
        data_dict.append(dict_entry)
    return data_dict

def get_test_data(dict_vectorizer):
    ids = []
    entries = []
    with open(test_data) as f:
        line_no = 0
        for line in f:
            line_no+=1
            contents =line.strip().split(",")
            if line_no > 1:
                ids.append(contents[0])
            entries.append(contents[1:])
    entries_dict = convert_dict(entries)
    x_test = dict_vectorizer.transform(entries_dict)
    return x_test.toarray(), ids
            
    
def get_data():
    entries = []
    y = []
    line_no = 1
    with open(train_data) as f:
        for line in f:
            contents = line.strip().split(",")
            if line_no != 1:
                y.append(contents[1])
            line_no+=1
            entries.append(contents[2:])
    return get_np_vectors(convert_dict(entries), y)

    
def get_np_vectors(data_dict, y=None):
    return_y = None
    if y != None:
        return_y = np.asarray(y, dtype=float)
    dictvec = DictVectorizer()
    x = dictvec.fit_transform(data_dict)
    x = x.toarray()
    return x, return_y, dictvec.feature_names_, dictvec
