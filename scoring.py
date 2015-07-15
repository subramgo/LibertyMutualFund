# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 10:28:13 2015

@author: gsubramanian
"""
import numpy as np
def gini(solution, submission):                                                 
    df = sorted(zip(solution, submission),    
            key=lambda x: x[1], reverse=True)
    random = [float(i+1)/float(len(df)) for i in range(len(df))]                
    totalPos = np.sum([x[0] for x in df])                                       
    cumPosFound = np.cumsum([x[0] for x in df])                                     
    Lorentz = [float(x)/totalPos for x in cumPosFound]                          
    Gini = [l - r for l, r in zip(Lorentz, random)]                             
    return np.sum(Gini)                                                         


def normalized_gini(solution, submission):                                      
    normalized_gini = gini(solution, submission)/gini(solution, solution)       
    return normalized_gini 
