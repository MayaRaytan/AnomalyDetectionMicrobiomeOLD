import math
import numpy as np
import pandas as pd
import matplotlib as plt
import random

class Tree_Node:
    def __init__(self, depth=0, left=None, right=None, split_att=None, split_val=None, size=0):
        self.depth = depth
        self.left = left
        self.right = right
        self.split_att = split_att
        self.split_val = split_val
        self.size = size

'''omriTree is represented by it's root (treeNode type)'''
class Forest:
    def __init__(self, number_of_trees, trees=[]):
        self.number_of_trees = number_of_trees
        self.trees = trees

    '''
        inputs: X=input data, t=number of trees, psi=sub sampling size, l=height limit
        function: Creates forest of omriTrees.
        output: an omriForest
    '''
    def fit(self, X,psi, l, t=100):
        for i in range(t):
            self.trees.append(omri_tree(X, 0, l, psi))
            self.numberOfTrees += 1

'''
    inputs: X=input data, OTUs=taxa, e=current tree height, l=height limit, psi=sub sampling size
    function:
    output: an omriTree represented by it's root (treeNode type)
'''
def omri_tree(X, OTUs, e, l, psi):
    #np.random.seed(0)
    if e >= l or len(X) <= 1:
        return Tree_Node(depth=e, size=len(X))
    else:
        taxa = X.sample(n=psi, random_state=2)
        '''Continue from Here!!!!!!!!!!!!!!!!!!!!!!!!'''


def create_data(data):
    data, OTUs = filter_data(data)
    data = relative_abundence(data)
    return data, OTUs

def filter_data(data):
    filtered_data_with_OTUs = data.loc[:, data.sum() > 3000]
    filtered_data = filtered_data_with_OTUs.iloc[:, 1:]
    OTUs = filtered_data_with_OTUs.iloc[:, :1]
    return filtered_data, OTUs

def relative_abundence(data):
    final_data = data.copy()
    for colname in data.columns:
        final_data.loc[:, colname] = data[colname] / sum(data[colname])
    return final_data

metadata = pd.read_csv("C:\Maya\CS\AnomalyDetectionMicrobiome\Human_Microbiome_Project_1_6_2017_metadata.tab",sep='\t')
data = pd.read_csv("C:\Maya\CS\AnomalyDetectionMicrobiome\Human_Microbiome_Project_1_6_2017_data.tab",sep='\t')
X, OTUs = create_data(data)
