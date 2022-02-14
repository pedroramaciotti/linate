# standard libs
import numpy as np
import pandas as pd

# viz libs
import matplotlib.pyplot as plt

# CA libs
import prince


#########################
#
# Parameters
#                      
###########################################################################

degree_threshold = 3 # nodes that follow, or are followed by, less than this number, are taken out of the network


#########################
#
# Load the network data 
#                      
###########################################################################

"""

First, the user loads a directed graph. 
One column is source and the other is target (of the edges)
Nodes ids are whatever the user wants: strings, integers

"""

folder = '/Users/pedroramaciotti/Proyectos/SciencesPo/WorkFiles/Programs2022/ConsolidatedTwitter2019Dataset/DataSources/'

# <- this one is a bipartite graph (uncomment one)
# path_to_network_data = folder+'bipartite_831MPs_4424402followers.csv'   

# <- this one is a social graph subtended by folowers (uncomment one)
path_to_network_data = folder+'social_graph.csv'


# this assumes that the files have no header (which should be the default input?)
input_df = pd.read_csv(path_to_network_data,header=None,dtype={0:str,1:str}).rename(columns={0:'source',1:'target',2:'multiplicity'})


#########################
#
# Initial checks and validations 
#                      
###########################################################################

"""

Once the user loads the graph, there are several things that must be checked automatically:
    
- Is the graph regular or bipartite?  This is stored in a variable, since it is important for later
- Is the graph a multigraph (edges have multiplicity)? If so, it is stored in a flag variable, since it will be important
       - Here there is an important thing: the user should be able to load to formats of multigraph:
              - A 3-column file were the third columns are integers
              - Or just a list edges, with some edges just being repeated
- Other things?

"""

# checking if it is bipartite:
is_bipartite = np.intersect1d(input_df['source'],input_df['target']).size==0
    
# Is the graph a multigraph ?
# There are two ways of being multigraph:
# 1) because there is a third column of integers
# 2) because there are repeated lines

# checking 1
has_more_columns = True if input_df.columns.size>2 else False
# checking 2
has_repeated_edges = True if input_df.duplicated(subset=['source','target']).sum() else False
    
# if there is a third column, it must containt integers
if has_more_columns:
    input_df[2] = input_df[2].astype(int) # this will fail if missing element, or if there is string
    input_df.rename(columns={2:'multiplicity'},inplace=True)
    
if has_more_columns and has_repeated_edges: #it cannot be both
    raise ValueError('There cannot be repeated edges AND a 3rd column with edge multiplicities.')
    

# Before going forwards with the analysis, we need sanity checks:
# - All nodes have string identifiers in the first 2 columns
# - All there is a third column of integers. If not multigraph, put a third column with 1 (int)


#########################
#
# Assemble the matrices to be fed to CA 
#                      
###########################################################################

#
#
#
#
#


#########################
#
# Compute the CA 
#                      
###########################################################################




#########################
#
# Organize and store results according to the type of graph 
#                      
###########################################################################




#########################
#
# Load attitudinal reference data
#                      
###########################################################################




#########################
#
# Fit an affine transformation
#                      
###########################################################################
