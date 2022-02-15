# standard libs

import os

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix

# viz libs
#import matplotlib.pyplot as plt

# CA libs
import prince


#########################
#
# Parameters
#                      
###########################################################################

# Foula : should we have two thresholds?
degree_threshold = 3 # nodes that follow, or are followed by, less than this number, are taken out of the network
# number of CA components to keep
n_components = 2

# PRINCE/CA computation hyper-parameters
n_iter = 10
copy = True
check_input = True
engine = 'auto'    # 'auto' 'sklearn' 'fbpca'
random_state = None

#########################
#
# Load the network data 
#                      
###########################################################################

"""

First, the user loads a directed graph. 
One column is source and the other is target (of the edges)
Nodes ids are whatever the user wants: strings, integers, internally they are converted to strings

"""

#folder = '/Users/pedroramaciotti/Proyectos/SciencesPo/WorkFiles/Programs2022/ConsolidatedTwitter2019Dataset/DataSources/'
folder = '/home/foula/linate_ca/correspondence_analysis/linate_module/data/twitter_networks/'

# <- this one is a bipartite graph (uncomment one)
# path_to_network_data = folder+'bipartite_831MPs_4424402followers.csv'   

# <- this one is a social graph subtended by folowers (uncomment one)
#path_to_network_data = os.path.join(folder, 'test_directed_incomplete_graph_header.csv')
#path_to_network_data = os.path.join(folder, 'test_directed_complete_graph_header.csv')
path_to_network_data = os.path.join(folder, 'test_bipirtite_graph_no_header.csv')

output_folder = 'ca_results/'

# check network file exists
if not os.path.isfile(path_to_network_data):
    raise ValueError('Network file does not exist.')

# handles files have or without header (which should be the default input?)
header_df = pd.read_csv(path_to_network_data, nrows = 0)
column_no = len(header_df.columns)
if column_no < 2:
    raise ValueError('Network file has to have at least two columns.')

input_df = None
if 'source' in header_df.columns:
    if 'target' not in header_df.columns:
        raise ValueError('Network file has to have a \'target\' column.')
    input_df = pd.read_csv(path_to_network_data, dtype = {'source':str, 'target':str})
else:
    if column_no == 2: 
        input_df = pd.read_csv(path_to_network_data, header = None, 
                dtype = {0:str,1:str}).rename(columns = {0:'source', 1:'target'})
    else: 
        input_df = pd.read_csv(path_to_network_data, header = None, 
                dtype = {0:str,1:str}).rename(columns = {0:'source', 1:'target', 2:'multiplicity'})

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
is_bipartite_ = np.intersect1d(input_df['source'], input_df['target']).size == 0
print('Bipartite graph: ', is_bipartite_)
    
# Is the graph a multigraph ?
# There are two ways of being multigraph:
# 1) because there is a third column of integers
# 2) because there are repeated lines

# checking 1
has_more_columns = True if input_df.columns.size > 2 else False
# checking 2
has_repeated_edges = True if input_df.duplicated(subset=['source','target']).sum() else False
    
# if there is a third column, it must containt integers
if has_more_columns:
    input_df['multiplicity'] = input_df['multiplicity'].astype(int) # this will fail if missing element, or if there is string
    
if has_more_columns and has_repeated_edges: #it cannot be both
    raise ValueError('There cannot be repeated edges AND a 3rd column with edge multiplicities.')

#########################
#
# Assemble the matrices to be fed to CA 
#                      
###########################################################################

ntwrk_df = input_df[['source', 'target']]
#print(ntwrk_df)
n_i, r = ntwrk_df['target'].factorize()
source_users_no_ = len(np.unique(n_i))
column_ids_ = r.values
print('Network columns: ', column_ids_)
#print(source_users_no, len(n_i), len(r))
n_j, c = ntwrk_df['source'].factorize()
assert len(n_i) == len(n_j)
target_users_no_ = len(np.unique(n_j))
row_ids_ = c.values
print('Network rows: ', row_ids_)
#print(target_users_no, len(n_j), len(c))
network_edge_no = len(n_i)
n_in_j, tups = pd.factorize(list(zip(n_j, n_i)))
ntwrk_csr = csr_matrix((np.bincount(n_in_j), tuple(zip(*tups))))
# TODO : add delayed sparse?
ntwrk_np = ntwrk_csr.toarray()
#print(ntwrk_np.shape)
#print(ntwrk_np)

#########################
#
# Compute the CA 
#                      
###########################################################################
ca_model = prince.CA(n_components = n_components, n_iter = n_iter,
        copy = copy, check_input = check_input, engine = engine, random_state = random_state)
ca_model.fit(ntwrk_np)

#########################
#
# Organize and store results according to the type of graph 
#                      
###########################################################################

eigenvalues_ = ca_model.eigenvalues_  # list
print('Eigenvalues: ', eigenvalues_)
total_inertia_ = ca_model.total_inertia_  # numpy.float64
print('Total inertia: ', total_inertia_)
explained_inertia_ = ca_model.explained_inertia_ # list
print('Explained inertia: ', explained_inertia_)

ca_row_coordinates_ = ca_model.row_coordinates(ntwrk_np) # pandas data frame
column_names = ca_row_coordinates_.columns
new_column_names = []
for c in column_names:
    new_column_names.append('ca_component_' + str(c))
ca_row_coordinates_.columns = new_column_names
ca_row_coordinates_.index = row_ids_
ca_row_coordinates_.index.name = 'source ID'
#print(ca_row_coordinates_)
ca_row_coordinates_.to_csv(os.path.join(output_folder, 'ca_row_coordinates.csv'))

ca_column_coordinates_ = ca_model.column_coordinates(ntwrk_np) # pandas data frame
column_names = ca_column_coordinates_.columns
new_column_names = []
for c in column_names:
    new_column_names.append('ca_component_' + str(c))
ca_column_coordinates_.columns = new_column_names
ca_column_coordinates_.index = column_ids_
ca_column_coordinates_.index.name = 'target ID'
#print(ca_column_coordinates_)
ca_column_coordinates_.to_csv(os.path.join(output_folder, 'ca_column_coordinates.csv'))

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
