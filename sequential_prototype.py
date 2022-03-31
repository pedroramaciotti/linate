# compute CA of a graph and re-embed to 

default_engines = ['sklearn', 'auto', 'fbpca']  # by default use the 'prince' code for CA computation

engine = 'fbpca'    # 'auto' 'sklearn' 'fbpca' etc
#engine = 'linate_ds'

# the user can specify the CA computation library to use
from importlib import import_module

ca_module_name = 'prince'
if engine not in default_engines:
    ca_module_name = engine

try:
    ca_module = import_module(ca_module_name)
except ModuleNotFoundError:
    raise ValueError(ca_module_name + ' module is not installed; please install and make it visible it if you want to use it')

import os

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix

# viz libs
#import matplotlib.pyplot as plt

#########################
#
# Parameters
#                      
###########################################################################

in_degree_threshold = 2 # nodes that are followed by, less than this number (in the original graph), are taken out of the network, default None
out_degree_threshold = 2 # nodes that follow, less than this number (in the original graph), are taken out of the network

# number of CA components to keep
n_components = 3

# PRINCE/CA computation hyper-parameters
n_iter = 10
copy = True
check_input = True
random_state = None

# file-reading parameters
header_names = None # default: no header
#header_names = {'source':'source', 'target':'target1'} # default = None

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
#folder = '/home/foula/linate_ca/correspondence_analysis/linate_module/data/twitter_networks/'
#folder = '/home/foula/correspondence_analysis/data/twitter_bipartite_graphs/'
folder = '/home/foula/FoulaDatasetAttitudinalEmbedding/'

# <- this one is a bipartite graph (uncomment one)
# path_to_network_data = folder+'bipartite_831MPs_4424402followers.csv'   

# <- this one is a social graph subtended by folowers (uncomment one)
#path_to_network_data = os.path.join(folder, 'test_directed_incomplete_graph_header.csv')
#path_to_network_data = os.path.join(folder, 'test_directed_complete_graph_header.csv')
#path_to_network_data = os.path.join(folder, 'test_bipirtite_graph_no_header.csv')
#path_to_network_data = os.path.join(folder, 'UnitedStates_20201027_MPs_followers_tids_test.csv')
# path_to_network_data = os.path.join(folder, 'FranceOwn_20201026_MPs_followers_tids_test.csv')
# path_to_network_data = os.path.join(folder, 'FranceOwn_20201026_MPs_followers_tids_test.csv')
#path_to_network_data = os.path.join(folder, 'Malta_20201027_MPs_followers_tids_test.csv')
path_to_network_data = os.path.join(folder, 'bipartite_831MPs_4424402followers.csv')

output_folder = 'ca_results/'
#output_folder = 'ca_results_ds/'

# check network file exists
if not os.path.isfile(path_to_network_data):
    raise ValueError('Network file does not exist.')

# handles files with or without header (which should be the default input?)
header_df = pd.read_csv(path_to_network_data, nrows = 0)
column_no = len(header_df.columns)
if column_no < 2:
    raise ValueError('Network file has to have at least two columns.')

# sanity checks in header
if header_names is not None:
    if header_names['source'] not in header_df.columns:
        raise ValueError('Network file has to have a ' + header_names['source'] + ' column.')
    if header_names['target'] not in header_df.columns:
        raise ValueError('Network file has to have a ' + header_names['target'] + ' column.')

input_df = None
if header_names is None:
    if column_no == 2: 
        input_df = pd.read_csv(path_to_network_data, header = None, 
                dtype = {0:str, 1:str}).rename(columns = {0:'source', 1:'target'})
    else: 
        input_df = pd.read_csv(path_to_network_data, header = None, 
                dtype = {0:str, 1:str}).rename(columns = {0:'source', 1:'target', 2:'multiplicity'})
else:
    input_df = pd.read_csv(path_to_network_data, dtype = {header_names['source']:str,
        header_names['target']:str}).rename(columns = {header_names['source']:'source', header_names['target']:'target'})

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

# remove NAs
input_df.dropna(subset = ['source', 'target'], inplace = True)

# in some python/pandas versions this seems as needed
input_df['source'] = input_df['source'].astype(str)
input_df['target'] = input_df['target'].astype(str)

# Is the graph a multigraph ?
# There are two ways of being multigraph:
# 1) because there is a third column of integers
# 2) because there are repeated lines

# checking 1
has_more_columns = True if input_df.columns.size > 2 else False
# checking 2
has_repeated_edges = True if input_df.duplicated(subset = ['source', 'target']).sum() > 0 else False
    
# if there is a third column, it must containt integers
if has_more_columns:
    input_df['multiplicity'] = input_df['multiplicity'].astype(int) # will fail if missing element, or cannot convert
    
if has_more_columns and has_repeated_edges: #it cannot be both
    raise ValueError('There cannot be repeated edges AND a 3rd column with edge multiplicities.')

# remove nodes with small degree
degree_per_target = None
if in_degree_threshold is not None:
    degree_per_target = input_df.groupby('target').count()

degree_per_source = None
if out_degree_threshold is not None:
    degree_per_source = input_df.groupby('source').count()

if degree_per_target is not None:
    if 'multiplicity' in degree_per_target.columns:
        degree_per_target.drop('multiplicity', axis = 1, inplace = True)
    degree_per_target = degree_per_target[degree_per_target >= in_degree_threshold].dropna().reset_index()
    degree_per_target.drop('source', axis = 1, inplace = True)
    input_df = pd.merge(input_df, degree_per_target, on = ['target'], how = 'inner')

if degree_per_source is not None:
    if 'multiplicity' in degree_per_source.columns:
        degree_per_source.drop('multiplicity', axis = 1, inplace = True)
    degree_per_source = degree_per_source[degree_per_source >= out_degree_threshold].dropna().reset_index()
    degree_per_source.drop('target', axis = 1, inplace = True)
    input_df = pd.merge(input_df, degree_per_source, on = ['source'], how = 'inner')

# checking if final network is bipartite:
is_bipartite_ = np.intersect1d(input_df['source'], input_df['target']).size == 0
print('Bipartite graph: ', is_bipartite_)

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
#print('Network columns: ', column_ids_)
#print(source_users_no, len(n_i), len(r))
n_j, c = ntwrk_df['source'].factorize()
assert len(n_i) == len(n_j)
target_users_no_ = len(np.unique(n_j))
row_ids_ = c.values
#print('Network rows: ', row_ids_)
#print(target_users_no, len(n_j), len(c))
network_edge_no = len(n_i)
n_in_j, tups = pd.factorize(list(zip(n_j, n_i)))
ntwrk_csr = csr_matrix((np.bincount(n_in_j), tuple(zip(*tups)))) # COO might be faster 

if engine in default_engines:
    ntwrk_np = ntwrk_csr.toarray()
    #print(ntwrk_np.shape)

#########################
#
# Compute the CA
#                      
###########################################################################
n_components_tmp = source_users_no_
if n_components_tmp > target_users_no_:
    n_components_tmp = target_users_no_
if n_components < 0:
    n_components = n_components_tmp
else:
    if n_components > n_components_tmp:
        n_components = n_components_tmp

print('Computing CA...')
ca_class = getattr(ca_module, 'CA')

if engine in default_engines:
    ca_model = ca_class(n_components = n_components, n_iter = n_iter,
            copy = copy, check_input = check_input, engine = engine, random_state = random_state)
    ca_model.fit(ntwrk_np)
else:
    ca_model = ca_class(n_components = n_components)
    ca_model.fit(ntwrk_csr)

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

if engine in default_engines:
    ca_row_coordinates_ = ca_model.row_coordinates(ntwrk_np) # pandas data frame
else:
    ca_row_coordinates_ = ca_model.row_coordinates() # pandas data frame
column_names = ca_row_coordinates_.columns
new_column_names = []
for c in column_names:
    new_column_names.append('ca_component_' + str(c))
ca_row_coordinates_.columns = new_column_names
ca_row_coordinates_.index = row_ids_
ca_row_coordinates_.index.name = 'source ID'
#print(ca_row_coordinates_)
ca_row_coordinates_.to_csv(os.path.join(output_folder, 'ca_row_coordinates.csv'))

if engine in default_engines:
    ca_column_coordinates_ = ca_model.column_coordinates(ntwrk_np) # pandas data frame
else:
    ca_column_coordinates_ = ca_model.column_coordinates() # pandas data frame
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

#### input/parameters
# whether the attitudinal data are given at the group or the node level
group_attitudinal_reference_data_is_given = True
#group_attitudinal_reference_data_is_given = False

path_attitudinal_reference_data = '/home/foula/FoulaDatasetAttitudinalEmbedding/groups_attitudinal_coordinates.csv'
#path_attitudinal_reference_data = '/home/foula/correspondence_analysis/linate_module/ca_results/generated_node_attitudinal_coordinates.csv'

#attitudinal_reference_data_header_names = None # no header : first is group name, rest is dimensions in attitudinal space
#attitudinal_reference_data_header_names = {'entity' : 'party'} # must have a 'entity' column and an optional 'dimensions' column
# entity refers to group when group level attitudinal reference data is given and to node when node level attitudinal reference data is given
attitudinal_reference_data_header_names = {'entity' : 'party', 
        'dimensions' : ['ches_eu_position', 'ches_eu_foreign', 'ches_protectionism']}

# for node groups
node_group_filename = 'ca_results/real_node_groups.csv'  # None if attirudinal reference data are given at the node level
#
#node_group_data_header_names = None # no header : first is node ID, second is group name
node_group_data_header_names = {'group' : 'party', 'node_id' : 'twitter_id'} # must have a 'group' column and a 'node_id' column

node_ca_coordinates_filename = 'ca_results/ca_column_coordinates.csv'

#node_ca_coordinates_header_names = None # no header : first is node ID, rest is dimensions
node_ca_coordinates_header_names = {'node_id' : 'target ID'} # must have at least a 'node_id' column

# number of latent ideological dimensions to be considered
N = None # default : P (number of groups) - 1

# aggregate function to use in order to create group CA coordinates
def foula_mean(x):
    upper = []
    for val in x:
        if val > 1.2:
            continue
        else:
            upper.append(val)
    return np.mean(upper)

group_ca_agg_fun = None # default 'mean' is employed
#group_ca_agg_fun = foula_mean
#group_ca_agg_fun = 'min'

#####

# check if attitudinal reference data file exists

if not os.path.isfile(path_attitudinal_reference_data):
    raise ValueError('Attitudinal reference data file does not exist.')

# handles files with or without header
header_df = pd.read_csv(path_attitudinal_reference_data, nrows = 0)
column_no = len(header_df.columns)
if column_no < 2:
    raise ValueError('Attitudinal reference data file has to have at least two columns.')

if attitudinal_reference_data_header_names is not None:
    if attitudinal_reference_data_header_names['entity'] not in header_df.columns:
        raise ValueError('Attitudinal reference data file has to have a ' 
                + attitudinal_reference_data_header_names['entity'] + ' column.')

# load attitudinal reference data
attitudinal_reference_data_df = None
if attitudinal_reference_data_header_names is None:
    attitudinal_reference_data_df = pd.read_csv(path_attitudinal_reference_data, 
            header = None).rename(columns = {0:'entity'})
else:
    attitudinal_reference_data_df = pd.read_csv(path_attitudinal_reference_data).rename(columns 
            = {attitudinal_reference_data_header_names['entity']:'entity'})
    if 'dimensions' in attitudinal_reference_data_header_names.keys():
        cols = attitudinal_reference_data_header_names['dimensions'].copy()
        cols.append('entity')
        attitudinal_reference_data_df = attitudinal_reference_data_df[cols]
#print(attitudinal_reference_data_df.shape)

# exclude groups with a NaN in any of the dimensions (or group)
attitudinal_reference_data_df.dropna(inplace = True)
attitudinal_reference_data_df['entity'] = attitudinal_reference_data_df['entity'].astype(str)

#print(attitudinal_reference_data_df.shape)
#print(attitudinal_reference_data_df.head())

if group_attitudinal_reference_data_is_given: # need to load node-group mapping
    # check if node group file exists
    if not os.path.isfile(node_group_filename):
        raise ValueError('Node group data file does not exist.')

    # handles node group files with or without header
    header_df = pd.read_csv(node_group_filename, nrows = 0)
    column_no = len(header_df.columns)
    if column_no < 2:
        raise ValueError('Node group data file has to have at least two columns.')

    if node_group_data_header_names is not None:
        if node_group_data_header_names['group'] not in header_df.columns:
            raise ValueError('Node group data file has to have a ' 
                    + node_group_data_header_names['group'] + ' column.')

        if node_group_data_header_names['node_id'] not in header_df.columns: 
            raise ValueError('Node group data file has to have a ' 
                    + node_group_data_header_names['node_id'] + ' column.')

        # load node group data
        node_group_data_df = None
        if node_group_data_header_names is None:
            node_group_data_df = pd.read_csv(node_group_filename, header 
                    = None).rename(columns = {0:'node_id', 1:'group'})
        else:
            node_group_data_df = pd.read_csv(node_group_filename).rename(columns = 
                    {node_group_data_header_names['group']:'group',
                        node_group_data_header_names['node_id']:'node_id'})

        # exclude rows with a NaN in any of the columns
        node_group_data_df.dropna(inplace = True)

        #print(node_group_data_df.shape)
        #print(node_group_data_df.head())

# check if node ca coordinates file exists
if not os.path.isfile(node_ca_coordinates_filename):
    raise ValueError('Node CA cordinates data file does not exist.')

# handles files with or without header
header_df = pd.read_csv(node_ca_coordinates_filename, nrows = 0)
column_no = len(header_df.columns)
if column_no < 2:
    raise ValueError('Node CA cordinates data file has to have at least two columns.')

if node_ca_coordinates_header_names is not None:
    if node_ca_coordinates_header_names['node_id'] not in header_df.columns:
        raise ValueError('Node CA cordinates data file has to have a ' 
                + node_ca_coordinates_header_names['node_id'] + ' column.')

# load node CA coordinates data
node_ca_coordinates_df = None
if node_ca_coordinates_header_names is None:
    node_ca_coordinates_df = pd.read_csv(node_ca_coordinates_filename, 
            header = None).rename(columns = {0:'node_id'})
else:
    node_ca_coordinates_df = pd.read_csv(node_ca_coordinates_filename).rename(columns = 
            {node_ca_coordinates_header_names['node_id']:'node_id'})

#print(node_ca_coordinates_df.shape)
#print(node_ca_coordinates_df.head())

entity_ca_coordinates_df = None # maintains CA coordinates at the group or node level
if group_attitudinal_reference_data_is_given: # need to load node-group mapping

    # add group information to the CA coordinates
    node_group_ca_coordinates_df = pd.merge(node_ca_coordinates_df, node_group_data_df, on = 'node_id')
    node_group_ca_coordinates_df.drop('node_id', axis = 1, inplace = True)

    # also keep only the groups that exist in the attitudinal reference data
    node_group_ca_coordinates_df['group'] = node_group_ca_coordinates_df['group'].astype(str)
    ga_merge_df = pd.merge(attitudinal_reference_data_df, node_group_ca_coordinates_df, 
            left_on = 'entity', right_on = 'group', how = 'inner')
    node_group_ca_coordinates_df = node_group_ca_coordinates_df[node_group_ca_coordinates_df['group'].
            isin(ga_merge_df.group.unique())]
    attitudinal_reference_data_df = attitudinal_reference_data_df[attitudinal_reference_data_df['entity'].
            isin(ga_merge_df.entity.unique())]

    #print(node_group_ca_coordinates_df.shape)
    #print(node_group_ca_coordinates_df.head())

    # create ca coordinates aggregates : user can define custom (columnwise) aggregate
    if group_ca_agg_fun is None:
        entity_ca_coordinates_df = node_group_ca_coordinates_df.groupby(['group']).agg('mean').reset_index()
    else:
        entity_ca_coordinates_df = node_group_ca_coordinates_df.groupby(['group']).agg(group_ca_agg_fun).reset_index()
    entity_ca_coordinates_df.rename(columns = {'group': 'entity'}, inplace = True)

else:
    node_ca_coordinates_df['node_id'] = node_ca_coordinates_df['node_id'].astype(str)
    na_merge_df = pd.merge(attitudinal_reference_data_df, node_ca_coordinates_df, 
            left_on = 'entity', right_on = 'node_id', how = 'inner')
    node_ca_coordinates_df = node_ca_coordinates_df[node_ca_coordinates_df['node_id'].
            isin(na_merge_df.node_id.unique())]
    attitudinal_reference_data_df = attitudinal_reference_data_df[attitudinal_reference_data_df['entity'].
            isin(na_merge_df.entity.unique())]
    entity_ca_coordinates_df = node_ca_coordinates_df
    entity_ca_coordinates_df.rename(columns = {'node_id': 'entity'}, inplace = True)

#print(entity_ca_coordinates_df.shape, attitudinal_reference_data_df.shape)
#print(entity_ca_coordinates_df.head(8))

#########################
#
# Fit an affine transformation
#                      
###########################################################################

# sort attitudinal reference data by group (so as to have the right mapping with ideological embeddings)
attitudinal_reference_data_df = attitudinal_reference_data_df.sort_values('entity', ascending = True)

#print(attitudinal_reference_data_df.shape)
#print(attitudinal_reference_data_df.head(8))

# and convert to Y_tilda
Y_df = attitudinal_reference_data_df.drop('entity', axis = 1, inplace = False)
Y_np = Y_df.to_numpy().T
ones_np = np.ones((Y_np.shape[1],), dtype = float)
Y_tilda_np = np.append(Y_np, [ones_np], axis= 0)
#print(Y_np.shape, Y_tilda_np.shape)

# sort ideological coordinates by group (so as to have the right mapping with attitudinal embeddings)
entity_ca_coordinates_df = entity_ca_coordinates_df.sort_values('entity', ascending = True)

#print(group_ca_coordinates_df.shape)
#print(group_ca_coordinates_df.head(8))

# convert to X_tilda
X_df = entity_ca_coordinates_df.drop('entity', axis = 1, inplace = False)
X_np = X_df.to_numpy()
if N is None:
    N = X_np.shape[0] - 1
X_np = X_np[:,:N]
X_np = X_np.T
ones_np = np.ones((X_np.shape[1],), dtype = float)
X_tilda_np = np.append(X_np, [ones_np], axis= 0)
#print(X_tilda_np.shape)

# finally compute T_tilda_aff
T_tilda_aff_np_1 = np.matmul(Y_tilda_np, X_tilda_np.T)
T_tilda_aff_np_2 = np.matmul(X_tilda_np, X_tilda_np.T)
T_tilda_aff_np_3 = np.linalg.inv(T_tilda_aff_np_2)
T_tilda_aff_np = np.matmul(T_tilda_aff_np_1, T_tilda_aff_np_3)
#print(T_tilda_aff_np.shape)
print('Affine transformation: ', T_tilda_aff_np)
np.savetxt(os.path.join(output_folder, 'affine_transformation.csv'), T_tilda_aff_np, delimiter = ",")
