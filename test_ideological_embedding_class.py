# test 'IdeologicalEmbedding' class

from linate import IdeologicalEmbedding

import numpy as np

import sys

import configparser

def my_agg_fun(a):     # user defined error aggregate function
    return np.mean(a)

def main(params_filename):
    params = configparser.ConfigParser()  # read parameters from file
    params.read(sys.argv[1])

    random_state = None

    standardize_mean = True
    in_degree_threshold = params['ideological_embedding']['in_degree_threshold']
    if in_degree_threshold == 'None':
        in_degree_threshold = None
    out_degree_threshold = params['ideological_embedding']['out_degree_threshold']
    if out_degree_threshold == 'None':
        out_degree_threshold = None
    force_bipartite = params['ideological_embedding']['force_bipartite']
    if force_bipartite == 'True':
        force_bipartite = True
    else:
        force_bipartite = False
    standardize_mean = params['ideological_embedding']['standardize_mean']
    if standardize_mean == 'True':
        standardize_mean = True
    else:
        standardize_mean = False
    standardize_std = params['ideological_embedding']['standardize_std']
    if standardize_std == 'True':
        standardize_std = True
    else:
        standardize_std = False
    ideological_embedding_model = IdeologicalEmbedding(n_latent_dimensions = int(params['ideological_embedding']
        ['n_latent_dimensions']), engine = params['ideological_embedding']['engine'],
            in_degree_threshold = in_degree_threshold, out_degree_threshold = out_degree_threshold,
            force_bipartite = force_bipartite, standardize_mean = standardize_mean,
            standardize_std = standardize_std, random_state = random_state)

    network_file_header_names = None
    if 'source' in params['ideological_embedding'].keys():
        if 'target' in params['ideological_embedding'].keys():
            network_file_header_names = {'source': params['ideological_embedding']['source'],
                    'target': params['ideological_embedding']['target']}
    X = ideological_embedding_model.load_input_from_file(params['ideological_embedding']['network_filename'],
            network_file_header_names = network_file_header_names)
    #print(X)

    # Fitting
    ideological_embedding_model.fit(X)

    # Saving to file (either scaled or non-scaled coordinates depending on user choice)
    ideological_embedding_model.save_ideological_embedding_target_latent_dimensions(params['ideological_embedding']['target_dimensions_file'])
    ideological_embedding_model.save_ideological_embedding_source_latent_dimensions(params['ideological_embedding']['source_dimensions_file'])

    # Retrieving the coordinates in program
    #target_coords = ideological_embedding_model.ideological_embedding_target_latent_dimensions_
    #source_coords = ideological_embedding_model.ideological_embedding_source_latent_dimensions_

    # Check the number of source and target nodes
    print('Number of targets', ideological_embedding_model.target_entity_no_)
    print('Number of sources', ideological_embedding_model.source_entity_no_)

    #
    print('Eigenvalues', ideological_embedding_model.eigenvalues_)
    print('Total inertia', ideological_embedding_model.total_inertia_)
    print('Explained inertia', ideological_embedding_model.explained_inertia_)

    '''
    # compute distance of predicted dimensions with given benchmark dimensions
    #benchmark_ideological_dimensions_data_header_names = None # default: first column is entity, rest of columns are dimensions
    benchmark_ideological_dimensions_data_header_names = {'entity': 'target ID'} # has to define an 'entity' column
    #benchmark_ideological_dimensions_data_header_names = {'entity': 'target ID',
    #        'dimensions': ['latent_dimension_1', 'latent_dimension_3']}  # can have an optional 'dimensions' entry
    Bnchmark_Y = ideological_embedding_model.load_benchmark_ideological_dimensions_from_file('ideological_embedding_results/target_dimensions_scaled.csv', benchmark_ideological_dimensions_data_header_names = benchmark_ideological_dimensions_data_header_names)
    #print(Bnchmark_Y)
    
    ideological_dimension_mapping = None # default: if Y is pandas dataframe first column
                                         # is entity and the rest is the ideological dimensions
    #ideological_dimension_mapping = {'X': ['latent_dimension_1', 'latent_dimension_0', 'latent_dimension_3']}
    #
    error_aggregation_fun = None # default : RMSE variant
    #error_aggregation_fun = 'MAE' # default : mean average error
    #error_aggregation_fun = my_agg_fun
    err = ideological_embedding_model.compute_latent_embedding_distance(Bnchmark_Y, use_target_ideological_embedding = True,
            ideological_dimension_mapping = ideological_dimension_mapping, error_aggregation_fun = error_aggregation_fun)
    print('Ideologial dimension prediction error', err)
    '''

if __name__ == "__main__":
    main(sys.argv[1])
