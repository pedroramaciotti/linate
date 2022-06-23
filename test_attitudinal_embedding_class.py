# test AttitudinalEmbedding

from linate import AttitudinalEmbedding

import numpy as np

import configparser

import sys

def custom_agg_python_func(x):
    upper = []
    for val in x:
        if val > 1.2:
            continue
        else:
            upper.append(val)
    return np.mean(upper)

def main():

    params = configparser.ConfigParser()  # read parameters from file
    params.read(sys.argv[1])

    N = params['attitudinal_embedding']['N']
    if N == 'None':
        N = None
    else:
        N = int(N)
    ae_model = AttitudinalEmbedding(N = N)
   
    # load ideological embeddings at node level
    ideological_embedding_header_names = None # no header : first column is entity (node ID), rest are dimensions
    # must have an 'entity' column and an optional dimensions column
    if 'ideological_dimensions_entity' in params['attitudinal_embedding'].keys():
        ideological_embedding_header_names = {'entity' : params['attitudinal_embedding']['ideological_dimensions_entity']}
    X = ae_model.load_ideological_embedding_from_file(params['attitudinal_embedding']['ideological_dimensions_file'],
            ideological_embedding_header_names = ideological_embedding_header_names)
    X['entity'] = X['entity'].astype(float)  # TODO : take out
    X['entity'] = X['entity'].astype(int)
    X['entity'] = X['entity'].astype(str)
    if 'ideological_dimensions' in params['attitudinal_embedding'].keys():
        X = X[['entity'] + params['attitudinal_embedding']['ideological_dimensions'].split(',')]
    print(X)

    # if attitudinal_reference_data is given at the group level need to aggregate ideological embeddings accordingly
    entity_to_group_mapping_header_names = None # no header : first column is entity (node ID), second is group name
    # must have a 'group' column and a 'entity' column
    if 'entity_group_group' in params['attitudinal_embedding'].keys():
        entity_to_group_mapping_header_names = {'group' : params['attitudinal_embedding']['entity_group_group'],
                'entity' : params['attitudinal_embedding']['entity_group_entity']}
    XG = ae_model.load_entity_to_group_mapping_from_file(params['attitudinal_embedding']['entity_group_file'],
            entity_to_group_mapping_header_names = entity_to_group_mapping_header_names)

    # aggregate function to use in order to create group CA coordinates
    entity_to_group_agg_fun = None # default 'mean' is employed
    #entity_to_group_agg_fun = 'min'
    #entity_to_group_agg_fun = custom_agg_python_func
    X = ae_model.convert_to_group_ideological_embedding(X, XG, entity_to_group_agg_fun = entity_to_group_agg_fun)
    group_ideological = X.copy()

    # load attitudinal reference data
    attitudinal_reference_data_header_names = None # no header : first is group name, rest is dimensions in attitudinal space
    # must have a 'entity' column and an optional 'dimensions' column. 'entity' refers to group when group level
    # attitudinal reference data is given and to node when node level attitudinal reference data is given
    if 'attitudinal_referential_entity' in params['attitudinal_embedding'].keys():
        attitudinal_reference_data_header_names = {'entity': params['attitudinal_embedding']
                ['attitudinal_referential_entity']}
    if 'attitudinal_referential_dimensions' in params['attitudinal_embedding'].keys():
        attitudinal_reference_data_header_names['dimensions'] = params['attitudinal_embedding']['attitudinal_referential_dimensions'].split(',')
    # group level
    Y = ae_model.load_attitudinal_referential_coordinates_from_file(params['attitudinal_embedding']['attitudinal_referential_coordinates_file'], attitudinal_reference_data_header_names = attitudinal_reference_data_header_names)
    #
    # node level
    #Y = ae_model.load_attitudinal_referential_coordinates_from_file('/home/foula/correspondence_analysis/linate_module/ideological_embedding_results/generated_node_attitudinal_coordinates.csv', attitudinal_reference_data_header_names = attitudinal_reference_data_header_names)
    #print(Y)
    #exit(0)

    # Fitting
    ae_model.fit(X, Y)
    print('Number of considered ideological dimensions', ae_model.employed_N_)
    #print('Transformation parameters', ae_model.T_tilda_aff_np_)
    ae_model.save_transformation_parameters(params['attitudinal_embedding']['affine_transformation_file'])

    # apply transformation to create attitudinal embeddings
    ideological_embedding_header_names = None
    if 'target_ideological_entity' in params['attitudinal_embedding'].keys():
        ideological_embedding_header_names = {'entity' : params['attitudinal_embedding']['target_ideological_entity']}
    target_ideological = ae_model.load_ideological_embedding_from_file(params['attitudinal_embedding']['target_ideological_dimensions_file'],
            ideological_embedding_header_names = ideological_embedding_header_names)
    target_ideological['entity'] = target_ideological['entity'].astype(float)  # TODO : take out
    target_ideological['entity'] = target_ideological['entity'].astype(int)
    target_ideological['entity'] = target_ideological['entity'].astype(str)
    target_attitudinal = ae_model.transform(target_ideological)
    target_attitudinal.to_csv(params['attitudinal_embedding']['target_attitudinal_dimensions_file'], sep = ',', index = None)
    #
    ideological_embedding_header_names = None
    if 'source_ideological_entity' in params['attitudinal_embedding'].keys():
        ideological_embedding_header_names = {'entity' : params['attitudinal_embedding']['source_ideological_entity']}
    source_ideological = ae_model.load_ideological_embedding_from_file(params['attitudinal_embedding']['source_ideological_dimensions_file'],
            ideological_embedding_header_names = ideological_embedding_header_names)
    source_attitudinal = ae_model.transform(source_ideological)
    source_attitudinal.to_csv(params['attitudinal_embedding']['source_attitudinal_dimensions_file'], sep = ',', index = None)
    #
    group_attitudinal = ae_model.transform(group_ideological)
    group_attitudinal.to_csv(params['attitudinal_embedding']['group_attitudinal_dimensions_file'], sep = ',', index = None)

if __name__ == "__main__":
    main()
