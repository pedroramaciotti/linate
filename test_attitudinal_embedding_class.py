# test AttitudinalEmbedding

from linate import AttitudinalEmbedding

import numpy as np

def custom_agg_python_func(x):
    upper = []
    for val in x:
        if val > 1.2:
            continue
        else:
            upper.append(val)
    return np.mean(upper)

def main():
    ae_model = AttitudinalEmbedding(N = None)
    #ae_model = AttitudinalEmbedding(N = 3)
   
    # load ideological embeddings at node level
    #ideological_embedding_header_names = None # no header : first column is entity (node ID), rest are dimensions
    ideological_embedding_header_names = {'entity' : 'target ID'} # must have at least a 'entity' column
    #ideological_embedding_header_names = {'entity' : 'target ID',       # can have an optional 'dimension' column
    #        'dimensions': ['latent_dimension_1', 'latent_dimension_3']}   # to choose dimensions to load
    X = ae_model.load_ideological_embedding_from_file('ideological_embedding_results/target_dimensions_auto.csv',
            ideological_embedding_header_names = ideological_embedding_header_names)
    #print(X)

    # if attitudinal_reference_data is given at the group level need to aggregate ideological embeddings accordingly
    #entity_to_group_mapping_header_names = None # no header : first column is entity (node ID), second is group name
    entity_to_group_mapping_header_names = {'group' : 'party', 'entity' : 'twitter_id'} # must have a 'group' 
                                                                                        # column and a 'entity' column
    XG = ae_model.load_entity_to_group_mapping_from_file('ideological_embedding_results/real_node_groups.csv',
            entity_to_group_mapping_header_names = entity_to_group_mapping_header_names)
    #print(XG)

    # aggregate function to use in order to create group CA coordinates
    entity_to_group_agg_fun = None # default 'mean' is employed
    #entity_to_group_agg_fun = 'min'
    #entity_to_group_agg_fun = custom_agg_python_func
    X = ae_model.convert_to_group_ideological_embedding(X, XG, entity_to_group_agg_fun = entity_to_group_agg_fun)
    #print(X)

    # load attitudinal reference data
    #attitudinal_reference_data_header_names = None # no header : first is group name, rest is dimensions in attitudinal space
    #attitudinal_reference_data_header_names = {'entity' : 'party'} # must have a 'entity' column and an optional 'dimensions' 
                                                                    # column. 'entity' refers to group when group level 
                                                                    # attitudinal reference data is given and to node when 
                                                                    # node level attitudinal reference data is given
    attitudinal_reference_data_header_names = {'entity' : 'party',
            'dimensions' : ['ches_eu_position', 'ches_eu_foreign', 'ches_protectionism']}
    # group level
    Y = ae_model.load_attitudinal_referential_coordinates_from_file('/home/foula/FoulaDatasetAttitudinalEmbedding/groups_attitudinal_coordinates.csv', attitudinal_reference_data_header_names = attitudinal_reference_data_header_names)
    #
    # node level
    #Y = ae_model.load_attitudinal_referential_coordinates_from_file('/home/foula/correspondence_analysis/linate_module/ideological_embedding_results/generated_node_attitudinal_coordinates.csv', attitudinal_reference_data_header_names = attitudinal_reference_data_header_names)
    #print(Y)

    # Fitting
    ae_model.fit(X, Y)
    print('Number of considered idelogocal dimensions', ae_model.employed_N_)
    #print('Transformation parameters', ae_model.T_tilda_aff_np_)

    ae_model.save_transformation_parameters('ideological_embedding_results/affine_transformation.csv')

    # apply transformation to create attitudinal embeddings
    ideological_embedding_header_names = {'entity' : 'target ID'} # must have at least a 'entity' column
    X_MPs = ae_model.load_ideological_embedding_from_file('ideological_embedding_results/target_dimensions_auto.csv',
            ideological_embedding_header_names = ideological_embedding_header_names)
    #print(X_MPs)

    Y_MPs = ae_model.transform(X_MPs)
    print(Y_MPs)

if __name__ == "__main__":
    main()
