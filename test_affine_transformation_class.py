# linate class

from linate import AffineTransformation


# Note from 18th March meeting:
# Naming: instead of AffineTransformation, it could be AttitudinalEmbedding:
# Naming: loose references to "ca", just "coordinates", as 
# instead of "load_node_ca_coordinates_from_file" use "load_node_coordinates_from_file"

def main():
    at_model = AffineTransformation(N = None)
   
    # load ideological embeddings at node level
    # node_ca_coordinates_header_names = None # no header : first is entity (node ID), rest is dimensions
    node_ca_coordinates_header_names = {'entity' : 'target ID'} # must have at least a 'node_id' column
    X = at_model.load_node_ca_coordinates_from_file('ca_results/ca_target_cordinates.csv',
            node_ca_coordinates_header_names = node_ca_coordinates_header_names)
    #print(X)

    # Note from 18th March meeting:
    # Just checking: what is the format of input "load_node_ca_coordinates_from_file"?:
    # 1st column is entity ID, nest columns are dimensions. 
    # node_ca_coordinates_header_names is used to specify the name of the entity id column

    # Note from 18th March meeting:
    # Option to consider, include or not in node_ca_coordinates_header_names
    # a key called "dimensions":['latent_dimension_1', etc etc]


    # if attitudinal_reference_data is given at the group level need to aggregate CA coordinates accordingly
    #node_to_group_mapping_header_names = None # no header : first is entity (node ID), second is group name
    node_to_group_mapping_header_names = {'group' : 'party', 'entity' : 'twitter_id'} # must have a 'group' 
                                                                                      # column and a 'entity' column
    XG = at_model.load_node_to_group_mapping_from_file('ca_results/real_node_groups.csv',
            node_to_group_mapping_header_names = node_to_group_mapping_header_names)
    print(XG)

    # Note from 18th March meeting:
    # Just checking: real_node_groups.csv has format
    # group, entity 
    # party1, a
    # party2, b
    # party1, c
    # ....
    # Add check: an entity belong just to 1 group

    #
    # aggregate function to use in order to create group CA coordinates
    group_ca_agg_fun = None # default 'mean' is employed
    #group_ca_agg_fun = 'min'
    #group_ca_agg_fun = custom_agg_python_func
    X = at_model.convert_to_group_ca_coordinates(X, XG, group_ca_agg_fun = group_ca_agg_fun)
    print(X)

    # load attitudinal reference data
    #attitudinal_reference_data_header_names = None # no header : first is group name, rest is dimensions in attitudinal space
    #attitudinal_reference_data_header_names = {'entity' : 'party'} # must have a 'entity' column and an optional 'dimensions' 
                                                                    # column. 'entity' refers to group when group level 
                                                                    # attitudinal reference data is given and to node when 
                                                                    # node level attitudinal reference data is given
    attitudinal_reference_data_header_names = {'entity' : 'party',
            'dimensions' : ['ches_eu_position', 'ches_eu_foreign', 'ches_protectionism']}
    # group level

    # Note from 18th March meeting:
    # Naming:
    # "load_attitudinal_coordinates_from_file" should be "load_attitudinal_referential_coordinates_from_file" 

    Y = at_model.load_attitudinal_coordinates_from_file('/home/foula/FoulaDatasetAttitudinalEmbedding/groups_attitudinal_coordinates.csv', attitudinal_reference_data_header_names = attitudinal_reference_data_header_names)
    # node level
    #Y = at_model.load_attitudinal_coordinates_from_file('/home/foula/correspondence_analysis/linate_module/ca_results/generated_node_attitudinal_coordinates.csv', attitudinal_reference_data_header_names = attitudinal_reference_data_header_names)
    #print(Y)

    # Fitting
    at_model.fit(X, Y)
    print(at_model.employed_N_)
    print(at_model.T_tilda_aff_np_)

    # Note from 18th March meeting:
    # If "fit" computes the affine transformation matrix, then "transform" applies the transformation 
    # to any input: the 1) ideological coordinate of sources, 2) of targets, and 3) of any other points


    X_MPs = at_model.load_node_ca_coordinates_from_file('ca_results/ideological_coordinates_of_MPs.csv',
            node_ca_coordinates_header_names = node_ca_coordinates_header_names)
    X_followers = at_model.load_node_ca_coordinates_from_file('ca_results/ideological_followers_of_MPs.csv',
            node_ca_coordinates_header_names = node_ca_coordinates_header_names)
    X_whatever = np.array([[bla bla bla],[bla bla bla],[bla bla bla],])

    Y_MPs = at_mode.transform(X_MPs)
    Y_followers = at_mode.transform(X_followers)
    Y_whatever = at_mode.transform(X_whatever)

    # Note from 18th March meeting:
    # save index and columns of T_tilde so people can know what are te domain and image dimensions of the transformation

    at_model.save_affine_transformation('ca_results/affine_transformation.csv')

if __name__ == "__main__":
    main()
