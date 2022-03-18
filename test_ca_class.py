# linate class

#from sklearn.utils.estimator_checks import check_estimator

from linate import CA


# Note from 18th March meeting:
# Naming: instead of CA, it could be:
# "IdeologicalEmbedding"
# Accordingly, remove mentions to "ca" across ideological code: for example 
# ca_model.save_ca_target_coordinates should be ca_model.save_target_coordinates
# ca_model.ca_target_coordinates_ should be ca_model.target_coordinates_

# Note from 18th March meeting:
# force_bipartite issue
# By default, force_bipartite=True (because if an MP is following (source), we don't want that signal)
# In practice, to implement force_bipartite, the sequence is:
# 1) remove the source nodes that are also in target, and then 2) applied degree thresholds

def main():
    standardize_mean = True
    ca_model = CA(n_components = 5, engine = 'auto', in_degree_threshold = 2, 
            out_degree_threshold = 2, force_bipartite = False, standardize_mean = standardize_mean, standardize_std = True)
    #ca_model = CA(n_components = 5, engine = 'auto', in_degree_threshold = 2, 
    #        out_degree_threshold = 2, force_bipartite = False, standardize_mean = True, standardize_std = False)
    
    # This is the original example
    #network_file_header_names = {'source':'source', 'target':'target'}
    network_file_header_names = None
    X = ca_model.load_input_from_file('/home/foula/FoulaDatasetAttitudinalEmbedding/bipartite_831MPs_4424402followers.csv',
            network_file_header_names = network_file_header_names)
    #X = ca_model.load_input_from_file('data/twitter_networks/test_graph_A.csv',
    #X = ca_model.load_input_from_file('data/twitter_networks/test_graph_B.csv',

    # Example loading the "RawData" filder examples
    #X = ca_model.load_input_from_file('Australia_20201027_MPs_followers_tids.csv',
     #       network_file_header_names = {'source':'followers_id', 'target':'twitter_id'})

    # Fitting
    ca_model.fit(X)

    # Saving to file
    ca_model.save_ca_target_coordinates('ca_results/ca_target_cordinates.csv')
    ca_model.save_ca_source_coordinates('ca_results/ca_source_cordinates.csv')


    # Retrieving the coordinates in program
    target_coords = ca_model.ca_target_coordinates_
    source_coords = ca_model.ca_source_coordinates_

    # Check the number of source and target nodes
    ca_model.target_users_no_
    ca_model.source_users_no_

    #
    print('eigenvalues', ca_model.eigenvalues_)
    print('total inertia', ca_model.total_inertia_)
    print('explained inertia', ca_model.explained_inertia_)

    # Note from 18th March meeting:
    # Overwrite target coordinates depending on whether the user tells to standardize mean and std
    if standardize_mean: 
        # Saving scaled versions to file
        ca_model.save_ca_scaled_target_coordinates('ca_results/ca_scaled_target_cordinates.csv')
        ca_model.save_ca_scaled_source_coordinates('ca_results/ca_scaled_source_cordinates.csv')

    #check_estimator(ca_model)

if __name__ == "__main__":
    main()
