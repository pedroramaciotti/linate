# test 'IdeologicalEmbedding' class

from linate import IdeologicalEmbedding

def main():
    standardize_mean = False
    #ideological_embedding_model = IdeologicalEmbedding(n_latent_dimensions = 5, engine = 'auto',
     #       in_degree_threshold = 2, out_degree_threshold = 2, force_bipartite = False,
     #       standardize_mean = standardize_mean, standardize_std = True)
    ideological_embedding_model = IdeologicalEmbedding(n_latent_dimensions = 5, engine = 'linate_ds',
            in_degree_threshold = 2, out_degree_threshold = 2, force_bipartite = False,
            standardize_mean = standardize_mean, standardize_std = True)
   
    # This is the original example
    #network_file_header_names = {'source':'source', 'target':'target'}
    network_file_header_names = None
    X = ideological_embedding_model.load_input_from_file('/home/foula/FoulaDatasetAttitudinalEmbedding/bipartite_831MPs_4424402followers.csv', network_file_header_names = network_file_header_names)
    #X = ideological_embedding_model.load_input_from_file('data/twitter_networks/test_graph_A.csv',
    #X = ideological_embedding_model.load_input_from_file('data/twitter_networks/test_graph_B.csv',

    # Example loading the "RawData" filder examples
    #X = ideological_embedding_model.load_input_from_file('Australia_20201027_MPs_followers_tids.csv',
     #       network_file_header_names = {'source':'followers_id', 'target':'twitter_id'})

    # Fitting
    ideological_embedding_model.fit(X)

    # Saving to file (either scaled or non-scaled coordinates depending on user choice)
    ideological_embedding_model.save_ideological_embedding_target_latent_dimensions('ideological_embedding_results/target_dimensions_ds.csv')
    ideological_embedding_model.save_ideological_embedding_source_latent_dimensions('ideological_embedding_results/source_dimensions_ds.csv')

    # Retrieving the coordinates in program
    target_coords = ideological_embedding_model.ideological_embedding_target_latent_dimensions_
    source_coords = ideological_embedding_model.ideological_embedding_source_latent_dimensions_

    # Check the number of source and target nodes
    print('Number of targets', ideological_embedding_model.target_entity_no_)
    print('Number of soutces', ideological_embedding_model.source_entity_no_)

    #
    print('Eigenvalues', ideological_embedding_model.eigenvalues_)
    print('Total inertia', ideological_embedding_model.total_inertia_)
    print('Explained inertia', ideological_embedding_model.explained_inertia_)

    # compute distance of predicted dimensions with given benchmark dimensions
    #benchmark_ideological_dimensions_data_header_names = None # default: first column is entity, rest of columns are dimensions
    #benchmark_ideological_dimensions_data_header_names = {'entity': 'target ID'} # has to define an 'entity' column
    benchmark_ideological_dimensions_data_header_names = {'entity': 'target ID',
            'dimensions': ['latent_dimension_1', 'latent_dimension_3']}  # can have an optional 'dimensions' entry
    Bnchmark_Y = ideological_embedding_model.load_benchmark_ideological_dimensions('ideological_embedding_results/target_dimensions_auto.csv',
            benchmark_ideological_dimensions_data_header_names = benchmark_ideological_dimensions_data_header_names)
    print(Bnchmark_Y)


if __name__ == "__main__":
    main()
