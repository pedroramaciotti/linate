# linate class

from linate import CA

def main():
    ca_model = CA(n_components = 5, engine = 'auto', in_degree_threshold = 2, out_degree_threshold = 2)
    #ca_model = CA(n_components = 5, engine = 'linate_ds', in_degree_threshold = 2, out_degree_threshold = 2)
    #network_file_header_names = {'source':'source', 'target':'target1'}
    network_file_header_names = None
    X = ca_model.load_input_from_file('/home/foula/FoulaDatasetAttitudinalEmbedding/bipartite_831MPs_4424402followers.csv',
            network_file_header_names = network_file_header_names)
    ca_model.fit(X)
    ca_model.save_ca_target_coordinates('ca_results/ca_target_cordinates.csv')
    ca_model.save_ca_source_coordinates('ca_results/ca_source_cordinates.csv')

    # TODO request intertia

if __name__ == "__main__":
    main()
