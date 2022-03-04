# linate class

from linate import CA

def main():
    ca_model = CA(n_components = 2, engine = 'auto', in_degree_threshold = 2, out_degree_threshold = 2)
    #ca_model = CA(n_components = 2, engine = 'linate_ds', in_degree_threshold = 2, out_degree_threshold = 2)
    #network_file_header_names = {'source':'source', 'target':'target1'}
    network_file_header_names = None
    X = ca_model.load_input_from_file('/home/foula/FoulaDatasetAttitudinalEmbedding/bipartite_831MPs_4424402followers.csv',
            network_file_header_names = network_file_header_names)
    print(type(X))

    #print(model.ca_row_coordinates_)

    #print(len(model.ca_row_coordinates_), len(model.row_ids_), 
     #       len(model.ca_column_coordinates_), len(model.column_ids_))

if __name__ == "__main__":
    main()
