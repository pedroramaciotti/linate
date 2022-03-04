# linate class

from linate import LINATE

def main():
    model = LINATE(n_components = 2, out_degree_threshold = 2, in_degree_threshold = 2, engine = 'fbpca')
    model.fit('data/twitter_networks/Austria_20201027_MPs_followers_tids.csv')

    #print(len(model.ca_row_coordinates_), len(model.row_ids_), 
     #       len(model.ca_column_coordinates_), len(model.column_ids_))

if __name__ == "__main__":
    main()
