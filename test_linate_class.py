# linate class

from linate import LINATE

def main():
    model = LINATE(n_components = 2, out_degree_threshold = 2, in_degree_threshold = 2, engine = 'fbpca')
    model.fit('data/twitter_networks/Austria_20201027_MPs_followers_tids.csv')

    print(type(model.eigenvalues_), type(model.total_inertia_), type(model.explained_inertia_))

if __name__ == "__main__":
    main()
