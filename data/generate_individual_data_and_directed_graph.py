# generate grouped data and a bipartite social graph

import synthetic_data_generation as gen

import numpy as np

def main():

    random_state = None # for random number generation

    N_entities = 1000

    ######################################################
    # Generating synthetic data in ideological space :   #
    ######################################################

    # reference users in ideological space
    ref_mixmod = gen.load_gaussian_mixture_model_mu_and_cov_from_files('parameters/phi_mu.txt',
            'parameters/phi_cov.txt')
    #print(ref_mixmod)
    phi_indvdl_directed, phi_indvdl_directed_info = gen.generate_entities_in_idelogical_space(N_entities, 
            ref_mixmod, random_state = random_state, produce_group_dimensions = False)
    gen.save_array_to_file(phi_indvdl_directed, 'generated_data/phi_indvdl_directed.txt')
    gen.save_array_to_file(phi_indvdl_directed_info, 'generated_data/phi_indvdl_directed_info.txt', is_float = False)

    #######################################################################
    # Creating unobservable attitudes from prescribed ideologies: A       #
    #######################################################################

    T_tilda_aff = gen.load_augmented_transformation_from_file('parameters/T_tilda_aff.txt')

    phi_indvdl_directed = gen.load_array_from_file('generated_data/phi_indvdl_directed.txt')
    r_indvdl_directed = gen.transform_entity_dimensions_to_new_space(phi_indvdl_directed.T, T_tilda_aff,
            produce_group_dimensions = False)
    #print(r_indvdl_directed)
    gen.save_array_to_file(r_indvdl_directed, 'generated_data/r_indvdl_directed.txt')

    #############################################################################
    # Computing the social graph using distances within a given space           #
    #############################################################################
    r_indvdl_directed_info = gen.load_array_from_file('generated_data/phi_indvdl_directed_info.txt', is_float = False)
    r_indvdl_directed = gen.load_array_from_file('generated_data/r_indvdl_directed.txt')

    alpha = 2
    beta = 2
    social_graph = gen.compute_social_graph(r_indvdl_directed_info.T[0], r_indvdl_directed, 
            r_indvdl_directed_info.T[0], r_indvdl_directed, random_state, alpha, beta)
    print(social_graph.shape)

if __name__ == "__main__":
    main()
