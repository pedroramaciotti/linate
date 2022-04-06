# generate grouped data and a general directed social graph

import synthetic_data_generation as gen

import numpy as np

def main():

    random_state = None # for random number generation

    N_entities = 500

    ######################################################
    # Generating synthetic data in ideological space :   #
    ######################################################

    # entities in ideological space
    entity_mixmod = gen.load_gaussian_mixture_model_mu_and_cov_from_files('parameters/phi_mu.txt',
            'parameters/phi_cov.txt')
    #print(entity_mixmod)
    phi_directed, phi_directed_info, phi_directed_group, \
            phi_directed_group_info = gen.generate_entities_in_idelogical_space(N_entities, 
                    entity_mixmod, random_state = random_state)
    gen.save_array_to_file(phi_directed, 'generated_data/phi_directed.txt')
    gen.save_array_to_file(phi_directed_info, 'generated_data/phi_directed_info.txt', is_float = False)
    gen.save_array_to_file(phi_directed_group, 'generated_data/phi_directed_group.txt')
    gen.save_array_to_file(phi_directed_group_info, 'generated_data/phi_directed_group_info.txt', is_float = False)

    #######################################################################
    # Creating unobservable attitudes from prescribed ideologies: A       #
    #######################################################################

    T_tilda_aff = gen.load_augmented_transformation_from_file('parameters/T_tilda_aff.txt')

    phi_directed = gen.load_array_from_file('generated_data/phi_directed.txt')
    r_directed, r_directed_group = gen.transform_entity_dimensions_to_new_space(phi_directed.T,
            T_tilda_aff, entity_dimensions_info = phi_directed_info.T)
    #print(r_directed)
    gen.save_array_to_file(r_directed, 'generated_data/r_directed.txt')
    gen.save_array_to_file(r_directed_group, 'generated_data/r_directed_group.txt')

    #############################################################################
    # Computing the social graph using distances within a given space           #
    #############################################################################
    r_directed_info = gen.load_array_from_file('generated_data/phi_directed_info.txt', is_float = False)
    r_directed = gen.load_array_from_file('generated_data/r_directed.txt')
    #print(r_directed_info.shape, r_directed.shape)

    alpha = 2
    beta = 2
    social_graph = gen.compute_social_graph(r_directed_info.T[0], r_directed, r_directed_info.T[0],
            r_directed, random_state, alpha, beta)
    print(social_graph.shape)

if __name__ == "__main__":
    main()
