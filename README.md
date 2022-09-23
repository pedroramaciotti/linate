# LINATE
___
Language-Independent Network Attitudinal Embedding


Check the quickstart jupyter notebook in the "tutorial" folder.



LINATE stands for "Language-Independent Network ATtitudinal Embedding". As its name suggests, it's a module for embedding social networks (graphs) in attitudinal spaces. Attitudinal spaces are geometrical opinion spaces where dimensions act as indicators of positive or negative opinions (i.e., attitudes) towards identifiable attitudinal objects (e.g., ideological positions such as left- or right-wing ideologies, or policy positions such as increasing tax redistribution, or increasing environmental protection).

This module provides tools for two methods: 

1) Ideological embedding: producing a graph embedding in an latent ideological space, where dimensions don't have explicit meaning, but are related to an homophilic model underlying the choises of users forming the graph.

2) Attitudinal embedding: mapping this embedded graph onto a second space that does have explicit meaning for its dimensions. For this, the module uses the position of some reference points that have known positions in both spaces.

Check our publications for further details:

Ramaciotti Morales, Pedro ,Jean-Philippe Cointet, Gabriel Muñoz Zolotoochin, Antonio Fernández Peralta, Gerardo Iñiguez, and Armin Pournaki. "Inferring Attitudinal Spaces in Social Networks." (2022).
https://hal.archives-ouvertes.fr/hal-03573188/document

Ramaciotti Morales, Pedro , Zografoula Vagena. "Embedding social graphs from multiple national settings in common empirical opinion spaces", Proceedings of the 2022 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining.

### Installation

    pip install linate

### Acknowledgements

This software package been funded by the “European Polarisation Observatory” (EPO) of the CIVICA Consortium, and by Data Intelligence Institute of Paris through the French National Agency for Research (ANR) grant ANR-18-IDEX-0001 “IdEx Universite de Paris”. Data declared the 19 March 2020 and 15 July 2021 at the registry of data processing at the Fondation Nationale de Sciences Politiques (Sciences Po) in accordance with General Data Protection Reg- ulation 2016/679 (GDPR) and Twitter policy. For further details and the respec- tive legal notice, please visit https://medialab.sciencespo.fr/en/activities/epo/.