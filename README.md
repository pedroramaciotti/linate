# LINATE
___
Language-Independent Network Attitudinal Embedding


Check the quickstart jupyter notebook in the "tutorial" folder.



LINATE stands for "Language-Independent Network ATtitudinal Embedding". As its name suggests, it's a module for embedding social networks (graphs) in attitudinal spaces. Attitudinal spaces are geometrical opinion spaces where dimensions act as indicators of positive or negative opinions (i.e., attitudes) towards identifiable attitudinal objects (e.g., ideological positions such as left- or right-wing ideologies, or policy positions such as increasing tax redistribution, or increasing environmental protection).

This module provides tools for two methods: 

1) Ideological embedding: producing a graph embedding in an latent ideological space, where dimensions don't have explicit meaning, but are related to an homophilic model underlying the choises of users forming the graph.

2) Attitudinal embedding: mapping this embedded graph onto a second space that does have explicit meaning for its dimensions. For this, the module uses the position of some reference points that have known positions in both spaces.

Check our publication for further details:

Ramaciotti Morales, Pedro ,Jean-Philippe Cointet, Gabriel Muñoz Zolotoochin, Antonio Fernández Peralta, Gerardo Iñiguez, and Armin Pournaki. "Inferring Attitudinal Spaces in Social Networks." (2022).
https://hal.archives-ouvertes.fr/hal-03573188/document