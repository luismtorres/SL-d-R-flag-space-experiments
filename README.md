FlagTools is a custom-built library which serves to provide computational tools for computing with complete flags of Euclidean space R^d.
A (complete) flag is a sequence F of nested subspaces F^j (j=1,...,d-1) of R^d where the dimension of F^j is j.
In particular, a complete flag of R^d is represented by a (non-unique) basis of R^d.
Two flags F and G are 'transverse' if F^j and G^{d-j} direct sum to R^d for each j=1,...,d-1.
The author is interested in the space of ordered triples (F,G,H) of complete flags of R^d which are 'transverse,' i.e. all possible pairs are transverse.
As it turns out, the space of flags transverse to a given fixed flag F-, which without loss of generality we may assume is represented by the identity matrix, can be parametrized by the space of unipotent, upper triangular matrices in SL(d,R).
As such, matrix inversion induces an operation on the space of flags transverse to F-, which we refer to as the operation 'iota' within the library documentation.

We are interested in studying the space of flags antipodal to F-. This space is of interest since it appears in various areas of mathematics as an important geometric environment. For example, this is where limit sets of Borel Anosov subgroups of SL(d,R) live (a primary focus of research for the author), and knowing which components of the space of triples are preserved (or not) under the operation F -> iota F informs whether certain constructions of Borel Anosov subgroups are possible (or not).
As such, we are namely interested in its connected components, and we implement a well-known correspondence between the components and the orbits of a representation which uses so-called Lusztig coefficients in order to perform computations.
This allows us to represent components by matrices, and our iota operation admits a corresponding matrix operation which we also implement.
We are also interested in the space of ordered triples of transverse flags, and we implement an invariant of such triples known as the 'triple ratio.'
This invariant is useful for detecting when the connected components of the space of ordered transverse triples are preserved by certain permutations of the triple, or preserved by fixing the first and third flags in a triple and applying iota to the middle flag, verifying certain guesses about what the space of triples can look like.
Moreover, the limit sets of Borel Anosov subgroups of SL(d,R) also live in this more restricted space, and understanding the connected components by permutations and applying iota to the middle flag informs whether or not certain constructions of these subgroups is possible.
