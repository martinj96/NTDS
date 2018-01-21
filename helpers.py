# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 16:52:13 2018

@author: andrea
"""

def project_graph(U, F):
    """ Project graph U (similar user-user network) on F (social network).
        The projection here consists in using the weights from U for the
        connections of F where the connections exists. This allows to weight an
        unweighted graph, without changing its connectivity.
        
        Args:
            U : array_like
                User-user network adjacency matrix
            F : array-like
                Social network adjacency matrix
        
        Returns:
            G : array_like
                The projected graph adjacency matrix
    """
    
    G = F.copy()
    ind = G.nonzero()
    G[ind] = F[ind]
    return G