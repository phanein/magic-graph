__author__ = 'bperozzi'

import magicgraph

from itertools import permutations


def clique(size):
    return magicgraph.from_adjlist(permutations(range(1,size+1)))


def grid(size):
  G = magicgraph.Graph()

  for i in xrange(size):
    for j in xrange(size):

      node_id = i * size + j

      #left
      if j > 0:
        G[node_id].append(node_id - 1)

      #right
      if j < size-1:
        G[node_id].append(node_id + 1)

      #left
      if i > 0:
        G[node_id].append(node_id - size)

      #bottom
      if i < size-1:
        G[node_id].append(node_id + size)

  G.make_consistent()

  return G