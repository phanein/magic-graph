"""Graph utilities."""

import logging
from io import open
from time import time
from itertools import izip
from collections import defaultdict, Iterable
import math
import random
import scipy.sparse as sparse


__author__ = "Bryan Perozzi"
__email__ = "bperozzi@acm.org"
__version__ = "0.0.1"

logger = logging.getLogger("magicgraph")


class DiGraph(defaultdict):
  """Efficient basic implementation of undirected graphs with self loops"""
  def __init__(self, node_class=list):
    super(DiGraph, self).__init__(node_class)

  def nodes(self):
    return self.keys()

  def adjacency_iter(self):
    return self.iteritems()

  def subgraph(self, nodes={}):
    subgraph = Graph()

    for n in nodes:
      if n in self:
        subgraph[n] = [x for x in self[n] if x in nodes]

    return subgraph

  def make_undirected(self):

    t0 = time()

    for v in self.keys():
      for other in self[v]:
        if v != other:
          self[other].append(v)

    t1 = time()
    logger.info('make_undirected: added missing edges {}s'.format(t1-t0))

    self.make_consistent()
    return self

  def make_consistent(self):
    t0 = time()
    for k in self.iterkeys():
      self[k] = list(sorted(set(self[k])))

    t1 = time()
    logger.info('make_consistent: made consistent in {}s'.format(t1-t0))

    self.remove_self_loops()

    return self

  def remove_self_loops(self):

    removed = 0
    t0 = time()

    for x in self:
      if x in self[x]:
        self[x].remove(x)
        removed += 1

    t1 = time()

    logger.info('remove_self_loops: removed {} loops in {}s'.format(removed, (t1-t0)))
    return self

  def check_self_loops(self):
    for x in self:
      for y in self[x]:
        if x == y:
          return True

    return False

  def has_edge(self, v1, v2):
    if v2 in self[v1] or v1 in self[v2]:
      return True
    return False

  def degree(self, nodes=None):
    if isinstance(nodes, Iterable):
      return {v: len(self[v]) for v in nodes}
    else:
      return len(self[nodes])

  def number_of_edges(self):
    """Returns the number of directed edges in the graph"""
    #return sum([self.degree(x) for x in self.iterkeys()]) / 2.0
    return sum([self.degree(x) for x in self.iterkeys()])

  def number_of_nodes(self):
    """Returns the number of nodes in the graph"""
    return self.order()

  def order(self):
    """Returns the number of nodes in the graph"""
    return len(self)

  def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
    """ Returns a truncated random walk.

        path_length: Length of the random walk.
        alpha: probability of restarts.
        start: the start node of the random walk.
    """
    G = self
    if start:
      path = [start]
    else:
      # Sampling is uniform w.r.t V, and not w.r.t E
      path = [rand.choice(G.keys())]

    while len(path) < path_length:
      cur = path[-1]
      if len(G[cur]) > 0:
        if rand.random() >= alpha:
          path.append(rand.choice(G[cur]))
        else:
          path.append(path[0])
      else:
        break
    return path


class Graph(DiGraph):

  def number_of_edges(self):
    """Returns the number of undirected edges in the graph"""
    return super(Graph, self).number_of_edges() / 2.0


class WeightedNode(list):
  """A class which overrides a list to keep a companion list of weights."""
  def __init__(self, **kwargs):
    super(WeightedNode, self).__init__(kwargs)
    self.weights = []
    self.total_weight = None

  def weight(self, index):
    return self.weights[index]

  def append(self, to_append, weight=None):
    super(WeightedNode, self).append(to_append)

    if weight is not None:
      self.weights.append(weight)
    else:
      self.weights.append(1.0)

  def extend(self, iterable, weights=None):

    l_iterable = list(iterable)

    super(WeightedNode, self).extend(l_iterable)

    if weights is not None:
      assert len(l_iterable) == len(weights)
      self.weights.extend(weights)
    else:
      self.weights.extend([1.0 for x in l_iterable])

  def insert(self, index, p_object, weight=None):
    super(WeightedNode, self).insert(index, p_object)

    if weight is not None:
      self.weights.insert(index, weight)
    else:
      self.weights.insert(index, 1.0)

  def pop(self, index=None):
    ret = super(WeightedNode, self).pop(index)
    w = self.weights.pop(index)

    return ret,w

  def remove(self, value):
    idx = self.index(value)

    super(WeightedNode, self).pop(idx)
    self.weights.pop(idx)

  def reverse(self):
    super(WeightedNode, self).reverse()
    self.weights.reverse()

  def sort(self, cmp=None, key=None, reverse=False):

    if key is None:
      key = lambda z: z

    zipped = zip(self, self.weights)
    zipped_sorted = sorted(zipped, cmp=cmp, key=lambda y: key(y[0]), reverse=reverse)

    for idx, x in enumerate(zipped_sorted):
      nid, weight = x
      self[idx] = nid
      self.weights[idx] = weight

  def choice(self, rand):
    upper = self.total_weight
    if upper is None:
      upper = sum(self.weights)
      self.total_weight = upper

    stop = rand.uniform(0., upper)

    total = 0.
    for idx, x in enumerate(self.weights):
      total += x
      if total >= stop:
        #print idx, ':', total, self[idx]
        return self[idx]

    # floating point summations :/
    return self[-1]


class WeightedDiGraph(DiGraph):
  """A weighted directed graph"""
  def __init__(self):
    super(WeightedDiGraph, self).__init__(node_class=WeightedNode)

  def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
    """ Returns a truncated random walk.

        path_length: Length of the random walk.
        alpha: probability of restarts.
        start: the start node of the random walk.
    """
    G = self
    if start:
      path = [start]
    else:
      # Sampling is uniform w.r.t V, and not w.r.t E
      path = [rand.choice(G.keys())]

    while len(path) < path_length:
      cur = path[-1]
      if len(G[cur]) > 0:
        if rand.random() >= alpha:
          path.append(G[cur].choice(rand))
        else:
          path.append(path[0])
      else:
        break
    return path


def save_edgelist(G, file_, sep='\t'):

  t0 = time()

  edges = 0

  with open(file_, 'w') as f:

    for src in G:
      for dst in G[src]:
        f.write(str(src))
        f.write(sep)
        f.write(str(dst))
        f.write('\n')
        edges += 1

  t1 = time()

  logger.info('Wrote {} edges from edge list in {}s'.format(edges,  t1-t0))


def load_edgelist(file_, undirected=False, leftIsSource=True):

  if undirected == False:
    G = DiGraph()
  else:
    G = Graph()

  t0 = time()

  edges = 0

  with open(file_) as f:
    for l in f:
      x, y = l.strip().split()[:2]
      x = int(x)
      y = int(y)

      if leftIsSource:
        G[x].append(y)
        edges += 1
        if undirected:
          G[y].append(x)
      else:
        G[y].append(x)
        edges += 1
        if undirected:
          G[x].append(y)

  t1 = time()

  logger.info('Parsed {} edges from edge list in {}s'.format(edges,  t1-t0))

  G.make_consistent()
  return G


def load_weighted_edgelist(file_, undirected=False):

  G = WeightedDiGraph()
  t0 = time()

  edges = 0

  with open(file_) as f:
    for l in f:
      x, y, w = l.strip().split()[:3]
      x = int(x)
      y = int(y)
      w = int(w)
      G[x].append(y, weight=w)
      edges += 1
      if undirected:
        G[y].append(x, weight=w)

  t1 = time()

  logger.info('Parsed {} edges from edge list in {}s'.format(edges,  t1-t0))

  return G

def load_multigraph_edgelist(file_, second_seperator=':', undirected=False):

  G = WeightedDiGraph()
  t0 = time()

  edges = 0

  with open(file_) as f:
    for l in f:
      x, y, w = l.strip().split()[:3]
      x = int(x)
      y = int(y)
      w = sum([float(a) for a in w.split(second_seperator)])
      G[x].append(y, weight=w)
      edges += 1
      if undirected:
        G[y].append(x, weight=w)

  t1 = time()

  logger.info('Parsed {} edges from edge list in {}s'.format(edges,  t1-t0))

  return G

def from_networkx(G_input, undirected=False):
    G = Graph()

    for idx, x in enumerate(G_input.nodes_iter()):
        for y in G_input[x].iterkeys():
            G[x].append(y)

    if undirected:
        G.make_undirected()

    return G


def from_numpy(x):
    G = Graph()

    # TODO add handling for dense numpy too
    cx = x.tocoo()
    for i,j,v in izip(cx.row, cx.col, cx.data):
        G[i].add(j)
    G.make_consistent()
    return G


def from_adjlist(adjlist):
    G = Graph()

    total_lines = 0
    total_edges = 0

    for row in adjlist:
        node = row[0]
        neighbors = row[1:]
        G[node] = list(sorted(set(neighbors)))
        total_lines += 1
        total_edges += len(neighbors)

    if total_edges == total_lines:
        logger.warn('WARN: May be parsing edge list as adjacency list (# lines == # edges)')

    return G


def to_adjacency_matrix(G, nodelist=None, dtype=None):
    """Inspired by the NetworkX equivalent."""

    if nodelist is None:
        nodelist = sorted(G.nodes())

    nodeset = set(nodelist)
    if len(nodelist) != len(nodeset):
        raise ValueError("Ambiguous ordering: `nodelist` contained duplicates.")

    nlen=len(nodelist)
    index=dict(zip(nodelist,range(nlen)))

    I = []
    J = []
    A_ij = []

    # TODO support weights
    for i,nbrs in G.adjacency_iter():
        for j in nbrs:
            try:
                I.append(index[i])
                J.append(index[j])
                A_ij.append(1)
            except KeyError:
                pass

    A = sparse.coo_matrix((A_ij, (I, J)), shape=(nlen, nlen), dtype=dtype).tocsr()

    return A


def to_laplacian(G, nodelist=None):
    A = to_adjacency_matrix(G, nodelist=nodelist)
    I = sparse.identity(A.shape[0])

    A_sum = sparse.csr_matrix(A.sum(axis=1))
    D = I.multiply(A_sum)
    L = D - A

    return L, D