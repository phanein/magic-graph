import unittest
import magicgraph
import random

from magicgraph.generators import clique


class TestDiGraph(unittest.TestCase):
  def test_nodes(self):

    network = clique(5)
    self.assertEqual(set(range(1, 6)).difference(network.nodes()), set())

  def test_adjacency_iter(self):
    network = clique(3)

    self.assertEqual([x for x in network.adjacency_iter()], [(1, [2, 3]), (2, [1, 3]), (3,[1, 2])])

  def test_subgraph(self):
    network = clique(5)

    self.assertEqual(network.subgraph({1, 2}), {1: [2], 2: [1]})

  def test_make_undirected(self):
    network = clique(5)

    network[4] = []
    network.make_undirected()

    self.assertTrue(network == {1: [2, 3, 4, 5], 2: [1, 3, 4, 5], 3: [1, 2, 4, 5], 4: [1, 2, 3, 5], 5: [1, 2, 3, 4]})

  def test_make_consistent(self):
    network = clique(5)

    network[5].extend([1, 2, 3, 4])
    self.assertNotEqual(network[5], [1, 2, 3, 4])

    network.make_consistent()
    self.assertEqual(network[5], [1, 2, 3, 4])

  def test_remove_self_loops(self):
    network = clique(5)

    network[5].append(5)
    network.remove_self_loops()

    self.assertEqual(network[5], [1, 2, 3, 4])

  def test_check_self_loops(self):
    network = clique(5)
    self.assertFalse(network.check_self_loops())

    network[5].append(5)
    self.assertTrue(network.check_self_loops())

  def test_has_edge(self):
    network = clique(5)

    self.assertTrue(network.has_edge(1, 5))
    self.assertFalse(network.has_edge(1, 6))
    self.assertFalse(network.has_edge(6, 1))

  def test_degree(self):
    network = clique(5)

    self.assertEqual(network.degree(1), 4)
    self.assertEqual(network.degree([1, 2, 3]), {1: 4, 2: 4, 3: 4})

  def test_order(self):
    network = clique(5)
    self.assertEqual(network.order(), 5)

  def test_number_of_edges(self):
    network = clique(5)
    self.assertEqual(network.number_of_edges(), 4 + 3 + 2 + 1)

    network = clique(4)
    self.assertEqual(network.number_of_edges(), 3 + 2 + 1)

  def test_number_of_nodes(self):
    network = clique(5)
    self.assertEqual(network.number_of_nodes(), 5)


class TestWeightedNode(unittest.TestCase):
  def test_append(self):
    node = magicgraph.WeightedNode()
    self.assertEqual(node.weights, [])

    node.append(1, 1.0)

    self.assertEqual(node, [1])
    self.assertEqual(node.weight(0), 1.0)

  def test_extend(self):
    node = magicgraph.WeightedNode()
    self.assertEqual(node.weights, [])

    node.extend([1, 2, 3, 4], [1., 0.5, 0.25, 0.125])

    self.assertEqual(node, [1, 2, 3, 4])
    self.assertEqual(node.weights, [1., 0.5, 0.25, 0.125])

  def test_pop(self):
    node = magicgraph.WeightedNode()
    node.extend([1, 2, 3, 4], [1., 0.5, 0.25, 0.125])

    dst_removed, weight_removed = node.pop(2)

    self.assertEqual(dst_removed, 3)
    self.assertEqual(weight_removed, 0.25)
    self.assertEqual(node, [1, 2, 4])
    self.assertEqual(node.weights, [1., 0.5, 0.125])

  def test_remove(self):
    node = magicgraph.WeightedNode()
    node.extend([1, 2, 3, 4], [1., 0.5, 0.25, 0.125])

    node.remove(3)

    self.assertEqual(node, [1, 2, 4])
    self.assertEqual(node.weights, [1., 0.5, 0.125])

  def test_choice(self):
    node = magicgraph.WeightedNode()
    node.extend([1, 2, 3, 4], [1., 0.5, 0.25, 0.125])

    rand = random.Random(0)
    times_chose = {x:0 for x in node}
    for x in range(0,100):
      times_chose[node.choice(rand)] += 1

    self.assertLess(times_chose[2], times_chose[1])
    self.assertLess(times_chose[3], times_chose[2])
    self.assertLess(times_chose[4], times_chose[3])

#class TestWeightedDiGraph(unittest.TestCase):
#  def test_random_walk(self):


if __name__ == '__main__':
    unittest.main()