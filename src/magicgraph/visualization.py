__author__ = 'bperozzi'

import graph_tool.all as gt
import seaborn as sns


def find_color(value, maxv, minv, palette):
  # XXX need to get min in there

  percentage = (value - minv) / (maxv - minv)

  idx = int(percentage * len(palette))

  idx = min(len(palette) - 1, idx)
  #print value, idx
  #print palette[idx]

  return list(palette[idx]) + [1.0]


def draw_graph_heatmap(graph, value_map, output, directed=False, palette=sns.cubehelix_palette(10, start=.5, rot=-.75), position=None):

  # for normalize
  values = value_map.values()

  maxv = max(values)
  minv = min(values)

  if len(values) != len(graph):
    # some graph nodes missing from map.
    # make them 0
    minv = min(minv, 0)

  gt_graph = gt.Graph(directed=directed)

  node_map = {node: gt_graph.add_vertex() for node in graph}

  if not directed:
    seen_edges = set()

  for node, edges in graph.iteritems():
    i = node_map[node]
    for e in edges:
      j = node_map[e]

      if directed:
        gt_graph.add_edge(i, j)
      else:
        if (j, i) not in seen_edges:
          gt_graph.add_edge(i, j)
          seen_edges.add((i, j))

  node_intensity = gt_graph.new_vertex_property("vector<float>")
  node_label = gt_graph.new_vertex_property("string")

  for id, value in value_map.iteritems():
    node = node_map[id]
    node_intensity[node] = find_color(value, maxv, minv, palette)
    node_label[node] = id

  for id in graph:
    if id not in value_map:
      node = node_map[id]
      node_intensity[node] = find_color(0, maxv, minv, palette)
      node_label[node] = id

  if position is None:
    position = gt.sfdp_layout(gt_graph)

  gt.graph_draw(gt_graph, pos=position,
                vertex_text=node_label,
                vertex_fill_color=node_intensity,
                output=output)

  return position

def draw_graph(graph, value_map=None, output=None, show_ids=False, directed=False, position=None):

  gt_graph = gt.Graph(directed=directed)

  node_map = {node: gt_graph.add_vertex() for node in graph}

  if not directed:
    seen_edges = set()

  for node, edges in graph.iteritems():
    i = node_map[node]
    for e in edges:
      j = node_map[e]

      if directed:
        gt_graph.add_edge(i, j)
      else:
        if (j, i) not in seen_edges:
          gt_graph.add_edge(i, j)
          seen_edges.add((i, j))

  if position is None:
    position = gt.sfdp_layout(gt_graph)

  node_label = gt_graph.new_vertex_property("string")

  if value_map is not None:
    for id, value in value_map.iteritems():
      node = node_map[id]
      node_label[node] = id

    for id in graph:
      if id not in value_map:
        node = node_map[id]
        node_label[node] = id

  elif show_ids:
      for id in graph:
        node = node_map[id]
        node_label[node] = id

  if show_ids:
    gt.graph_draw(gt_graph, pos=position,
                  vertex_text=node_label,
                  output=output)
  else:
    gt.graph_draw(gt_graph, pos=position, output=output)

  return position