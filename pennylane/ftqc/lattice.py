# Copyright 2025 Xanadu Quantum Technologies Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=protected-access, inconsistent-return-statements
"""
This file defines classes and functions for creating lattice objects that store topological
connectivity information.
"""

from enum import Enum

import networkx as nx


class Lattice:
    """Represents a qubit lattice structure.

    This Lattice class, inspired by the design of :class:`~pennylane.spin.Lattice`, leverages `NetworkX` to represent the relationships within the lattice structure.

        Args:
            lattice_shape: Name of the lattice shape.
            graph (nx.Graph): A NetworkX undirected graph object. If provided, `nodes` and `edges` are ignored.
            nodes (List): Nodes to construct a graph object. Ignored if `graph` is provided.
            edges (List): Edges to construct the graph. Ignored if `graph` is provided.
        Raises:
            ValueError: If neither `graph` nor both `nodes` and `edges` are provided.
    """

    # TODOs: To support braiding operations, Lattice should support nodes/edges addition/deletion.

    def __init__(
        self, lattice_shape: str, graph: nx.Graph = None, nodes: list = None, edges: list = None
    ):
        self._lattice_shape = lattice_shape
        if graph is None:
            if nodes is None and edges is None:
                raise ValueError(
                    "Neither a networkx Graph object nor nodes together with edges are provided."
                )
            self._graph = nx.Graph()
            self._graph.add_nodes_from(nodes)
            self._graph.add_edges_from(edges)
        else:
            self._graph = graph

    @property
    def get_lattice_shape(self) -> str:
        r"""Returns the lattice shape name."""
        return self._lattice_shape

    def get_neighbors(self, node):
        r"""Returns the neighbors of a given node in the lattice.

        Args:
            node: a target node label.
        """
        return self._graph.neighbors(node)

    @property
    def get_nodes(self):
        r"""Returns all nodes in the lattice."""
        return self._graph.nodes

    @property
    def get_edges(self):
        r"""Returns all edges in the lattice."""
        return self._graph.edges

    @property
    def get_graph(self) -> nx.Graph:
        r"""Returns the underlying NetworkX graph object representing the lattice."""
        return self._graph


class LatticeShape(Enum):
    """Enum to define valid set of lattice shape supported."""

    chain = 1
    square = 2
    rectangle = 3
    triangle = 4
    honeycomb = 5
    cubic = 6


# map between lattice name and dimensions
_LATTICE_DIM_MAP = {
    "chain": 1,
    "square": 2,
    "rectangle": 2,
    "cubic": 3,
    "triangle": 2,
    "honeycomb": 2,
}

# map between lattice name and networkx method name
_LATTICE_GENERATOR_MAP = {
    "chain": "grid_graph",
    "square": "grid_graph",
    "rectangle": "grid_graph",
    "cubic": "grid_graph",
    "triangle": "triangular_lattice_graph",
    "honeycomb": "hexagonal_lattice_graph",
}


def generate_lattice(dims: list[int], lattice: str) -> Lattice:
    r"""Generates a :class:`~pennylane.ftqc.Lattice` object with a given lattice dimensions and its shape name.

    Args:
        dims(List[int]): Dimensions for lattice generation. For lattices generated by `nx.grid_graph` ( ``'chain'``, ``'rectangle'``,  ``'square'``, ``'cubic'``),
        `dims` contains the number of nodes in the each direction of grid. Per ``'honeycomb'`` or ``'triangle'``, the generated lattices will have dims[0] rows and dims[1]
        columns of hexagons or triangles.
        lattice (str): Shape of the lattice. Input values can be ``'chain'``, ``'square'``, ``'rectangle'``, ``'honeycomb'``, ``'triangle'``, ``'cubic'``.

    Returns:
        a :class:`~pennylane.ftqc.Lattice` object.

    Raises:
        ValueError: If the lattice shape is not supported or the dimensions are invalid.
    """

    lattice_shape = lattice.strip().lower()
    supported_shape = [shape.name for shape in LatticeShape]

    if lattice_shape not in supported_shape:
        raise ValueError(
            f"Lattice shape, '{lattice}' is not supported."
            f"Please set lattice to: {supported_shape}."
        )

    if _LATTICE_DIM_MAP[lattice_shape] != len(dims):
        raise ValueError(
            f"For a {lattice_shape} lattice, the length of dims should {_LATTICE_DIM_MAP[lattice_shape]} instead of {len(dims)}"
        )

    lattice_generate_method = getattr(nx, _LATTICE_GENERATOR_MAP[lattice_shape])

    if _LATTICE_GENERATOR_MAP[lattice_shape] == "grid_graph":
        lattice_obj = Lattice(lattice_shape, lattice_generate_method(dims))
        return lattice_obj

    if _LATTICE_GENERATOR_MAP[lattice_shape] in [
        "triangular_lattice_graph",
        "hexagonal_lattice_graph",
    ]:
        lattice_obj = Lattice(lattice_shape, lattice_generate_method(dims[0], dims[1]))
        return lattice_obj
