# SPDX-License-Identifier: Apache-2.0
"""Unit tests for DAGTopology."""
import pytest

from vllm.v1.dag.topology import DAGNode, DAGTopology


def _make_fork_join_dag() -> DAGTopology:
    """
    A -> B
    A -> C
    B -> D
    C -> D
    """
    dag = DAGTopology()
    dag.add_node(DAGNode("A", "prefix", []))
    dag.add_node(DAGNode("B", "tool1 result", ["A"]))
    dag.add_node(DAGNode("C", "tool2 result", ["A"]))
    dag.add_node(DAGNode("D", "merge", ["B", "C"]))
    return dag


def test_topo_sort_fork_join():
    dag = _make_fork_join_dag()
    order = dag.topo_sort()
    assert order[0] == "A"
    assert order[-1] == "D"
    assert set(order[1:3]) == {"B", "C"}


def test_ancestors():
    dag = _make_fork_join_dag()
    assert dag.ancestors("D") == {"A", "B", "C"}
    assert dag.ancestors("B") == {"A"}
    assert dag.ancestors("C") == {"A"}
    assert dag.ancestors("A") == set()


def test_is_merge_node():
    dag = _make_fork_join_dag()
    assert dag.is_merge_node("D") is True
    assert dag.is_merge_node("B") is False
    assert dag.is_merge_node("A") is False


def test_is_fork_point():
    dag = _make_fork_join_dag()
    assert dag.is_fork_point("A") is True
    assert dag.is_fork_point("B") is False
    assert dag.is_fork_point("D") is False


def test_linear_dag():
    dag = DAGTopology()
    dag.add_node(DAGNode("X", "p1", []))
    dag.add_node(DAGNode("Y", "p2", ["X"]))
    dag.add_node(DAGNode("Z", "p3", ["Y"]))
    assert dag.topo_sort() == ["X", "Y", "Z"]
    assert dag.ancestors("Z") == {"X", "Y"}
    assert dag.is_merge_node("Z") is False
    assert dag.is_fork_point("X") is False
