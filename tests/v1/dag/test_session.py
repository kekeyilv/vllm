# SPDX-License-Identifier: Apache-2.0
"""Unit tests for DAGSession."""
import pytest

from vllm.v1.dag.session import DAGSession
from vllm.v1.dag.topology import DAGNode, DAGTopology


def _fork_join_dag() -> DAGTopology:
    dag = DAGTopology()
    dag.add_node(DAGNode("A", "prefix", []))
    dag.add_node(DAGNode("B", "tool1", ["A"]))
    dag.add_node(DAGNode("C", "tool2", ["A"]))
    dag.add_node(DAGNode("D", "merge", ["B", "C"]))
    return dag


def test_position_assignment_symmetry():
    """Parallel branches must share the same starting offset."""
    dag = _fork_join_dag()
    session = DAGSession(dag)
    session.register_completion("A", num_tokens=100, block_ids=[[1, 2]])

    offset_B = session.compute_offset("B")
    offset_C = session.compute_offset("C")
    assert offset_B == offset_C == 100


def test_merge_offset_uses_max_parent():
    """Merge node offset = max(parent offset + parent length)."""
    dag = _fork_join_dag()
    session = DAGSession(dag)
    session.register_completion("A", num_tokens=100, block_ids=[[1]])
    session.register_completion("B", num_tokens=200, block_ids=[[2, 3]])
    session.register_completion("C", num_tokens=50, block_ids=[[4]])

    # D's offset = max(B.offset+B.len, C.offset+C.len) = max(300, 150) = 300
    assert session.compute_offset("D") == 300


def test_tail_alignment():
    """Tail deltas should make all branches appear the same length."""
    dag = _fork_join_dag()
    session = DAGSession(dag)
    session.register_completion("A", 200, [[]])
    session.register_completion("B", 1000, [[]])  # long branch
    session.register_completion("C", 30, [[]])    # short branch

    deltas = session.compute_tail_deltas("D")
    assert deltas["B"] == 0
    assert deltas["C"] == 970  # 1000 - 30


def test_inherited_block_ids_ordered():
    """Ancestor blocks should be deduplicated and ordered by offset."""
    dag = _fork_join_dag()
    session = DAGSession(dag)
    session.register_completion("A", 100, [[10, 11]])
    session.register_completion("B", 50, [[20, 21]])
    session.register_completion("C", 50, [[30]])

    inherited = session.get_inherited_block_ids("D")
    assert len(inherited) == 1
    # A's blocks come first (smallest offset), then B and C in offset order
    assert 10 in inherited[0]
    assert 11 in inherited[0]


def test_get_position_ids():
    dag = _fork_join_dag()
    session = DAGSession(dag)
    session.register_completion("A", 5, [[]])

    ids = session.get_position_ids("B", 3)
    assert ids.tolist() == [5, 6, 7]


def test_register_completion_offset_zero_for_root():
    dag = DAGTopology()
    dag.add_node(DAGNode("root", "hello", []))
    session = DAGSession(dag)
    assert session.compute_offset("root") == 0
