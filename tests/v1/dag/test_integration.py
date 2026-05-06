# SPDX-License-Identifier: Apache-2.0
"""Integration tests for DAG-RoPE.

These tests verify semantic properties of the implementation without
requiring a GPU.  Full end-to-end tests (fork_join_correctness,
zero_prefill_at_merge) require a live vLLM engine and GPU.
"""
import pytest

from vllm.v1.dag.context import DAGContext
from vllm.v1.dag.session import DAGSession
from vllm.v1.dag.topology import DAGNode, DAGTopology


def _build_tool_call_dag(prompt_len: int = 10, branch_len: int = 5) -> DAGTopology:
    """LLM1 -> [Tool1, Tool2] -> LLM2 pattern."""
    dag = DAGTopology()
    dag.add_node(DAGNode("llm1", "initial prompt" * prompt_len, []))
    dag.add_node(DAGNode("tool1", "tool1 result" * branch_len, ["llm1"]))
    dag.add_node(DAGNode("tool2", "tool2 result" * branch_len, ["llm1"]))
    dag.add_node(DAGNode("llm2", "merge and summarize", ["tool1", "tool2"]))
    return dag


def test_fork_join_dag_structure():
    dag = _build_tool_call_dag()
    order = dag.topo_sort()
    assert order[0] == "llm1"
    assert order[-1] == "llm2"
    assert {"tool1", "tool2"} == set(order[1:3])


def test_branch_order_invariance():
    """Parallel branches must receive the same position offset regardless of order."""
    dag = _build_tool_call_dag(prompt_len=3, branch_len=4)
    session = DAGSession(dag)
    session.register_completion("llm1", 30, [[1, 2, 3]])

    offset_t1 = session.compute_offset("tool1")
    offset_t2 = session.compute_offset("tool2")
    assert offset_t1 == offset_t2, (
        "Parallel branches must start at the same RoPE position"
    )


def test_linear_dag_sequential_positions():
    """A linear DAG A->B->C must produce sequential positions like standard inference."""
    dag = DAGTopology()
    dag.add_node(DAGNode("A", "p1", []))
    dag.add_node(DAGNode("B", "p2", ["A"]))
    dag.add_node(DAGNode("C", "p3", ["B"]))

    session = DAGSession(dag)
    session.register_completion("A", 10, [[]])
    session.register_completion("B", 5, [[]])

    # C starts right after B ends
    assert session.compute_offset("C") == 15
    ids = session.get_position_ids("C", 3)
    assert ids.tolist() == [15, 16, 17]


def test_dag_context_fields():
    """DAGContext must carry the expected fields serialisably."""
    ctx = DAGContext(
        node_id="llm2",
        position_offset=100,
        inherited_block_ids=[[1, 2, 3], [4, 5]],
        is_merge=True,
        tail_deltas={"tool1": 0, "tool2": 970},
    )
    assert ctx.node_id == "llm2"
    assert ctx.position_offset == 100
    assert ctx.is_merge is True
    assert ctx.tail_deltas["tool2"] == 970


# ---------------------------------------------------------------------------
# Placeholder tests that require a live engine (marked xfail until wired up)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="requires GPU + full vLLM engine")
def test_fork_join_correctness():
    """DAG-RoPE output should be order-invariant for parallel branches."""
    # Run same DAG with (tool1, tool2) and (tool2, tool1) ordering.
    # With DAG-RoPE the outputs should be identical (modulo fp16 noise).
    # With flatten baseline the outputs differ.
    pass


@pytest.mark.skip(reason="requires GPU + full vLLM engine")
def test_zero_prefill_at_merge():
    """Merge node should not re-prefill ancestor tokens."""
    # Instrument prefill token counter.
    # Verify merge node only prefills its own prompt tokens, not inherited ones.
    pass
