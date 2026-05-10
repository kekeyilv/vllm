# SPDX-License-Identifier: Apache-2.0
"""DAG session manager - tracks KV cache state across a DAG execution."""

from dataclasses import dataclass, field
import itertools
from typing import Dict, Iterable, List, Set

import torch

from vllm.v1.dag.topology import DAGTopology


@dataclass
class NodeState:
    """Runtime state for a completed DAG node."""

    completed: bool = False
    offset: int = 0  # RoPE position offset (first token position)
    length: int = 0  # Number of tokens in this node
    request_id: str = ""


@dataclass
class DAGContext:
    """Attached to a request to enable DAG-aware position assignment.

    Serializable (plain Python types only) so it can transit the ZMQ boundary
    between EngineCore and the model-runner worker process.
    """

    node_id: str
    # Absolute RoPE start position for this node's first token.
    position_offset: int
    # Request id of ancestors
    ancestor_req_ids: List[str] = field(default_factory=list)
    is_merge: bool = False
    # tail-alignment deltas: {parent_node_id: shift}
    tail_deltas: Dict[str, int] = field(default_factory=dict)
    # Total number of tokens in ancestors' prompt and output
    num_inherited_tokens: int = 0


class DAGSession:
    """Manages KV cache and position state for a single DAG execution."""

    def __init__(
        self,
        dag: DAGTopology,
        sys_prompt: str,
        session_id: int = 0,
    ) -> None:
        self.dag = dag
        self.session_id = session_id
        self.node_states: Dict[str, NodeState] = {}
        self.system_prompt = sys_prompt
        self._ancestors_cache: Dict[str, Set[str]] = {}
        self._tail_deltas_cache: Dict[str, Dict[str, int]] = {}

        for nid in dag.topo_sort():
            self._ancestors_cache[nid] = dag.ancestors(nid)

    def get_context(self, node_id: str) -> DAGContext:
        return DAGContext(
            node_id=node_id,
            position_offset=self.compute_offset(node_id),
            ancestor_req_ids=self.get_ancestor_req_ids(node_id),
            tail_deltas=self.compute_tail_deltas(node_id),
            num_inherited_tokens=self.get_num_inherited_tokens(node_id),
        )

    def compute_offset(self, node_id: str) -> int:
        """Compute RoPE position offset for a node based on DAG topology.

        Parallel branches share the same starting offset (the max over parents).
        """
        node = self.dag.get_node(node_id)
        if not node.parents:
            return 0
        return max(
            self.node_states[pid].offset + self.node_states[pid].length
            for pid in node.parents
        )

    def compute_tail_deltas(self, node_id: str) -> Dict[str, int]:
        """Compute tail-alignment shifts per parent.

        Returns {parent_id: delta} where delta = L_max - len(parent).
        Shorter branches are shifted so their tails align with the longest.
        """
        if node_id in self._tail_deltas_cache:
            return self._tail_deltas_cache[node_id]
        node = self.dag.get_node(node_id)

        if len(node.parents) == 0:
            return {}
        L_max = max(self.node_states[pid].length for pid in node.parents)
        tail_deltas = {
            pid: L_max - self.node_states[pid].length for pid in node.parents
        }

        for pid in node.parents:
            # Recursively update tail_deltas
            tail_deltas.update(self.compute_tail_deltas(pid))

        self._tail_deltas_cache[node_id] = tail_deltas
        return tail_deltas

    def get_num_inherited_tokens(self, node_id: str) -> int:
        return sum(
            self.node_states[anc_id].length for anc_id in self._ancestors_cache[node_id]
        )

    def get_position_ids(self, node_id: str, num_tokens: int) -> torch.Tensor:
        """Get position IDs for a node's own tokens."""
        offset = self.compute_offset(node_id)
        return torch.arange(offset, offset + num_tokens, dtype=torch.long)

    def get_ancestor_req_ids(self, node_id: str) -> List[str]:
        ancestors = self._ancestors_cache[node_id]
        return [self.node_states[anc_id].request_id for anc_id in ancestors]

    def register_completion(
        self, node_id: str, num_tokens: int, request_id: str
    ) -> None:
        """Register a completed node's state for use by downstream nodes."""
        self.node_states[node_id] = NodeState(
            offset=self.compute_offset(node_id),
            length=num_tokens,
            completed=True,
            request_id=request_id,
        )
        self.node_states[node_id].completed = True
        print(node_id, self.node_states[node_id])
