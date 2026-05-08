# SPDX-License-Identifier: Apache-2.0
"""DAG session manager - tracks KV cache state across a DAG execution."""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Set

import torch

from vllm.v1.dag.topology import DAGTopology


@dataclass
class NodeState:
    """Runtime state for a completed DAG node."""

    completed: bool = False
    offset: int = 0  # RoPE position offset (first token position)
    length: int = 0  # Number of tokens in this node
    block_ids: List[List[int]] = field(default_factory=list)  # Per-group block IDs

    def submit_blocks(self, block_ids: Iterable[List[int]]):
        self.block_ids.extend(block_ids)


@dataclass
class DAGContext:
    """Attached to a request to enable DAG-aware position assignment.

    Serializable (plain Python types only) so it can transit the ZMQ boundary
    between EngineCore and the model-runner worker process.
    """

    node_id: str
    node_state: NodeState
    # Absolute RoPE start position for this node's first token.
    position_offset: int
    # Per-group ancestor block IDs to pass as pre-computed blocks.
    inherited_block_ids: List[List[int]] = field(default_factory=list)
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
        if node_id not in self.node_states:
            self.node_states[node_id] = NodeState()
        return DAGContext(
            node_id=node_id,
            node_state=self.node_states[node_id],
            position_offset=self.compute_offset(node_id),
            inherited_block_ids=self.get_inherited_block_ids(node_id),
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

    def get_inherited_block_ids(self, node_id: str) -> List[List[int]]:
        """Get deduplicated ancestor block IDs ordered by offset.

        Returns per-group block ID lists for use with KVCacheManager.
        """
        ancestors = self._ancestors_cache[node_id]
        ordered_ancestors = sorted(
            ancestors,
            key=lambda a: self.node_states[a].offset,
        )

        seen: set = set()
        all_blocks: List[int] = []
        for anc_id in ordered_ancestors:
            assert anc_id in self.node_states
            for bid in self.node_states[anc_id].block_ids[0]:
                if bid not in seen:
                    seen.add(bid)
                    all_blocks.append(bid)

        return [all_blocks]

    def get_visible_block_ids(self, node_id: str) -> List[List[int]]:
        """Get block IDs visible to this node (ancestors + own blocks)."""
        inherited = self.get_inherited_block_ids(node_id)
        if node_id in self.node_states:
            own = self.node_states[node_id].block_ids
            return [inherited[g] + own[g] for g in range(len(inherited))]
        return inherited

    def register_completion(
        self,
        node_id: str,
        num_tokens: int,
    ) -> None:
        """Register a completed node's state for use by downstream nodes."""
        if node_id not in self.node_states:
            self.node_states[node_id] = NodeState()
        self.node_states[node_id].offset = self.compute_offset(node_id)
        self.node_states[node_id].length = num_tokens
        self.node_states[node_id].completed = True
