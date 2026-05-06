# SPDX-License-Identifier: Apache-2.0
"""DAG session manager - tracks KV cache state across a DAG execution."""
from dataclasses import dataclass, field
from typing import Dict, List, Set

import torch

from vllm.v1.dag.topology import DAGTopology


@dataclass
class NodeState:
    """Runtime state for a completed DAG node."""
    offset: int          # RoPE position offset (first token position)
    length: int          # Number of tokens in this node
    block_ids: List[List[int]]  # Per-group block IDs


class DAGSession:
    """Manages KV cache and position state for a single DAG execution."""

    def __init__(self, dag: DAGTopology) -> None:
        self.dag = dag
        self.node_states: Dict[str, NodeState] = {}
        self._ancestors_cache: Dict[str, Set[str]] = {}

        for nid in dag.topo_sort():
            self._ancestors_cache[nid] = dag.ancestors(nid)

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

    def compute_tail_deltas(self, merge_node_id: str) -> Dict[str, int]:
        """For a merge node, compute tail-alignment shifts per parent.

        Returns {parent_id: delta} where delta = L_max - len(parent).
        Shorter branches are shifted so their tails align with the longest.
        """
        node = self.dag.get_node(merge_node_id)
        assert len(node.parents) > 1, "Not a merge node"

        L_max = max(self.node_states[pid].length for pid in node.parents)
        return {
            pid: L_max - self.node_states[pid].length
            for pid in node.parents
        }

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
            if anc_id not in self.node_states:
                continue
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
        block_ids: List[List[int]],
    ) -> None:
        """Register a completed node's state for use by downstream nodes."""
        offset = self.compute_offset(node_id)
        self.node_states[node_id] = NodeState(
            offset=offset,
            length=num_tokens,
            block_ids=block_ids,
        )
