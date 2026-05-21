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
    padding: int = 0  # Number of padding tokens for block aligmnet
    position_offset: int = 0  # RoPE position offset (first token position)
    padding_offset: int = 0  # Total padding for block alignment
    length: int = 0  # Number of tokens in this node
    request_id: str = ""
    data: List[int] = field(default_factory=list)  # prompt + output + padding


@dataclass
class AncestorState:
    """
    Data of ancestors for position corrections and KV Cache indexing
    """

    node_id: str
    request_id: str
    position_offset: int
    padding_offset: int
    padding: int
    length: int  # number of tokens
    position_delta: int  # ancestor position correction delta


@dataclass
class DAGContext:
    """Attached to a request to enable DAG-aware position assignment.

    Serializable (plain Python types only) so it can transit the ZMQ boundary
    between EngineCore and the model-runner worker process.
    """

    node_id: str
    # Total padding for block alignment
    padding_offset: int
    # Absolute RoPE start position for this node's first token.
    position_offset: int
    # States of ancestors nodes
    ancestor_states: List[AncestorState] = field(default_factory=list)


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
        self._ancestors_cache: Dict[str, List[str]] = {}

        # To calculate position corrections deltas,
        # every node's ancestors must be arranged by their topological order
        topo_ids: dict[str, int] = {}
        for topo_id, nid in enumerate(dag.topo_sort()):
            topo_ids[nid] = topo_id
            ancestors = list(dag.ancestors(nid))
            # A node's ancestors appear before the node,
            # so their topo_ids have been determined.
            ancestors.sort(key=lambda idx: topo_ids[idx])
            self._ancestors_cache[nid] = ancestors

    def get_context(self, node_id: str) -> DAGContext:
        ancestors = self._ancestors_cache[node_id]
        states = []
        current_pos = 0
        padding_offset = 0
        for anc_id in ancestors:
            node_state = self.node_states[anc_id]
            states.append(
                AncestorState(
                    node_id=anc_id,
                    position_offset=node_state.position_offset,
                    padding_offset=node_state.padding_offset,
                    request_id=node_state.request_id,
                    length=node_state.length,
                    padding=node_state.padding,
                    position_delta=current_pos - node_state.position_offset,
                )
            )
            current_pos += node_state.length
            padding_offset += node_state.padding

        return DAGContext(
            node_id=node_id,
            position_offset=current_pos,
            ancestor_states=states,
            padding_offset=padding_offset,
        )

    def register_completion(
        self,
        node_id: str,
        request_id: str,
        block_size: int,
        prompt_tokens: list[int],
        output_tokens: list[int],
    ) -> None:
        """Register a completed node's state for use by downstream nodes."""
        length = len(prompt_tokens) + len(output_tokens)
        padding = (block_size - length % block_size) % block_size
        context = self.get_context(node_id)
        self.node_states[node_id] = NodeState(
            position_offset=context.position_offset,
            padding_offset=context.padding_offset,
            length=length,
            completed=True,
            request_id=request_id,
            padding=padding,
            data=prompt_tokens + output_tokens + [0] * padding,
        )
        self.node_states[node_id].completed = True
