# SPDX-License-Identifier: Apache-2.0
"""DAGContext: per-request metadata that flows through the V1 engine pipeline."""

from dataclasses import dataclass, field
from typing import Dict, List

from vllm.v1.dag.session import DAGSession


@dataclass
class DAGContext:
    """Attached to a request to enable DAG-aware position assignment.

    Serializable (plain Python types only) so it can transit the ZMQ boundary
    between EngineCore and the model-runner worker process.
    """

    node_id: str
    dag_session: DAGSession
    # Absolute RoPE start position for this node's first token.
    position_offset: int
    # Per-group ancestor block IDs to pass as pre-computed blocks.
    inherited_block_ids: List[List[int]] = field(default_factory=list)
    is_merge: bool = False
    # tail-alignment deltas: {parent_node_id: shift}
    tail_deltas: Dict[str, int] = field(default_factory=dict)
    # Total number of tokens in ancestors' prompt and output
    num_inherited_tokens: int = 0
