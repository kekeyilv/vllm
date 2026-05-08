# SPDX-License-Identifier: Apache-2.0
"""DAG topology specification for agent workflows."""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Set

from vllm.entrypoints.chat_utils import ChatCompletionMessageParam


@dataclass
class DAGNode:
    """A node in the agent workflow DAG."""

    node_id: str
    prompt: str | list[ChatCompletionMessageParam]
    parents: List[str] = field(default_factory=list)


@dataclass
class DAGTopology:
    """DAG topology for an agent workflow."""

    nodes: Dict[str, DAGNode] = field(default_factory=dict)

    def add_node(self, node: DAGNode) -> None:
        self.nodes[node.node_id] = node

    def get_node(self, node_id: str) -> DAGNode:
        return self.nodes[node_id]

    def topo_sort(self) -> List[str]:
        """Kahn's algorithm for topological sort."""
        in_degree = {nid: 0 for nid in self.nodes}
        children: Dict[str, List[str]] = {nid: [] for nid in self.nodes}
        for nid, node in self.nodes.items():
            for pid in node.parents:
                children[pid].append(nid)
                in_degree[nid] += 1

        queue = deque([nid for nid, d in in_degree.items() if d == 0])
        order: List[str] = []
        while queue:
            nid = queue.popleft()
            order.append(nid)
            for cid in children[nid]:
                in_degree[cid] -= 1
                if in_degree[cid] == 0:
                    queue.append(cid)
        return order

    def ancestors(self, node_id: str) -> Set[str]:
        """Compute transitive closure (all ancestors) via DFS."""
        result: Set[str] = set()

        def dfs(nid: str) -> None:
            for pid in self.nodes[nid].parents:
                if pid not in result:
                    result.add(pid)
                    dfs(pid)

        dfs(node_id)
        return result

    def is_merge_node(self, node_id: str) -> bool:
        return len(self.nodes[node_id].parents) > 1

    def is_fork_point(self, node_id: str) -> bool:
        return sum(1 for n in self.nodes.values() if node_id in n.parents) > 1
