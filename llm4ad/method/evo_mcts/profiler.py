from __future__ import annotations
import json
import os
import traceback
from typing import Optional

from ...tools.profiler import ProfilerBase
from .mcts import MCTSNode

class EvoMCTSProfiler(ProfilerBase):
    """
    A specialized profiler for EvoMCTS that handles logging, checkpointing,
    and serialization of the MCTS tree to disk.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.exp_path:
            self._tree_dir = os.path.join(self.exp_path, 'mcts_trees')
            os.makedirs(self._tree_dir, exist_ok=True)
        else:
            self._tree_dir = None

    def save_mcts_tree(self, root_node: MCTSNode, iteration: int):
        """Serializes the current MCTS tree and saves it to a JSON file."""
        if not self._tree_dir:
            if self.debug_mode:
                print("  > Profiler `exp_path` not set. Skipping MCTS tree save.")
            return

        tree_info = self._serialize_node(root_node)
        file_path = os.path.join(self._tree_dir, f"mcts_tree_iter_{iteration}.json")

        try:
            with open(file_path, 'w') as f:
                json.dump(tree_info, f, indent=4, default=str)
            if self.debug_mode:
                print(f"  > MCTS tree saved to {file_path}")
        except Exception:
            print(f"  > ERROR: Failed to save MCTS tree to {file_path}")
            if self.debug_mode:
                traceback.print_exc()

    def _serialize_node(self, node: MCTSNode) -> dict:
        """Recursively serializes an MCTSNode and its children."""
        serializable_raw_info = None
        if node.raw_info:
            serializable_raw_info = {
                "algorithm": node.raw_info.algorithm,
                "code": str(node.raw_info),
                "score": node.raw_info.score,
                "operator": getattr(node.raw_info, 'operator', None),
                "reflection": getattr(node.raw_info, 'reflection', None)
            }

        return {
            "depth": node.depth,
            "visits": node.visits,
            "Q": node.Q,
            "raw_info": serializable_raw_info,
            "children": [self._serialize_node(child) for child in node.children]
        }
