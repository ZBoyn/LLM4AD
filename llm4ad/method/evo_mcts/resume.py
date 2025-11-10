from __future__ import annotations
import json
import os
import traceback
from typing import TYPE_CHECKING

from .mcts import MCTSNode
from ...base import TextFunctionProgramConverter

if TYPE_CHECKING:
    from .evo_mcts import EvoMCTS

def resume_evomcts(method: "EvoMCTS", path: str) -> bool:
    """
    Restores the state of an EvoMCTS run from saved files in the experiment directory.
    """
    print(f"--- Attempting to resume EvoMCTS from path: {path} ---")
    
    tree_dir = os.path.join(path, 'mcts_trees')
    if not os.path.isdir(tree_dir):
        print(f"  > Resume failed: Directory not found: {tree_dir}")
        return False

    tree_files = [f for f in os.listdir(tree_dir) if f.startswith('mcts_tree_iter_') and f.endswith('.json')]
    if not tree_files:
        print(f"  > Resume failed: No tree files found in {tree_dir}")
        return False

    # Find the latest iteration file to resume from
    latest_iter = -1
    latest_file = None
    for f in tree_files:
        try:
            iter_num = int(f.replace('mcts_tree_iter_', '').replace('.json', ''))
            if iter_num > latest_iter:
                latest_iter = iter_num
                latest_file = f
        except ValueError:
            continue

    if not latest_file:
        print("  > Resume failed: Could not determine the latest tree file.")
        return False

    file_path = os.path.join(tree_dir, latest_file)
    print(f"  > Found latest state at iteration {latest_iter}. Restoring from {file_path}...")
    
    try:
        with open(file_path, 'r') as f:
            tree_data = json.load(f)
        
        # Rebuild the tree and get a flat list of all nodes
        new_root, all_nodes = _restore_mcts_tree(tree_data, method._mcts)
        method._mcts.root = new_root
        
        # Repopulate the population and find the elite
        for node in all_nodes:
            if node.raw_info:
                # Use the public interface of the population manager
                method._population.add(node.raw_info) 
                method.interface_ec.update_elite_offspring(node.raw_info)
        
        # Update main method's state to resume
        method._iteration = latest_iter
        method._tot_sample_nums = sum(node.visits for node in all_nodes) 
        
        print(f"  > Restore successful. Resuming at iteration {method._iteration + 1}.")
        return True
    except Exception:
        print(f"  > ERROR: Failed to load or parse MCTS tree from {file_path}")
        traceback.print_exc()
        return False

def _restore_mcts_tree(node_data: dict, mcts_instance, parent: MCTSNode = None) -> tuple[MCTSNode, list[MCTSNode]]:
    """Recursively reconstructs the MCTS tree from serialized data."""
    raw_info_obj = None
    if node_data.get("raw_info"):
        ri = node_data["raw_info"]
        # The code is already a string, so we can convert it back to a Function object
        raw_info_obj = TextFunctionProgramConverter.text_to_function(ri["code"])
        raw_info_obj.score = ri["score"]
        raw_info_obj.algorithm = ri["algorithm"]
        raw_info_obj.operator = ri["operator"]
        raw_info_obj.reflection = ri["reflection"]

    node = MCTSNode(
        algorithm=raw_info_obj.algorithm if raw_info_obj else "Root",
        code=str(raw_info_obj) if raw_info_obj else "Root",
        obj=raw_info_obj.score if raw_info_obj else 0,
        parent=parent,
        depth=node_data["depth"],
        visit=node_data["visits"],
        Q=node_data["Q"],
        raw_info=raw_info_obj
    )
    
    # Update the global rank list in the MCTS instance
    if raw_info_obj and raw_info_obj.score is not None:
        mcts_instance.rank_list.append(node.Q)

    all_nodes = [node]
    for child_data in node_data.get("children", []):
        child_node, child_all_nodes = _restore_mcts_tree(child_data, mcts_instance, parent=node)
        node.add_child(child_node)
        all_nodes.extend(child_all_nodes)
        if parent is None: # This is the root node
            node.subtree.extend(child_all_nodes)

    return node, all_nodes
