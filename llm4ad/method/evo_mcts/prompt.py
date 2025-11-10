# This file is for defining prompts for Evo-MCTS

from __future__ import annotations
import copy
from typing import List
from ...base import Function, Program

class EvoMCTSPrompt:

    @staticmethod
    def _format_functions_with_code(functions: List[Function]) -> str:
        """Helper to format a list of functions with their code bodies."""
        formatted_str = ""
        for i, func in enumerate(functions):
            score = getattr(func, 'score', 'N/A')
            score_str = f"{-score:.4f}" if isinstance(score, float) else "N/A"
            
            formatted_str += f"### Algorithm {i+1}\n"
            formatted_str += f"Description: {getattr(func, 'algorithm', 'N/A')}\n"
            formatted_str += f"Score: {score_str}\n" # Presenting score as lower-is-better
            formatted_str += f"Code:\n```python\n{str(func)}\n```\n\n"
        return formatted_str
    
    @staticmethod
    def _get_empty_function_template(function_to_evolve: Function) -> str:
        temp_func = copy.deepcopy(function_to_evolve)
        temp_func.body = ''
        return str(temp_func)

    @staticmethod
    def get_prompt_reflection_depth_adaptive(task_description: str, worse_func: Function, better_func: Function, depth: int, max_depth: int) -> str:
        depth_level = "Deep"
        if depth <= max_depth * 0.3:
            depth_level = "Shallow"
        elif depth <= max_depth * 0.6:
            depth_level = "Medium"

        analysis_focus = {
            "Shallow": "High-level algorithmic approach, overall structure, and main logic flow.",
            "Medium": "Specific implementation choices, library function calls, data structures, and key parameter values.",
            "Deep": "Low-level details, mathematical formulas, numerical precision, and subtle logical conditions."
        }

        prompt = (
            f"You are an expert programming analyst. Your task is to analyze two algorithms and generate depth-specific improvement strategies.\n\n"
            f"## Context\n"
            f"Problem: {task_description}\n"
            f"Search Depth: {depth}/{max_depth} ({depth_level})\n\n"
            f"## Algorithms for Analysis\n"
            f"### Algorithm 1 (Worse Score)\n{EvoMCTSPrompt._format_functions_with_code([worse_func])}"
            f"### Algorithm 2 (Better Score)\n{EvoMCTSPrompt._format_functions_with_code([better_func])}"
            f"\n## Your Task\n"
            "Follow these steps:\n"
            f"1.  **Analyze Differences**: Focusing on **{analysis_focus[depth_level]}**, what is the single most important difference that makes Algorithm 2 superior to Algorithm 1?\n"
            f"2.  **Generate Principles**: Based on your analysis, generate a concise, transferable principle for improvement.\n\n"
            f"## Output Format\n"
            f"Provide only the single improvement principle. Do not include any other explanatory text."
        )
        return prompt

    @staticmethod
    def get_prompt_i2_domain_knowledge_generation(task_description: str, function_to_evolve: Function) -> str:
        """Generates a prompt to create an initial solution from scratch using domain expertise."""
        empty_func_template = EvoMCTSPrompt._get_empty_function_template(function_to_evolve)
        prompt = (
            f"You are an expert algorithm designer with deep domain knowledge in the relevant field. "
            f"Your task is to provide a complete, high-quality Python function to solve the given problem.\n\n"
            f"## Context\n"
            f"Problem: {task_description}\n\n"
            f"## Your Task\n"
            "Follow these steps:\n"
            "1. **Propose an Algorithm**: Briefly describe the core idea of your proposed algorithm based on your expertise.\n"
            "2. **Describe the Implementation**: Summarize your implementation's logic in one sentence, enclosed in curly braces like {{your thought}}.\n"
            f"3. **Implement the Code**: Provide the complete Python code for your algorithm in the function template below.\n\n"
            "```python\n"
            f"{empty_func_template}\n"
            "```\n"
            "Provide only the completed summary (step 2) and code (step 3)."
        )
        return prompt

    @staticmethod
    def get_prompt_i3_seed_inspired_generation(task_description: str, function_to_evolve: Function) -> str:
        """Generates a prompt to create an initial solution inspired by general principles, not a specific seed."""
        empty_func_template = EvoMCTSPrompt._get_empty_function_template(function_to_evolve)
        prompt = (
            f"You are a creative and experienced algorithm designer. Your task is to brainstorm and implement a novel solution to the problem described below.\n\n"
            f"## Context\n"
            f"Problem: {task_description}\n\n"
            f"## Your Task\n"
            "Think about common strategies, data structures, and potential edge cases related to this problem. "
            "Your goal is to create a robust and effective initial solution.\n"
            "Follow these steps:\n"
            "1. **Brainstorm Ideas**: Consider several alternative approaches to solve this problem.\n"
            "2. **Select and Describe**: Choose the most promising approach and summarize its core logic in one sentence, enclosed in curly braces like {{your thought}}.\n"
            f"3. **Implement Code**: Implement your chosen algorithm in the function template below.\n\n"
            "```python\n"
            f"{empty_func_template}\n"
            "```\n"
            "Provide only the completed summary (step 2) and code (step 3)."
        )
        return prompt

    @staticmethod
    def get_prompt_i4_seed_based_generation(task_description: str, seed_func: Function, function_to_evolve: Function, depth: int) -> str:
        depth_word_map = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth"}
        depth_word = depth_word_map.get(depth, f"{depth}th")
        empty_func_template = EvoMCTSPrompt._get_empty_function_template(function_to_evolve)
        
        prompt = (
            f"You are an expert algorithm designer. Your task is to create a new algorithm inspired by a provided seed.\n\n"
            f"## Context\n"
            f"Problem: {task_description}\n"
            f"This is your {depth_word} attempt to create a novel solution.\n\n"
            f"## Seed Algorithm\n{EvoMCTSPrompt._format_functions_with_code([seed_func])}"
            f"\n## Your Task\n"
            "Follow these steps:\n"
            "1. **Analyze Seed**: Identify one key idea from the seed algorithm.\n"
            "2. **Propose a Novel Idea**: Brainstorm a new, different approach to solve the problem, potentially improving upon the seed's idea.\n"
            "3. **Describe New Algorithm**: Summarize your new algorithm's core logic in one sentence, enclosed in curly braces like {{your thought}}.\n"
            f"4. **Implement Code**: Implement your new algorithm in the function template below.\n\n"
            "```python\n"
            f"{empty_func_template}\n"
            "```\n"
            "Provide only the completed summary (step 3) and code (step 4)."
        )
        return prompt
    
    @staticmethod
    def get_prompt_from_reflection(task_description: str, reflection: str, context_funcs: List[Function], func_to_evolve: Function, instruction: str) -> str:
        """A generic, guided, multi-step template for operators that consume reflections."""
        empty_func_template = EvoMCTSPrompt._get_empty_function_template(func_to_evolve)
        context_str = "## Contextual Algorithms\n" + EvoMCTSPrompt._format_functions_with_code(context_funcs) if context_funcs else ""

        prompt = (
            f"You are an expert algorithm designer. Your task is to evolve a new algorithm based on a guiding reflection.\n\n"
            f"## Context\n"
            f"Problem: {task_description}\n\n"
            "### Guiding Reflection\n"
            f"\"{reflection}\"\n\n"
            f"{context_str}"
            f"\n## Your Task\n"
            "Follow these steps:\n"
            f"1. **Formulate a Plan**: Based on the Guiding Reflection, formulate a concrete plan to {instruction}.\n"
            "2. **Describe New Algorithm**: Summarize your new algorithm's core logic in one sentence, enclosed in curly braces like {{your thought}}.\n"
            f"3. **Implement Code**: Implement your new algorithm in the function template below.\n\n"
            "```python\n"
            f"{empty_func_template}\n"
            "```\n"
            "Provide only the completed summary (step 2) and code (step 3)."
        )
        return prompt

    @staticmethod
    def get_prompt_e3_crossover(task_description: str, reflection: str, parent1: Function, parent2: Function, func_to_evolve: Function) -> str:
        instruction = "create a new algorithm that combines the best ideas from the two parent algorithms"
        return EvoMCTSPrompt.get_prompt_from_reflection(
            task_description, reflection, [parent1, parent2], func_to_evolve, instruction
        )

    @staticmethod
    def get_prompt_m3_mutation(task_description: str, reflection: str, father: Function, context_parents: List[Function], func_to_evolve: Function) -> str:
        instruction = "Your goal is to mutate the 'Father' algorithm. Use the collective insights from the Reflection (derived from a group of other algorithms) to create a new, improved version."
        return EvoMCTSPrompt.get_prompt_from_reflection(
            task_description, reflection, context_parents + [father], func_to_evolve, instruction
        )

    @staticmethod
    def get_prompt_m6_mutation(task_description: str, reflection: str, parent: Function, elite: Function, func_to_evolve: Function) -> str:
        instruction = "Your goal is to mutate the Parent Algorithm. Use the insights from the Reflection and the strategy of the Elite Algorithm to create a new, improved version."
        return EvoMCTSPrompt.get_prompt_from_reflection(
            task_description, reflection, [parent, elite], func_to_evolve, instruction
        )

    @staticmethod
    def get_prompt_m7_elite_guided_mutation(task_description: str, reflection: str, elite: Function, parents: List[Function], func_to_evolve: Function) -> str:
        instruction = "Your goal is to create a new algorithm. Synergize the strengths of the 'Elite Algorithm' with promising features from the 'Other Algorithms' based on the 'Guiding Reflection'."
        return EvoMCTSPrompt.get_prompt_from_reflection(
            task_description, reflection, [elite] + parents, func_to_evolve, instruction
        )

    @staticmethod
    def get_prompt_s2_improvement(task_description: str, reflection: str, father: Function, path: List[Function], func_to_evolve: Function) -> str:
        instruction = "Your goal is to perform a significant leap in improvement. Use the deep insights from the entire evolutionary path's Reflection to create a substantially new and better version of the current algorithm."
        return EvoMCTSPrompt.get_prompt_from_reflection(
            task_description, reflection, path, func_to_evolve, instruction
        )

    @staticmethod
    def get_prompt_s3_path_summary_improvement(task_description: str, reflection: str, father: Function, path: List[Function], func_to_evolve: Function) -> str:
        instruction = "Your goal is to propose a new direction. Analyze the trend from the evolutionary path's Reflection, identify its limitations, and create a divergent but potentially more effective algorithm."
        return EvoMCTSPrompt.get_prompt_from_reflection(
            task_description, reflection, path, func_to_evolve, instruction
        )
