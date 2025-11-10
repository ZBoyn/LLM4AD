from __future__ import annotations
import traceback
from typing import List, Dict, Any, Optional

from .prompt import EvoMCTSPrompt
from ...base import LLM, Function, Program, TextFunctionProgramConverter

class Evolution:
    def __init__(self,
                 llm: LLM,
                 sampler,
                 template_program: Program,
                 function_to_evolve: Function,
                 task_description: str,
                 debug_mode: bool = False):
        self._llm = llm
        self._sampler = sampler
        self._template_program = template_program
        self._function_to_evolve = function_to_evolve
        self._task_description = task_description
        self._debug_mode = debug_mode
        self.rechat = False

    def _get_alg(self, prompt: str, operator: str) -> Optional[Function]:
        if self._debug_mode:
            print(f"    > Operator '{operator}' prompt sent to LLM.")

        try:
            thought, func_obj = self._sampler.get_thought_and_function(self._task_description, prompt)
            
            if func_obj:
                func_obj.operator = operator
                func_obj.algorithm = thought
                return func_obj
                
            if self._debug_mode:
                print(f"    > Sampler failed to return a valid function for operator '{operator}'.")
            return None
        except Exception:
            if self._debug_mode:
                print(f"    > An exception occurred during LLM sampling for operator '{operator}'.")
                traceback.print_exc()
            return None

    def i2_domain_knowledge_generation(self, offspring: Dict[str, Any]):
        prompt = EvoMCTSPrompt.get_prompt_i2_domain_knowledge_generation(
            self._task_description, self._function_to_evolve
        )
        new_func = self._get_alg(prompt, "i2")
        if new_func:
            offspring['thought'] = new_func.algorithm
            offspring['code'] = str(new_func)
            offspring['func_obj'] = new_func

    def i3_seed_inspired_generation(self, offspring: Dict[str, Any]):
        """Generates a solution from scratch using general brainstorming."""
        prompt = EvoMCTSPrompt.get_prompt_i3_seed_inspired_generation(
            self._task_description, self._function_to_evolve
        )
        new_func = self._get_alg(prompt, "i3")
        if new_func:
            offspring['thought'] = new_func.algorithm
            offspring['code'] = str(new_func)
            offspring['func_obj'] = new_func

    def i4_seed_generation(self, offspring: Dict[str, Any], seed_func: Function, depth: int):
        # root.raw_info = seed_func
        prompt = EvoMCTSPrompt.get_prompt_i4_seed_based_generation(
            self._task_description, seed_func, self._function_to_evolve, depth
        )
        new_func = self._get_alg(prompt, "i4")
        if new_func:
            offspring['thought'] = new_func.algorithm
            offspring['code'] = str(new_func)
            offspring['func_obj'] = new_func

    def e3_crossover(self, offspring: Dict[str, Any], parent1: Function, parent2: Function, depth: Optional[int] = None):
        """Consumes a reflection to perform crossover between two parents."""
        reflection = getattr(parent1, 'reflection', '') + "\n" + getattr(parent2, 'reflection', '')
        print(f"   >  reflection: {reflection}")
        if not reflection.strip(): reflection = "No reflection available."
        prompt = EvoMCTSPrompt.get_prompt_e3_crossover(
            self._task_description, reflection, parent1, parent2, self._function_to_evolve
        )
        new_func = self._get_alg(prompt, "e3")
        if new_func:
            offspring['thought'] = new_func.algorithm
            offspring['code'] = str(new_func)
            offspring['func_obj'] = new_func

    def m3_mutation(self, offspring: Dict[str, Any], father: Function, parents: List[Function], depth: Optional[int] = None):
        """Consumes a collective reflection from a group to mutate a father."""
        reflections = [getattr(p, 'reflection', '') for p in parents]
        reflection = "\n---\n".join(filter(None, reflections))
        if not reflection: reflection = "No collective reflection available."

        prompt = EvoMCTSPrompt.get_prompt_m3_mutation(
            self._task_description, reflection, father, parents, self._function_to_evolve
        )
        new_func = self._get_alg(prompt, "m3")
        if new_func:
            offspring['thought'] = new_func.algorithm
            offspring['code'] = str(new_func)
            offspring['func_obj'] = new_func

    def m6_mutation(self, offspring: Dict[str, Any], parent: Function, elite: Function, depth: Optional[int] = None):
        """Consumes a reflection from the elite to mutate a parent."""
        reflection = getattr(elite, 'reflection', "No reflection available on elite.")
        prompt = EvoMCTSPrompt.get_prompt_m6_mutation(
            self._task_description, reflection, parent, elite, self._function_to_evolve
        )
        new_func = self._get_alg(prompt, "m6")
        if new_func:
            offspring['thought'] = new_func.algorithm
            offspring['code'] = str(new_func)
            offspring['func_obj'] = new_func

    def m7_elite_guided_mutation(self, offspring: Dict[str, Any], elite: Function, parents: List[Function], depth: Optional[int] = None):
        """Consumes reflections from elite and other parents to create a new solution."""
        reflections = [getattr(p, 'reflection', '') for p in parents] + [getattr(elite, 'reflection', '')]
        reflection = "\n---\n".join(filter(None, reflections))
        if not reflection: reflection = "No collective reflection available."

        prompt = EvoMCTSPrompt.get_prompt_m7_elite_guided_mutation(
            self._task_description, reflection, elite, parents, self._function_to_evolve
        )
        new_func = self._get_alg(prompt, "m7")
        if new_func:
            offspring['thought'] = new_func.algorithm
            offspring['code'] = str(new_func)
            offspring['func_obj'] = new_func

    def s2_improvement(self, offspring: Dict[str, Any], father: Function, path: List[Function], depth: Optional[int] = None):
        """Consumes a reflection from an evolutionary path to improve the father."""
        reflections = [getattr(p, 'reflection', '') for p in path]
        reflection = "\n---\n".join(filter(None, reflections))
        if not reflection: reflection = "No path reflection available."

        prompt = EvoMCTSPrompt.get_prompt_s2_improvement(
            self._task_description, reflection, father, path, self._function_to_evolve
        )
        new_func = self._get_alg(prompt, "s2")
        if new_func:
            offspring['thought'] = new_func.algorithm
            offspring['code'] = str(new_func)
            offspring['func_obj'] = new_func
        
    def s3_path_summary_improvement(self, offspring: Dict[str, Any], father: Function, path: List[Function], depth: Optional[int] = None):
        """Consumes a reflection from an evolutionary path to propose a new direction."""
        reflections = [getattr(p, 'reflection', '') for p in path]
        reflection = "\n---\n".join(filter(None, reflections))
        if not reflection: reflection = "No path reflection available."

        prompt = EvoMCTSPrompt.get_prompt_s3_path_summary_improvement(
            self._task_description, reflection, father, path, self._function_to_evolve
        )
        new_func = self._get_alg(prompt, "s3")
        if new_func:
            offspring['thought'] = new_func.algorithm
            offspring['code'] = str(new_func)
            offspring['func_obj'] = new_func
            
    # --- Reflection Operator ---
    def R_reflect(self, worse_func: Function, better_func: Function, depth: int, max_depth: int) -> str | None:
        """Analyzes two functions and produces a depth-adaptive reflection."""
        prompt = EvoMCTSPrompt.get_prompt_reflection_depth_adaptive(
            self._task_description, worse_func, better_func, depth, max_depth
        )
        try:
            reflection_text = self._sampler.get_reflection(prompt)
            return reflection_text
        except Exception:
            if self._debug_mode:
                print(f"    > An exception occurred during reflection LLM call.")
                traceback.print_exc()
            return None

    def R_reflect_on_list(self, functions: List[Function], reference_func: Function, depth: Optional[int] = None) -> str | None:
        """
        Analyzes a list of functions in relation to a reference function and produces a reflection.
        """
        prompt = EvoMCTSPrompt.get_prompt_reflection_on_list(self._task_description, functions, reference_func)
        reflection_text = self._sampler.get_reflection(prompt)
        return reflection_text
