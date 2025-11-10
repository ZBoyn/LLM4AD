from __future__ import annotations
import time
import traceback
import numpy as np
from typing import List, Dict, Any, Optional
import random

from .evolution import Evolution
from ...base import Function, TextFunctionProgramConverter
from .population import Population

class InterfaceEC:
    def __init__(self,
                 evolution_module: Evolution,
                 evaluation_executor,
                 evaluator,
                 population: Population,
                 debug_mode: bool = False,
                 max_retries: int = 3):

        # evolution module
        self.evol = evolution_module
        self._evaluation_executor = evaluation_executor
        self._evaluator = evaluator

        self._population = population
        self._debug_mode = debug_mode

        self._max_retries = max_retries
        self.elite_offspring: Optional[Function] = None

    def update_elite_offspring(self, offspring_func: Function):
        if offspring_func is None:
            return
        
        if self.elite_offspring is None or offspring_func.score > self.elite_offspring.score:
            self.elite_offspring = offspring_func
            if self._debug_mode:
                print(f"    > New elite found! Score: {self.elite_offspring.score:.4f}")

    def _get_alg(self, pop: List[Function], operator: str, father: Function = None, depth: int = None, rechat: bool = False) -> Dict[str, Any]:
        self.evol.rechat = rechat
        offspring = {
            'reflection': None,
            'algorithm': None,
            'thought': None,
            'code': None,
            'objective': None,
            'operator': operator,
            'depth': depth,
            'rechat': rechat,
            'func_obj': None
        }

        if operator == "i2":
            self.evol.i2_domain_knowledge_generation(offspring)
        elif operator == "i3":
            self.evol.i3_seed_inspired_generation(offspring)
        elif operator == "i4":
            seed_func = father
            self.evol.i4_seed_generation(offspring, seed_func, depth)
        elif operator == "e3":
            if len(pop) < 2:
                if self._debug_mode: print("    > Skipping 'e3': not enough parents in context.")
                return offspring
            other_parent = random.choice([p for p in pop if p != father])
            self.evol.e3_crossover(offspring, other_parent, father, depth=depth)
        elif operator == "m3":
            num_parents = min(len(pop), 5)
            context_parents = [p for p in pop if p != father]
            if not context_parents:
                if self._debug_mode: print("    > Skipping 'm3': no other parents in context.")
                return offspring
            selected_parents = random.sample(context_parents, min(len(context_parents), num_parents))
            self.evol.m3_mutation(offspring, father, selected_parents, depth=depth)
        elif operator == "m6":
            if not self.elite_offspring:
                if self._debug_mode: print("    > Skipping 'm6': elite not yet established.")
                return offspring
            candidates = [ind for ind in self._population if ind != self.elite_offspring]
            if not candidates:
                if self._debug_mode: print("    > Skipping 'm6': no other candidates to mutate.")
                return offspring
            parent_to_mutate = random.choice(candidates)
            self.evol.m6_mutation(offspring, parent_to_mutate, self.elite_offspring, depth=depth)
        elif operator == "m7":
            if not self.elite_offspring:
                if self._debug_mode: print("    > Skipping 'm7': elite not yet established.")
                return offspring
            num_parents = min(len(self._population), 3)
            candidates = [p for p in self._population if p != self.elite_offspring]
            if not candidates:
                 if self._debug_mode: print("    > Skipping 'm7': no other candidates for context.")
                 return offspring
            selected_parents = random.sample(candidates, min(len(candidates), num_parents))
            self.evol.m7_elite_guided_mutation(offspring, self.elite_offspring, selected_parents, depth=depth)
        elif operator == "s2":
            self.evol.s2_improvement(offspring, father, pop, depth=depth)
        elif operator == "s3":
            self.evol.s3_path_summary_improvement(offspring, father, pop, depth=depth)
        else:
            if self._debug_mode:
                print(f"  > WARNING: Evolution operator '{operator}' is not implemented!")

        offspring['timestamp'] = time.time()
        return offspring

    def evolve_algorithm(self,
                         pop: List[Function],
                         node: Function,
                         operator: str,
                         iteration: int,
                         depth: int) -> Optional[Function]:
        rechat = False
        for i in range(self._max_retries):
            try:
                start_time = time.time()
                offspring_dict = self._get_alg(pop, operator, father=node, depth=depth, rechat=rechat)
                sample_time = time.time() - start_time
                
                func_obj = offspring_dict.get('func_obj')
                if func_obj is None:
                    print(f"    > Generation failed for '{operator}'. Retrying ({i+1}/{self._max_retries})...")
                    rechat = True
                    continue
                
                is_duplicate_code = any(str(func_obj) == str(existing_func) for existing_func in self._population)
                if is_duplicate_code:
                    if self._debug_mode: print(f"    > Skipping evaluation for duplicate code from '{operator}'.")
                    return None

                program = TextFunctionProgramConverter.function_to_program(func_obj, self.evol._template_program)
                if not program:
                    if self._debug_mode: print("    > Failed to convert generated function to program.")
                    rechat = True
                    continue

                score, exec_time = self._evaluation_executor.submit(self._evaluator.evaluate_program_record_time, program).result()

                if score is None:
                    print(f"    > Evaluation failed (score is None) for '{operator}'. Retrying ({i+1}/{self._max_retries})...")
                    rechat = True
                    continue

                is_duplicate_obj = score in self._population._objective_history
                if is_duplicate_obj:
                    if self._debug_mode: print(f"    > Skipping duplicate objective score: {score:.4f}")
                    return None

                func_obj.score = np.round(score, 5)
                func_obj.sample_time = sample_time
                func_obj.evaluate_time = exec_time
                
                return func_obj

            except Exception:
                if self._debug_mode:
                    print(f"  > An exception occurred during evolution with operator '{operator}':")
                    traceback.print_exc()
                rechat = True
                continue

        print(f"  > Evolution failed for operator '{operator}' after {self._max_retries} retries.")
        return None

    def generate_reflection(self, new_func: Function, reference_func: Function, depth: int, max_depth: int):
        if self._debug_mode:
            print(f"    > Generating reflection for new function at depth {depth}...")

        try:
            if new_func.score > reference_func.score:
                better_func, worse_func = new_func, reference_func
            else:
                better_func, worse_func = reference_func, new_func

            reflection_text = self.evol.R_reflect(
                worse_func=worse_func, better_func=better_func,
                depth=depth, max_depth=max_depth
            )

            if reflection_text:
                new_func.reflection = reflection_text
                if self._debug_mode:
                    print(f"      > Reflection generated (first 100 chars): {reflection_text[:100]}...")
            else:
                if self._debug_mode: print("      > Failed to generate reflection.")
                new_func.reflection = "Reflection generation failed."

        except Exception:
            if self._debug_mode:
                print("      > An exception occurred during reflection generation.")
                traceback.print_exc()
            new_func.reflection = "Reflection generation failed due to an exception."
