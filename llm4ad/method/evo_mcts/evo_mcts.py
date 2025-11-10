from __future__ import annotations
from typing import Optional, Literal
import concurrent.futures
import numpy as np
import random

from .evolution import Evolution
from .evolution_interface import InterfaceEC
from .mcts import MCTS, MCTSNode
from .sampler import EvoMCTSSampler
from .population import Population
from .profiler import EvoMCTSProfiler
from .resume import resume_evomcts
from ...base import (
    Evaluation, LLM, Function, Program, TextFunctionProgramConverter, SecureEvaluator
)
from ...tools.profiler import ProfilerBase

class EvoMCTS:
    def __init__(self,
                 llm: LLM,
                 evaluation: Evaluation,
                 profiler: ProfilerBase = None,
                 seed_program_str: Optional[str] = None,
                 max_sample_nums: Optional[int] = 100,
                 init_size: Optional[int] = 4,
                 pop_size: Optional[int] = 10,
                 num_evaluators: int = 1,
                 operators: list = None,
                 operator_weights: list = None,
                 alpha: float = 0.5,
                 lambda_0: float = 0.1,
                 max_retries: int = 3,
                 *,
                 resume_mode: bool = False,
                 debug_mode: bool = False,
                 multi_thread_or_process_eval: Literal['thread', 'process'] = 'thread',
                 **kwargs):
        """Evolutionary Monte Carlo Tree Search
        Args:
            llm             : an instance of 'llm4ad.base.LLM', which provides the way to query LLM.
            evaluation      : an instance of 'llm4ad.base.Evaluator', which defines the way to calculate the score of a generated function.
            profiler        : an instance of 'llm4ad.method.eoh.EoHProfiler'. If you do not want to use it, you can pass a 'None'.
                              pass 'None' to disable this termination condition.
            seed_program_str: the seed program to use for the initialization.
            max_sample_nums : terminate after evaluating max_sample_nums functions (no matter the function is valid or not) or reach 'max_generations',
                              pass 'None' to disable this termination condition.
            init_size       : population size, if set to 'None', Evo-MCTS will automatically adjust this parameter.
            pop_size        : population size, if set to 'None', Evo-MCTS will automatically adjust this parameter.
            num_evaluators  : number of evaluators for parallel evaluation.
            operators       : list of operators to use for the expansion.
            operator_weights: list of weights for the operators.
            alpha           : a parameter for the UCT formula, which is used to balance exploration and exploitation.
            lambda_0        : a parameter for the UCT formula, which is used to balance exploration and exploitation.
            max_retries     : maximum number of retries for the LLM sampler.
            resume_mode     : in resume_mode, randsample will not evaluate the template_program, and will skip the init process. TODO: More detailed usage.
            debug_mode      : if set to True, we will print detailed information.
            multi_thread_or_process_eval: use 'concurrent.futures.ThreadPoolExecutor' or 'concurrent.futures.ProcessPoolExecutor' for the usage of
                multi-core CPU while evaluation. Please note that both settings can leverage multi-core CPU. As a result on my personal computer (Mac OS, Intel chip),
                setting this parameter to 'process' will faster than 'thread'. However, I do not sure if this happens on all platform so I set the default to 'thread'.
                Please note that there is one case that cannot utilize multi-core CPU: if you set 'safe_evaluate' argument in 'evaluator' to 'False',
                and you set this argument to 'thread'.
            **kwargs                    : some args pass to 'llm4ad.base.SecureEvaluator'. Such as 'fork_proc'.
        """
        self._template_program_str = evaluation.template_program
        self._task_description_str = evaluation.task_description
        self._max_sample_nums = max_sample_nums
        self.lambda_0 = lambda_0
        self.alpha = alpha
        self._init_pop_size = init_size
        self._pop_size = pop_size

        # operators
        self.operators = operators if operators is not None else ['e3', 's2', 's3', 'm3', 'm6', 'm7']
        self.operator_weights = operator_weights if operator_weights is not None else [1, 1, 1, 1, 1, 1]

        # statistics
        self._max_retries = max_retries
        self._seed_program_str = seed_program_str

        # samplers and evaluators
        self._num_evaluators = num_evaluators
        self._resume_mode = resume_mode
        self._debug_mode = debug_mode
        llm.debug_mode = debug_mode
        self._multi_thread_or_process_eval = multi_thread_or_process_eval

        # function to be evolved
        self._function_to_evolve: Function = TextFunctionProgramConverter.text_to_function(self._template_program_str)
        self._function_to_evolve_name: str = self._function_to_evolve.name
        self._template_program: Program = TextFunctionProgramConverter.text_to_program(self._template_program_str)

        # adjust population size
        self._adjust_pop_size()

        # population, sampler, and evaluator
        self._llm = llm
        self._population = Population(pop_size=self._pop_size)
        self._sampler = EvoMCTSSampler(llm, self._template_program_str)
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode, **kwargs)
        self._profiler = profiler

        # statistics
        self._tot_sample_nums = 0
        self._iteration = 0

        # reset _initial_sample_nums_max
        self._initial_sample_nums_max = min(
            self._max_sample_nums,
            10 * self._init_pop_size
        )

        # multi-thread executor for evaluation
        assert multi_thread_or_process_eval in ['thread', 'process']
        if multi_thread_or_process_eval == 'thread':
            self._evaluation_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=num_evaluators
            )
        else:
            self._evaluation_executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=num_evaluators
            )

        # pass parameters to profiler
        if profiler is not None:
            self._profiler.record_parameters(llm, evaluation, self)

        self._mcts = MCTS('Root',self.alpha, self.lambda_0)

        self.evolution_module = Evolution(
            llm=self._llm,
            sampler=self._sampler,
            template_program=self._template_program,
            function_to_evolve=self._function_to_evolve,
            task_description=self._task_description_str,
            debug_mode=self._debug_mode
        )

        self.interface_ec = InterfaceEC(
            evolution_module=self.evolution_module,
            evaluation_executor=self._evaluation_executor,
            evaluator=self._evaluator,
            population=self._population,
            debug_mode=self._debug_mode,
            max_retries=self._max_retries
        )

    def run(self):
        print("- Initialization Start (w. reflection) -")
        
        # TODO: Vilid the function of resume_mode
        if self._resume_mode:
            if self._profiler and self._profiler.exp_path and resume_evomcts(self, self._profiler.exp_path):
                print(f"Evo-MCTS Resumed from Iteration {self._iteration}")
            else:
                print("Resume failed. Starting new run.")
                self._initialize_population()
        else:
            self._initialize_population()

        if len(self._population) == 0:
            print("Initialization failed. Terminating.")
            return

        while self._continue_loop():
            self._iteration += 1
            elite = self.interface_ec.elite_offspring
            
            if self._mcts.root.children:
                print(f"Current performances of MCTS nodes: {sorted([np.round(q, 2) for q in self._mcts.rank_list], reverse=True)}")
                print(f"Current number of MCTS nodes in the subtree of each child of the root: {[len(node.subtree) for node in self._mcts.root.children]}")

            cur_node = self._mcts.root
            while cur_node.children:
                if cur_node.depth >= self._mcts.max_depth:
                    break
                uct_scores = [self._mcts.uct(node, max(1 - self._tot_sample_nums / self._max_sample_nums, 0)) for node in cur_node.children]
                selected_idx = uct_scores.index(max(uct_scores))
                cur_node = cur_node.children[selected_idx]
            
            if cur_node.depth < self._mcts.max_depth and (cur_node != self._mcts.root or len(self._mcts.root.children) > 0):
                for op in self.operators:
                    if not self._continue_loop(): break
                    self._expand(cur_node, op)
            
            if isinstance(self._profiler, EvoMCTSProfiler):
                self._profiler.save_mcts_tree(self._mcts.root, self._iteration)

        print("\nEvo-MCTS Finished")
        if self._profiler: self._profiler.finish()
        self._llm.close()

    def _adjust_pop_size(self):
        # adjust population size
        if self._max_sample_nums >= 10000:
            if self._pop_size is None:
                self._pop_size = 40
            elif abs(self._pop_size - 40) > 20:
                print(f'Warning: population size {self._pop_size} '
                      f'is not suitable, please reset it to 40.')
        elif self._max_sample_nums >= 1000:
            if self._pop_size is None:
                self._pop_size = 20
            elif abs(self._pop_size - 20) > 10:
                print(f'Warning: population size {self._pop_size} '
                      f'is not suitable, please reset it to 20.')
        elif self._max_sample_nums >= 200:
            if self._pop_size is None:
                self._pop_size = 10
            elif abs(self._pop_size - 10) > 5:
                print(f'Warning: population size {self._pop_size} '
                      f'is not suitable, please reset it to 10.')
        else:
            if self._pop_size is None:
                self._pop_size = 5
            elif abs(self._pop_size - 5) > 5:
                print(f'Warning: population size {self._pop_size} '
                      f'is not suitable, please reset it to 5.')

    def _initialize_population(self):
        print("\nInitializing MCTS...")
        
        if self._seed_program_str is None:
            print("  No seed provided, generating initial population with diverse operators...")
            
            self._mcts.root.raw_info = self._function_to_evolve
            self._mcts.root.raw_info.score = -float('inf')
            
            # Use a variety of operators to generate a diverse initial population
            init_operators = ['i2', 'i3', 'i4']
            for i in range(self._init_pop_size):
                if not self._continue_loop(): break
                
                op = init_operators[i % len(init_operators)]
                
                parent_node_for_op = self._mcts.root
                if op == 'i4' and self._mcts.root.children:
                    parent_node_for_op = random.choice(self._mcts.root.children)

                self._expand(parent_node_for_op, op)

            if not self._mcts.root.children:
                print("  FATAL: Failed to generate any initial solution. Aborting.")
                return

        else:
            seed_func = TextFunctionProgramConverter.text_to_function(self._seed_program_str)
            if not seed_func:
                print("  ERROR: Could not parse seed program. Aborting.")
                return
            seed_func.algorithm = "Initial Seed Algorithm"
            program = TextFunctionProgramConverter.function_to_program(seed_func, self._template_program)
            score, _ = self._evaluation_executor.submit(self._evaluator.evaluate_program_record_time, program).result()
            if score is None:
                print("  ERROR: Failed to evaluate seed program. Aborting.")
                return
            seed_func.score = score
            self._population.add(seed_func)
            self.interface_ec.update_elite_offspring(seed_func)
            print(f"  Seed program evaluated. Score: {seed_func.score}")
            
            seed_node = MCTSNode(
                algorithm=seed_func.algorithm, code=str(seed_func), obj=seed_func.score,
                parent=self._mcts.root, depth=1, visit=1, Q=seed_func.score, raw_info=seed_func
            )
            self._mcts.root.add_child(seed_node)
            self._mcts.backpropagate(seed_node)
            self.interface_ec.generate_reflection(seed_func, self._function_to_evolve, 1, self._mcts.max_depth)

            # Expand from the provided seed
            for _ in range(1, self._init_pop_size):
                if not self._continue_loop(): break
                self._expand(seed_node, "i4") # Use i4 to generate variants of the seed
        
        print(f"Initialization finished with {len(self._population)} valid solutions.")

    def _expand(self, parent_node: MCTSNode, operator: str):
        pop_context = []
        if operator in ['s1', 's2', 's3']:
            path_set = []
            curr = parent_node
            while curr and curr.raw_info:
                path_set.append(curr.raw_info)
                curr = curr.parent
            pop_context = list(reversed(path_set))
            if len(pop_context) < 2:
                return
        elif operator == 'e3':
            if parent_node.parent:
                pop_context = [child.raw_info for child in parent_node.parent.children if child.raw_info]
        elif operator in ['i2', 'i3', 'i4']:
            # For initializers, no population context is needed.
            # The 'node' argument in evolve_algorithm will be the parent_node.raw_info, used as a seed/template.
            pass
        else:
            pop_context = self._population.get_population()

        if not pop_context and operator not in ['i2', 'i3', 'i4']:
            return

        new_func = self.interface_ec.evolve_algorithm(
            pop=pop_context,
            node=parent_node.raw_info,
            operator=operator,
            iteration=self._iteration,
            depth=parent_node.depth + 1
        )
        self._tot_sample_nums += self._max_retries

        if new_func:
            self.interface_ec.generate_reflection(
                new_func=new_func,
                reference_func=parent_node.raw_info,
                depth=parent_node.depth + 1,
                max_depth=self._mcts.max_depth
            )

            if self._profiler:
                program = TextFunctionProgramConverter.function_to_program(new_func, self._template_program)
                self._profiler.register_function(new_func, program=str(program))

            self._population.add(new_func)
            new_node = MCTSNode(
                algorithm=new_func.algorithm, code=str(new_func), obj=new_func.score,
                parent=parent_node, depth=parent_node.depth + 1, visit=1, Q=new_func.score, raw_info=new_func
            )
            parent_node.add_child(new_node)
            self._mcts.backpropagate(new_node)
            self.interface_ec.update_elite_offspring(new_func)
            
            print(f"Action: {operator}, Father Obj: {np.round(parent_node.raw_info.score, 2)}, New Obj: {np.round(new_func.score, 2)}, Depth: {new_node.depth}")
        else:
            if self._debug_mode:
                print(f"  > Expansion failed for operator '{operator}'.")

    def _continue_loop(self) -> bool:
        return self._tot_sample_nums < self._max_sample_nums