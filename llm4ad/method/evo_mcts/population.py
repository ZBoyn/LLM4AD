import heapq
import random
from typing import List
from ...base import Function

class Population:

    def __init__(self, pop_size: int):
        self._pop_size = pop_size
        self._population: List[Function] = []
        self._objective_history = set()

    def __len__(self) -> int:
        return len(self._population)
    
    def __iter__(self):
        return iter(self._population)

    def add(self, individual: Function):
        if individual.score is None or individual.score in self._objective_history:
            return

        if len(self._population) < self._pop_size:
            # Population is not full, add directly
            self._population.append(individual)
            self._objective_history.add(individual.score)
        else:
            # Population is full, check against the worst individual
            worst_individual = min(self._population, key=lambda x: x.score)
            if individual.score > worst_individual.score:
                # New individual is better, so replace the worst
                self._population.remove(worst_individual)
                self._objective_history.remove(worst_individual.score)
                self._population.append(individual)
                self._objective_history.add(individual.score)

    def get_population(self) -> List[Function]:
        return self._population

    def get_random_individuals(self, k: int, exclude: Function = None) -> List[Function]:
        candidates = list(self._population)
        if exclude and exclude in candidates:
            candidates.remove(exclude)
        
        if not candidates:
            return []
        
        k = min(k, len(candidates))
        return list(random.sample(candidates, k))

    def parent_selection(self, m: int, unique: bool = False) -> List[Function]:
        if not self._population:
            return []

        # Sort population by score (descending)
        sorted_pop = sorted(self._population, key=lambda x: x.score, reverse=True)
        
        ranks = list(range(len(sorted_pop)))
        
        # Probabilities are inversely proportional to rank
        probs = [1 / (rank + 1 + len(sorted_pop)) for rank in ranks]
        
        m = min(m, len(sorted_pop))

        if unique:
            if m > len(sorted_pop): m = len(sorted_pop)
            selected_list = []
            # Use code content to track uniqueness as Function objects are unhashable
            selected_hashes = set() 
            
            # Safeguard against infinite loops if diversity is low
            max_attempts = m * 5 
            attempts = 0
            while len(selected_list) < m and attempts < max_attempts:
                chosen = random.choices(sorted_pop, weights=probs, k=1)[0]
                chosen_hash = str(chosen)
                if chosen_hash not in selected_hashes:
                    selected_list.append(chosen)
                    selected_hashes.add(chosen_hash)
                attempts += 1
            return selected_list
        else:
            return random.choices(sorted_pop, weights=probs, k=m)
