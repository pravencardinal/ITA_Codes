"""
ga_engine_v3.py

GA orchestration with logging hooks around heavy operations.
Parallel fitness evaluation included; decorated functions log execution details.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Any, Tuple
import numpy as np
import random
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from config import GAConfig, get_default_config
import fitness as fitness_mod
from logger_utils import log_execution, init_log

logger = logging.getLogger(__name__)
_cfg = get_default_config()
init_log(_cfg.LOG_FILE)

FitnessFn = Callable[[List[int], dict], tuple]

@dataclass
class Individual:
    gene_list: np.ndarray
    processed: dict
    fitness_value: tuple = field(init=False, default=None)
    fitness_function: FitnessFn = None

    def compute_fitness(self):
        if self.fitness_function is None:
            raise RuntimeError("Fitness function not set on Individual class.")
        self.fitness_value = self.fitness_function(self.gene_list, self.processed)
        return self.fitness_value

    @property
    def fitness(self):
        if self.fitness_value is None:
            return self.compute_fitness()
        return self.fitness_value

    @classmethod
    def set_fitness_function(cls, fn: FitnessFn):
        cls.fitness_function = fn

    @classmethod
    def generate_random(cls, processed: dict):
        # n_shipto = processed["dat"].shape[0]
        # choices = np.random.choice(processed["n_lvl5_unq"], n_shipto, replace=True)
        dat = processed["dat"]
        choices = dat['lvl5_int']
        return cls(choices, processed)

class GAEngine:
    def __init__(self, config: GAConfig, processed_data: dict):
        self.cfg = config
        self.processed = processed_data
        random.seed(self.cfg.SEED)
        np.random.seed(self.cfg.SEED)
        self.population: List[Individual] = []
        self.best_individual: Individual = None

    @log_execution("ga_engine", "initialize_population", "Initialize population and evaluate fitness", log_path=_cfg.LOG_FILE)
    def initialize_population(self):
        self.population = [Individual.generate_random(self.processed) for _ in range(self.cfg.POPULATION_SIZE)]
        self.evaluate_population(self.population)

    @log_execution("ga_engine", "evaluate_population", "Evaluate fitness for a population (parallel)", log_path=_cfg.LOG_FILE)
    def evaluate_population(self, population: List[Individual], parallel: bool = True):
        if parallel:
            with ProcessPoolExecutor() as exe:
                futures = {exe.submit(fitness_mod.fitness_function, ind.gene_list, self.processed): ind for ind in population}
                for fut in as_completed(futures):
                    ind = futures[fut]
                    try:
                        ind.fitness_value = fut.result()
                    except Exception:
                        logger.exception("Failed to evaluate fitness for an individual; falling back to direct call.")
                        ind.fitness_value = fitness_mod.fitness_function(ind.gene_list, self.processed)
        else:
            for ind in population:
                ind.compute_fitness()

    def select(self) -> List[Individual]:
        sorted_pop = sorted(self.population, key=lambda i: i.fitness[3], reverse=True)
        elite = sorted_pop[:self.cfg.ELITE_SIZE]
        fitness_scores = np.array([max(0.0, ind.fitness[3]) for ind in sorted_pop])
        total = fitness_scores.sum()
        if total == 0:
            selected = elite + random.sample(sorted_pop[self.cfg.ELITE_SIZE:], len(sorted_pop) - self.cfg.ELITE_SIZE)
            return selected
        probs = fitness_scores / total
        n_to_select = len(sorted_pop) - self.cfg.ELITE_SIZE
        chosen_idx = np.random.choice(range(len(sorted_pop)), size=n_to_select, replace=True, p=probs)
        selected = elite + [sorted_pop[i] for i in chosen_idx]
        return selected

    def crossover(self, parents: List[Individual]) -> List[Individual]:
        children = []
        for a, b in zip(parents[::2], parents[1::2]):
            if random.random() < self.cfg.CROSSOVER_PROBABILITY:
                mask = np.random.rand(len(a.gene_list)) > 0.5
                child1_genes = np.where(mask, a.gene_list, b.gene_list)
                child2_genes = np.where(mask, b.gene_list, a.gene_list)
                children.append(Individual(child1_genes, self.processed))
                children.append(Individual(child2_genes, self.processed))
            else:
                children.append(a)
                children.append(b)
        return children

    def mutate(self, population: List[Individual]) -> List[Individual]:
        mutated = []
        for ind in population:
            if random.random() < self.cfg.MUTATION_PROBABILITY:
                new_genes = ind.gene_list.copy()
                n_mut = max(1, int(0.01 * new_genes.size))
                idxs = np.random.choice(new_genes.size, n_mut, replace=False)
                for idx in idxs:
                    new_genes[idx] = np.random.choice(self.processed["n_lvl5_unq"])
                mutated.append(Individual(new_genes, self.processed))
            else:
                mutated.append(ind)
        return mutated

    @log_execution("ga_engine", "run", "Main GA loop with selection/crossover/mutation/checkpointing", log_path=_cfg.LOG_FILE)
    def run(self) -> Tuple[Individual, dict]:
        logger.info("Starting GA run: pop=%d, max_gen=%d", self.cfg.POPULATION_SIZE, self.cfg.MAX_GENERATIONS)
        if not self.population:
            self.initialize_population()

        best = max(self.population, key=lambda i: i.fitness[3])
        gen = 0
        stats = {"best_per_gen": [], "avg_per_gen": []}

        while gen < self.cfg.MAX_GENERATIONS:
            gen += 1
            logger.info("Generation %d", gen)
            parents = self.select()
            children = self.crossover(parents)
            next_pop = self.mutate(children)
            self.evaluate_population(next_pop)
            self.population = next_pop
            avg_fit = float(np.mean([ind.fitness[3] for ind in self.population]))
            best = max(self.population, key=lambda i: i.fitness[3])
            best_fit = float(best.fitness[3])
            stats["best_per_gen"].append(best_fit)
            stats["avg_per_gen"].append(avg_fit)
            logger.info("Gen %d: best=%s avg=%s", gen, best_fit, avg_fit)
            if gen % self.cfg.CHECKPOINT_FREQ == 0:
                cp_path = os.path.join(self.cfg.CHECKPOINT_DIR, f"ga_checkpoint_gen_{gen}.pkl")
                from data_io_v6 import save_checkpoint
                save_checkpoint(self.population, cp_path)
                logger.info("Checkpoint saved to %s", cp_path)
            if best.fitness[3] == 0:
                logger.info("Perfect solution found at generation %d", gen)
                break

        self.best_individual = best
        return best, stats
