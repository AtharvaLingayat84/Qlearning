from __future__ import annotations

import random

from agents import Agent
from game import play


def initialize_population(size: int, *, alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 0.1, memory: int = 1) -> list[Agent]:
    return [Agent(agent_id=index, alpha=alpha, gamma=gamma, epsilon=epsilon, memory=memory) for index in range(size)]


def run_generation(population: list[Agent], *, rounds: int = 10, rng: random.Random | None = None) -> list[Agent]:
    rng = rng or random
    order = population[:]
    rng.shuffle(order)

    for index in range(0, len(order) - 1, 2):
        play(order[index], order[index + 1], rounds=rounds, rng=rng)

    return population
