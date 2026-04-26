from __future__ import annotations

import random
from math import ceil

from agents import Agent


def evolve(
    population: list[Agent],
    *,
    mutation_rate: float = 0.05,
    mutation_scale: float = 0.1,
    rng: random.Random | None = None,
) -> list[Agent]:
    if not population:
        return []

    rng = rng or random
    ranked = sorted(population, key=lambda agent: agent.score, reverse=True)
    elite_count = max(1, ceil(len(ranked) / 2))
    elite = ranked[:elite_count]

    next_population: list[Agent] = []
    for index in range(len(population)):
        parent = elite[index] if index < elite_count else rng.choice(elite)
        child = parent.clone(
            new_id=index,
            mutate=index >= elite_count,
            mutation_rate=mutation_rate,
            mutation_scale=mutation_scale,
            rng=rng,
        )
        next_population.append(child)

    return next_population
