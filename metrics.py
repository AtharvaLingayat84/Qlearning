from __future__ import annotations

from statistics import mean

from agents import ACTIONS, Agent


def compute_metrics(population: list[Agent], generation: int | None = None, generation_summary: dict | None = None) -> dict:
    total_actions = sum(len(agent.action_history) for agent in population)
    cooperation_count = sum(agent.action_history.count("C") for agent in population)
    defection_count = total_actions - cooperation_count
    scores = [agent.score for agent in population]

    metrics = {
        "generation": generation,
        "cooperation_rate": (cooperation_count / total_actions) if total_actions else 0.0,
        "average_reward": mean(scores) if scores else 0.0,
        "action_distribution": {
            "C": (cooperation_count / total_actions) if total_actions else 0.0,
            "D": (defection_count / total_actions) if total_actions else 0.0,
        },
        "payoff_spread": (max(scores) - min(scores)) if scores else 0.0,
        "top_agent_score": max(scores) if scores else 0.0,
    }

    if generation_summary:
        metrics["generation_summary"] = generation_summary

    return metrics
