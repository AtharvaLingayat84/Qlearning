from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import random
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evolution import evolve
from metrics import compute_metrics
from simulation import initialize_population, run_generation


@dataclass(frozen=True)
class ExperimentConfig:
    epsilon: float
    alpha: float
    mutation_rate: float


def run_experiment(
    config: ExperimentConfig,
    *,
    generations: int,
    population_size: int,
    rounds: int,
    seed: int | None,
    memory: int = 1,
) -> list[dict]:
    rng = random.Random(seed) if seed is not None else random.Random()
    population = initialize_population(
        population_size,
        alpha=config.alpha,
        epsilon=config.epsilon,
        memory=memory,
    )

    history: list[dict] = []
    for generation in range(generations):
        run_generation(population, rounds=rounds, rng=rng)
        history.append(compute_metrics(population, generation=generation))
        population = evolve(population, mutation_rate=config.mutation_rate, rng=rng)

    return history


def plot_timeseries(history: list[dict], output_path: Path, title: str) -> None:
    generations = [entry["generation"] for entry in history]
    cooperation = [entry["cooperation_rate"] for entry in history]
    reward = [entry["average_reward"] for entry in history]
    c_share = [entry["action_distribution"]["C"] for entry in history]
    d_share = [entry["action_distribution"]["D"] for entry in history]

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(title)

    axes[0].plot(generations, cooperation, color="tab:green")
    axes[0].set_ylabel("Cooperation")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(generations, reward, color="tab:blue")
    axes[1].set_ylabel("Avg Reward")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(generations, c_share, label="C", color="tab:green")
    axes[2].plot(generations, d_share, label="D", color="tab:red")
    axes[2].set_ylabel("Action Share")
    axes[2].set_xlabel("Generation")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_comparison(results: dict[float, list[dict]], metric: str, output_path: Path, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for value, history in results.items():
        generations = [entry["generation"] for entry in history]
        series = [entry[metric] for entry in history]
        ax.plot(generations, series, label=f"{value}")
    ax.set_title(title)
    ax.set_xlabel("Generation")
    ax.set_ylabel(ylabel)
    ax.legend(title="Config")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def summarize_sweep(results: dict[float, list[dict]], label: str) -> str:
    scores = {value: history[-1]["cooperation_rate"] if history else 0.0 for value, history in results.items()}
    best_value = max(scores, key=scores.get)
    worst_value = min(scores, key=scores.get)
    return f"{label}: best final cooperation at {best_value} ({scores[best_value]:.3f}), weakest at {worst_value} ({scores[worst_value]:.3f})."


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent Q-learning Prisoner's Dilemma experiments")
    parser.add_argument("--generations", type=int, default=24)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    epsilon_values = [0.01, 0.1, 0.3]
    alpha_values = [0.1, 0.5]
    mutation_values = [0.0, 0.05, 0.1]

    epsilon_results: dict[float, list[dict]] = {}
    alpha_results: dict[float, list[dict]] = {}
    mutation_results: dict[float, list[dict]] = {}

    for epsilon in epsilon_values:
        config = ExperimentConfig(epsilon=epsilon, alpha=0.1, mutation_rate=0.05)
        epsilon_results[epsilon] = run_experiment(
            config,
            generations=args.generations,
            population_size=args.population,
            rounds=args.rounds,
            seed=args.seed,
        )

    for alpha in alpha_values:
        config = ExperimentConfig(epsilon=0.1, alpha=alpha, mutation_rate=0.05)
        alpha_results[alpha] = run_experiment(
            config,
            generations=args.generations,
            population_size=args.population,
            rounds=args.rounds,
            seed=args.seed,
        )

    for mutation_rate in mutation_values:
        config = ExperimentConfig(epsilon=0.1, alpha=0.1, mutation_rate=mutation_rate)
        mutation_results[mutation_rate] = run_experiment(
            config,
            generations=args.generations,
            population_size=args.population,
            rounds=args.rounds,
            seed=args.seed,
        )

    baseline_key = 0.1
    baseline_history = alpha_results[baseline_key]
    plot_timeseries(
        baseline_history,
        args.output / "baseline_timeseries.png",
        "Baseline Q-learning dynamics",
    )
    plot_comparison(
        epsilon_results,
        "cooperation_rate",
        args.output / "epsilon_comparison.png",
        "Epsilon sweep",
        "Final cooperation rate",
    )
    plot_comparison(
        alpha_results,
        "average_reward",
        args.output / "alpha_comparison.png",
        "Alpha sweep",
        "Average reward",
    )
    plot_comparison(
        mutation_results,
        "cooperation_rate",
        args.output / "mutation_comparison.png",
        "Mutation sweep",
        "Final cooperation rate",
    )

    conclusions = [
        summarize_sweep(epsilon_results, "Exploration"),
        summarize_sweep(alpha_results, "Learning rate"),
        summarize_sweep(mutation_results, "Mutation"),
    ]

    print("\n".join(conclusions))
    print(f"Plots saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
