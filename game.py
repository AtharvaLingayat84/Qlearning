from __future__ import annotations

from collections.abc import Sequence
import random

from agents import ACTIONS, Agent, encode_state

PAYOFF_MATRIX: dict[tuple[str, str], tuple[float, float]] = {
    ("C", "C"): (3.0, 3.0),
    ("C", "D"): (0.0, 5.0),
    ("D", "C"): (5.0, 0.0),
    ("D", "D"): (1.0, 1.0),
}


def play(agent_a: Agent, agent_b: Agent, rounds: int = 10, rng: random.Random | None = None) -> tuple[Agent, Agent]:
    rng = rng or random
    history_a: list[str] = []
    history_b: list[str] = []

    for _ in range(rounds):
        state_a = encode_state(history_a, history_b, memory=agent_a.memory)
        state_b = encode_state(history_b, history_a, memory=agent_b.memory)

        action_a = agent_a.choose_action(state_a, rng)
        action_b = agent_b.choose_action(state_b, rng)

        reward_a, reward_b = PAYOFF_MATRIX[(action_a, action_b)]
        agent_a.score += reward_a
        agent_b.score += reward_b

        history_a.append(action_a)
        history_b.append(action_b)

        next_state_a = encode_state(history_a, history_b, memory=agent_a.memory)
        next_state_b = encode_state(history_b, history_a, memory=agent_b.memory)

        agent_a.learn(reward_a, next_state_a)
        agent_b.learn(reward_b, next_state_b)

    return agent_a, agent_b
