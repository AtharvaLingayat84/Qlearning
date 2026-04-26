from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
import random
from typing import Iterable

ACTIONS = ("C", "D")


def encode_state(own_history: Iterable[str], opp_history: Iterable[str], memory: int = 1) -> tuple[str, ...]:
    """Build a compact, hashable repeated-PD state."""

    def _tail(values: Iterable[str]) -> tuple[str, ...]:
        tail = list(values)[-memory:]
        while len(tail) < memory:
            tail.insert(0, "START")
        return tuple(tail)

    return _tail(own_history) + _tail(opp_history)


@dataclass
class Agent:
    agent_id: int
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon: float = 0.1
    memory: int = 1
    q_table: dict[tuple[str, ...], dict[str, float]] = field(default_factory=dict)
    score: float = 0.0
    last_state: tuple[str, ...] | None = None
    last_action: str | None = None
    cooperation_count: int = 0
    action_history: list[str] = field(default_factory=list)

    def _ensure_state(self, state: tuple[str, ...]) -> dict[str, float]:
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in ACTIONS}
        return self.q_table[state]

    def choose_action(self, state: tuple[str, ...], rng: random.Random | None = None) -> str:
        rng = rng or random
        state = tuple(state)
        q_values = self._ensure_state(state)

        if rng.random() < self.epsilon:
            action = rng.choice(ACTIONS)
        else:
            best_value = max(q_values.values())
            best_actions = [action for action, value in q_values.items() if value == best_value]
            action = rng.choice(best_actions)

        self.last_state = state
        self.last_action = action
        self.action_history.append(action)
        if action == "C":
            self.cooperation_count += 1
        return action

    def learn(self, reward: float, new_state: tuple[str, ...] | None) -> None:
        if self.last_state is None or self.last_action is None:
            return

        current_values = self._ensure_state(self.last_state)
        next_state_values = self._ensure_state(tuple(new_state)) if new_state is not None else {action: 0.0 for action in ACTIONS}
        best_next = max(next_state_values.values())
        old_value = current_values[self.last_action]
        current_values[self.last_action] = old_value + self.alpha * (reward + self.gamma * best_next - old_value)
        if new_state is not None:
            self.last_state = tuple(new_state)

    def reset_generation_stats(self) -> None:
        self.score = 0.0
        self.last_state = None
        self.last_action = None
        self.cooperation_count = 0
        self.action_history = []

    def clone(self, *, new_id: int | None = None, mutate: bool = False, mutation_rate: float = 0.0, mutation_scale: float = 0.1, rng: random.Random | None = None) -> "Agent":
        rng = rng or random
        child = Agent(
            agent_id=self.agent_id if new_id is None else new_id,
            alpha=self.alpha,
            gamma=self.gamma,
            epsilon=self.epsilon,
            memory=self.memory,
            q_table=deepcopy(self.q_table),
        )
        if mutate:
            for state, action_values in child.q_table.items():
                for action in ACTIONS:
                    if rng.random() < mutation_rate:
                        action_values[action] += rng.uniform(-mutation_scale, mutation_scale)
        child.reset_generation_stats()
        return child
