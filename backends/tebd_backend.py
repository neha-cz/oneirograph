"""TEBD-inspired policy (stdlib only: no numpy)."""

from __future__ import annotations

import math
import random
from collections import deque


def _embedding_hash(embedding: list) -> int:
    return int(abs(hash(repr([float(x) for x in embedding])))) % (2**31)


class TEBDPolicy:
    def __init__(self, embedding_dim: int = 32, chi: int = 8, gamma: float = 0.95):
        self.embedding_dim = embedding_dim
        self.chi = chi
        self.gamma = gamma
        self.replay_buffer: deque = deque(maxlen=500)
        self.mps_tensors: list = []
        self.step = 0

    def _embedding_to_index(self, embedding: list) -> int:
        return _embedding_hash(embedding) % max(self.chi, 1)

    def select_action(self, current_embedding: list, neighbor_embeddings: list) -> int:
        if not neighbor_embeddings:
            return 0
        epsilon = max(0.05, 0.5 * math.exp(-self.step / 200))
        if random.random() < epsilon or len(self.mps_tensors) == 0:
            return random.randrange(len(neighbor_embeddings))

        q_values = [self._compute_q(current_embedding, emb) for emb in neighbor_embeddings]
        return int(max(range(len(q_values)), key=lambda i: q_values[i]))

    def _compute_q(self, state_emb: list, action_emb: list) -> float:
        if not self.mps_tensors:
            return 0.0
        state_idx = self._embedding_to_index(state_emb)
        action_idx = self._embedding_to_index(action_emb)
        result = 1.0
        for tensor in self.mps_tensors[:4]:
            if tensor is None:
                continue
            if isinstance(tensor, list) and tensor and isinstance(tensor[0], list):
                nrows = len(tensor)
                ncols = len(tensor[0]) if nrows else 0
                s0 = int(state_idx % nrows) if nrows else 0
                a1 = int(action_idx % ncols) if ncols else 0
                result *= float(tensor[s0][a1])
            elif isinstance(tensor, list):
                s0 = int(state_idx % len(tensor)) if tensor else 0
                result *= float(tensor[s0])
        return float(result)

    def tebd_update(self, trajectory: list, total_reward: float) -> None:
        self.step += 1
        if len(trajectory) < 2:
            return

        for i in range(len(trajectory) - 1):
            self.replay_buffer.append(
                {
                    "state": trajectory[i]["embedding"],
                    "action": trajectory[i + 1]["embedding"],
                    "reward": total_reward / len(trajectory),
                    "next_state": trajectory[i + 1]["embedding"],
                }
            )

        if len(self.replay_buffer) < 16:
            return

        buf = list(self.replay_buffer)
        batch_size = min(16, len(buf))
        batch = random.sample(buf, batch_size)

        def pad_row(row: list, dim: int) -> list[float]:
            row = list(row)
            if len(row) >= dim:
                return [float(x) for x in row[:dim]]
            return [float(x) for x in row] + [0.0] * (dim - len(row))

        dim = self.chi
        gate = [[0.0] * dim for _ in range(dim)]
        for b in batch:
            svec = pad_row(b["state"], dim)
            avec = pad_row(b["action"], dim)
            r = float(b["reward"])
            for i in range(dim):
                for j in range(dim):
                    gate[i][j] += svec[i] * avec[j] * r
        scale = batch_size
        for i in range(dim):
            for j in range(dim):
                gate[i][j] /= scale

        new_left = [[gate[i][j] for j in range(dim)] for i in range(dim)]
        new_right = [[gate[j][i] for j in range(dim)] for i in range(dim)]

        if len(self.mps_tensors) < 2:
            self.mps_tensors = [new_left, new_right]
        else:
            L, R = self.mps_tensors[0], self.mps_tensors[1]
            for i in range(dim):
                for j in range(dim):
                    if i < len(L) and j < len(L[i]):
                        L[i][j] = float(L[i][j]) * 0.9 + new_left[i][j] * 0.1
                    if i < len(R) and j < len(R[i]):
                        R[i][j] = float(R[i][j]) * 0.9 + new_right[i][j] * 0.1


global_tebd_policy = TEBDPolicy()


def policy_select_action(current_embedding: list, neighbor_embeddings: list) -> int:
    return global_tebd_policy.select_action(current_embedding, neighbor_embeddings)


def policy_tebd_update(trajectory: list, total_reward: float) -> None:
    global_tebd_policy.tebd_update(trajectory, total_reward)


def report_tebd_state() -> dict:
    """Snapshot of the TEBD-inspired policy after adaptive passes."""
    p = global_tebd_policy
    return {
        "tebd_step": p.step,
        "replay_buffer_size": len(p.replay_buffer),
        "mps_tensor_layers": len(p.mps_tensors),
        "chi": p.chi,
        "gamma": p.gamma,
    }
