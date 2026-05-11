from __future__ import annotations

from dataclasses import dataclass, field
import math
import random

import chess
import torch

from engine.encoding import POLICY_SIZE, board_to_tensor, index_to_move, move_to_index
from engine.network import MiniChessNet


@dataclass
class MCTSNode:
    prior: float
    visits: int = 0
    value_sum: float = 0.0
    children: dict[chess.Move, "MCTSNode"] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class MCTSResult:
    move: chess.Move
    policy: dict[int, float]
    visits: dict[str, int]


def choose_move(
    board: chess.Board,
    model: MiniChessNet | None = None,
    device: torch.device | str = "cpu",
    temperature: float = 0.0,
    simulations: int = 0,
    exploration_noise: bool = False,
) -> chess.Move:
    if simulations > 0:
        return mcts_search(
            board,
            model,
            device,
            simulations,
            temperature,
            exploration_noise=exploration_noise,
        ).move

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        raise ValueError("No legal moves available")

    if model is None:
        return random.choice(legal_moves)

    with torch.no_grad():
        x = board_to_tensor(board).unsqueeze(0).to(device)
        policy_logits, _ = model(x)
        logits = policy_logits[0].detach().cpu()

    scored = []
    for move in legal_moves:
        scored.append((float(logits[move_to_index(move)]), move))

    if temperature <= 0.0:
        return max(scored, key=lambda item: item[0])[1]

    max_logit = max(score for score, _ in scored)
    weights = [math.exp((score - max_logit) / temperature) for score, _ in scored]
    return random.choices([move for _, move in scored], weights=weights, k=1)[0]


def policy_move_from_index(board: chess.Board, index: int) -> chess.Move | None:
    return index_to_move(index, board)


def mcts_search(
    board: chess.Board,
    model: MiniChessNet | None = None,
    device: torch.device | str = "cpu",
    simulations: int = 64,
    temperature: float = 0.0,
    cpuct: float = 1.5,
    exploration_noise: bool = False,
    dirichlet_alpha: float = 0.3,
    dirichlet_frac: float = 0.25,
) -> MCTSResult:
    root = MCTSNode(prior=1.0)
    expand(root, board, model, device)
    if exploration_noise:
        add_root_noise(root, dirichlet_alpha, dirichlet_frac)

    for _ in range(max(simulations, 1)):
        simulate(root, board, model, device, cpuct)

    if not root.children:
        raise ValueError("No legal moves available")

    moves = list(root.children)
    counts = [root.children[move].visits for move in moves]
    move = select_from_counts(moves, counts, temperature)
    total = sum(counts) or 1
    policy = {move_to_index(child_move): root.children[child_move].visits / total for child_move in moves}
    visits = {child_move.uci(): root.children[child_move].visits for child_move in moves}
    return MCTSResult(move=move, policy=policy, visits=visits)


def simulate(
    node: MCTSNode,
    board: chess.Board,
    model: MiniChessNet | None,
    device: torch.device | str,
    cpuct: float,
) -> float:
    if board.is_game_over(claim_draw=True):
        return terminal_value(board)

    if not node.children:
        value = expand(node, board, model, device)
        node.visits += 1
        node.value_sum += value
        return value

    move, child = select_child(node, cpuct)
    board.push(move)
    value = -simulate(child, board, model, device, cpuct)
    board.pop()

    node.visits += 1
    node.value_sum += value
    return value


def expand(
    node: MCTSNode,
    board: chess.Board,
    model: MiniChessNet | None,
    device: torch.device | str,
) -> float:
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return terminal_value(board)

    priors, value = evaluate(board, legal_moves, model, device)
    node.children = {move: MCTSNode(prior=priors[move]) for move in legal_moves}
    return value


def evaluate(
    board: chess.Board,
    legal_moves: list[chess.Move],
    model: MiniChessNet | None,
    device: torch.device | str,
) -> tuple[dict[chess.Move, float], float]:
    if model is None:
        prior = 1.0 / len(legal_moves)
        return {move: prior for move in legal_moves}, 0.0

    with torch.no_grad():
        x = board_to_tensor(board).unsqueeze(0).to(device)
        policy_logits, value = model(x)
        logits = policy_logits[0].detach().cpu()

    legal_indices = [move_to_index(move) for move in legal_moves]
    legal_logits = torch.tensor([float(logits[index]) for index in legal_indices], dtype=torch.float32)
    probs = torch.softmax(legal_logits, dim=0).tolist()
    return {move: float(prob) for move, prob in zip(legal_moves, probs)}, float(value.item())


def select_child(node: MCTSNode, cpuct: float) -> tuple[chess.Move, MCTSNode]:
    parent_visits = max(node.visits, 1)

    def score(child: MCTSNode) -> float:
        exploration = cpuct * child.prior * math.sqrt(parent_visits) / (1 + child.visits)
        return -child.value + exploration

    return max(node.children.items(), key=lambda item: score(item[1]))


def select_from_counts(
    moves: list[chess.Move],
    counts: list[int],
    temperature: float,
) -> chess.Move:
    if temperature <= 0.0:
        return moves[max(range(len(moves)), key=lambda index: counts[index])]

    adjusted = [count ** (1.0 / temperature) for count in counts]
    if sum(adjusted) <= 0:
        return random.choice(moves)
    return random.choices(moves, weights=adjusted, k=1)[0]


def add_root_noise(node: MCTSNode, alpha: float, frac: float) -> None:
    if not node.children or frac <= 0:
        return

    noise = [random.gammavariate(alpha, 1.0) for _ in node.children]
    total = sum(noise) or 1.0
    for child, noise_value in zip(node.children.values(), noise):
        child.prior = child.prior * (1.0 - frac) + frac * noise_value / total


def terminal_value(board: chess.Board) -> float:
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0
