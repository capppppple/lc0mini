from __future__ import annotations

from dataclasses import dataclass, field
import math
import random

import chess
import torch

from engine.encoding import board_to_tensor, index_to_move, move_to_index
from engine.network import MiniChessNet


@dataclass
class MCTSNode:
    prior: float
    visits: int = 0
    value_sum: float = 0.0
    pending: int = 0
    children: dict[chess.Move, "MCTSNode"] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class MCTSResult:
    move: chess.Move
    policy: dict[int, float]
    visits: dict[str, int]


@dataclass
class SearchLeaf:
    path: list[MCTSNode]
    board: chess.Board
    node: MCTSNode
    terminal_value: float | None = None


def choose_move(
    board: chess.Board,
    model: MiniChessNet | None = None,
    device: torch.device | str = "cpu",
    temperature: float = 0.0,
    simulations: int = 0,
    exploration_noise: bool = False,
    mcts_batch_size: int = 1,
) -> chess.Move:
    if simulations > 0:
        return mcts_search(
            board,
            model,
            device,
            simulations,
            temperature,
            exploration_noise=exploration_noise,
            mcts_batch_size=mcts_batch_size,
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
    mcts_batch_size: int = 1,
) -> MCTSResult:
    root = MCTSNode(prior=1.0)
    expand(root, board, model, device)
    if exploration_noise:
        add_root_noise(root, dirichlet_alpha, dirichlet_frac)

    if model is not None and mcts_batch_size > 1:
        run_batched_simulations(root, board, model, device, simulations, cpuct, mcts_batch_size)
    else:
        for _ in range(max(simulations, 1)):
            simulate(root, board, model, device, cpuct)

    if not root.children:
        raise ValueError("No legal moves available")

    moves = list(root.children)
    counts = [root.children[move].visits for move in moves]
    move = select_from_counts(moves, counts, temperature)
    total = sum(counts) or 1
    policy = {
        move_to_index(child_move): root.children[child_move].visits / total
        for child_move in moves
        if root.children[child_move].visits > 0
    }
    visits = {
        child_move.uci(): root.children[child_move].visits
        for child_move in moves
        if root.children[child_move].visits > 0
    }
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
    expand_with_priors(node, legal_moves, priors)
    return value


def expand_with_priors(
    node: MCTSNode,
    legal_moves: list[chess.Move],
    priors: dict[chess.Move, float],
) -> None:
    node.children = {move: MCTSNode(prior=priors[move]) for move in legal_moves}


def run_batched_simulations(
    root: MCTSNode,
    board: chess.Board,
    model: MiniChessNet,
    device: torch.device | str,
    simulations: int,
    cpuct: float,
    mcts_batch_size: int,
) -> None:
    remaining = max(simulations, 1)
    while remaining > 0:
        batch_size = min(max(mcts_batch_size, 1), remaining)
        leaves = [select_leaf(root, board, cpuct) for _ in range(batch_size)]
        pending = [leaf for leaf in leaves if leaf.terminal_value is None]

        evaluations = []
        if pending:
            boards = [leaf.board for leaf in pending]
            legal_moves_batch = [list(leaf.board.legal_moves) for leaf in pending]
            evaluations = evaluate_batch(boards, legal_moves_batch, model, device)

        eval_index = 0
        for leaf in leaves:
            if leaf.terminal_value is None:
                legal_moves, priors, value = evaluations[eval_index]
                eval_index += 1
                expand_with_priors(leaf.node, legal_moves, priors)
            else:
                value = leaf.terminal_value
            backpropagate(leaf.path, value)
        remaining -= batch_size


def select_leaf(root: MCTSNode, root_board: chess.Board, cpuct: float) -> SearchLeaf:
    board = root_board.copy(stack=False)
    node = root
    path = [node]

    while True:
        if board.is_game_over(claim_draw=True):
            reserve_path(path)
            return SearchLeaf(path=path, board=board, node=node, terminal_value=terminal_value(board))

        if not node.children:
            reserve_path(path)
            return SearchLeaf(path=path, board=board, node=node)

        move, child = select_child(node, cpuct)
        board.push(move)
        node = child
        path.append(node)


def reserve_path(path: list[MCTSNode]) -> None:
    for node in path:
        node.pending += 1


def backpropagate(path: list[MCTSNode], value: float) -> None:
    for node in reversed(path):
        node.pending = max(node.pending - 1, 0)
        node.visits += 1
        node.value_sum += value
        value = -value


def evaluate_batch(
    boards: list[chess.Board],
    legal_moves_batch: list[list[chess.Move]],
    model: MiniChessNet | None,
    device: torch.device | str,
) -> list[tuple[list[chess.Move], dict[chess.Move, float], float]]:
    if model is None:
        results = []
        for legal_moves in legal_moves_batch:
            prior = 1.0 / len(legal_moves)
            results.append((legal_moves, {move: prior for move in legal_moves}, 0.0))
        return results

    with torch.no_grad():
        x = torch.stack([board_to_tensor(board) for board in boards]).to(device)
        policy_logits, values = model(x)
        policy_logits = policy_logits.detach().cpu()
        values = values.detach().cpu()

    results = []
    for row, legal_moves in enumerate(legal_moves_batch):
        legal_indices = [move_to_index(move) for move in legal_moves]
        legal_logits = policy_logits[row, legal_indices].float()
        probs = torch.softmax(legal_logits, dim=0).tolist()
        priors = {move: float(prob) for move, prob in zip(legal_moves, probs)}
        results.append((legal_moves, priors, float(values[row].item())))
    return results


def evaluate(
    board: chess.Board,
    legal_moves: list[chess.Move],
    model: MiniChessNet | None,
    device: torch.device | str,
) -> tuple[dict[chess.Move, float], float]:
    if model is None:
        prior = 1.0 / len(legal_moves)
        return {move: prior for move in legal_moves}, 0.0

    _, priors, value = evaluate_batch([board], [legal_moves], model, device)[0]
    return priors, value


def select_child(node: MCTSNode, cpuct: float) -> tuple[chess.Move, MCTSNode]:
    parent_visits = max(node.visits, 1)

    def score(child: MCTSNode) -> float:
        exploration = cpuct * child.prior * math.sqrt(parent_visits) / (1 + child.visits)
        return -child.value + exploration - child.pending

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
