from __future__ import annotations

import chess
import torch


PIECE_PLANES = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

NUM_PLANES = 18
SQUARE_POLICY_SIZE = 64 * 64
POLICY_SIZE = SQUARE_POLICY_SIZE * 5
PROMOTION_OFFSET = SQUARE_POLICY_SIZE
PROMOTION_TO_INDEX = {
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}
INDEX_TO_PROMOTION = {value: key for key, value in PROMOTION_TO_INDEX.items()}


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """Encode a board as [18, 8, 8] float tensor."""
    planes = torch.zeros((NUM_PLANES, 8, 8), dtype=torch.float32)

    for square, piece in board.piece_map().items():
        color_offset = 0 if piece.color == chess.WHITE else 6
        plane = color_offset + PIECE_PLANES[piece.piece_type]
        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)
        planes[plane, row, col] = 1.0

    planes[12].fill_(1.0 if board.turn == chess.WHITE else 0.0)
    planes[13].fill_(1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0)
    planes[14].fill_(1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0)
    planes[15].fill_(1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0)
    planes[16].fill_(1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0)
    planes[17].fill_(min(board.fullmove_number / 100.0, 1.0))
    return planes


def move_to_index(move: chess.Move) -> int:
    base = move.from_square * 64 + move.to_square
    if move.promotion:
        return PROMOTION_TO_INDEX.get(move.promotion, 4) * SQUARE_POLICY_SIZE + base
    return base


def index_to_move(index: int, board: chess.Board) -> chess.Move | None:
    if index >= POLICY_SIZE:
        return None

    promotion_plane = index // SQUARE_POLICY_SIZE
    square_index = index % SQUARE_POLICY_SIZE
    from_square = square_index // 64
    to_square = square_index % 64
    promotion = INDEX_TO_PROMOTION.get(promotion_plane)
    if promotion:
        move = chess.Move(from_square, to_square, promotion=promotion)
        return move if move in board.legal_moves else None

    move = chess.Move(from_square, to_square)
    if move in board.legal_moves:
        return move
    return None
