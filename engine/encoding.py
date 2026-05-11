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
POLICY_SIZE = 64 * 64 + 5
PROMOTION_OFFSET = 64 * 64
PROMOTION_TO_INDEX = {
    chess.KNIGHT: 0,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 3,
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
        return PROMOTION_OFFSET + PROMOTION_TO_INDEX.get(move.promotion, 4)
    return base


def index_to_move(index: int, board: chess.Board) -> chess.Move | None:
    if index >= PROMOTION_OFFSET:
        return None

    from_square = index // 64
    to_square = index % 64
    move = chess.Move(from_square, to_square)
    if move in board.legal_moves:
        return move

    for promotion in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
        promo_move = chess.Move(from_square, to_square, promotion=promotion)
        if promo_move in board.legal_moves:
            return promo_move
    return None

