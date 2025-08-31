# state_action.py
# core components for mdp: state and action definitions

from enum import Enum
from dataclasses import dataclass

class Action(Enum):
    """
    # enumeration of possible actions in grid world
    # each action has index and delta values for row/col movement
    """
    # cardinal directions (4-action space)
    UP = (0, -1, 0)          # move up: decrease row by 1
    DOWN = (1, 1, 0)         # move down: increase row by 1
    LEFT = (2, 0, -1)        # move left: decrease col by 1
    RIGHT = (3, 0, 1)        # move right: increase col by 1
    # diagonal directions (8-action space)
    UP_LEFT = (4, -1, -1)    # move diagonally up-left
    UP_RIGHT = (5, -1, 1)    # move diagonally up-right
    DOWN_LEFT = (6, 1, -1)   # move diagonally down-left
    DOWN_RIGHT = (7, 1, 1)   # move diagonally down-right
    
    def __init__(self, index, row_delta, col_delta):
        # store index and movement deltas for each action
        self.index = index
        self.row_delta = row_delta
        self.col_delta = col_delta

@dataclass
class State:
    """
    # represents a state in grid world as coordinate pair
    # using dataclass for clean init and automatic methods
    """
    row: int  # row position in grid (0 to grid_size-1)
    col: int  # column position in grid (0 to grid_size-1)
    
    def __hash__(self):
        # hash function needed for using state as dict key
        return hash((self.row, self.col))
    
    def __eq__(self, other):
        # equality comparison for states
        return self.row == other.row and self.col == other.col
    
    def to_tuple(self):
        # convert state to tuple for easy printing/storage
        return (self.row, self.col)
    
    def __str__(self):
        # string representation for printing
        return f"({self.row}, {self.col})"

class Heuristic(Enum):
    """
    # different heuristic functions for informed search
    # each optimal for different action spaces and cost models
    """
    MANHATTAN = "manhattan"    # optimal for 4-action space
    EUCLIDEAN = "euclidean"    # straight-line distance
    CHEBYSHEV = "chebyshev"    # optimal for 8-action uniform cost
    OCTILE = "octile"          # optimal for 8-action non-uniform cost

class CostModel(Enum):
    """
    # cost models for different action spaces
    # defines movement costs for cardinal vs diagonal moves
    """
    UNIFORM = "uniform"              # all moves cost -1
    NON_UNIFORM = "non_uniform"      # cardinal: -1, diagonal: -sqrt(2)