# environment.py
# maze environment implementation with mdp formalization

import numpy as np
from typing import List, Tuple, Set
from state_action import State, Action, Heuristic, CostModel

class MazeWorld:
    """
    # complete mdp formalization of grid maze environment
    # supports both 4-action and 8-action spaces with different cost models
    """
    
    def __init__(self, grid_size: int = 8, maze_config: str = "original", 
                 use_8_actions: bool = False, cost_model: CostModel = CostModel.UNIFORM,
                 discount_factor: float = 0.9):
        """
        # initialize maze world with specified configuration
        # grid_size: dimensions of square grid (default 8x8)
        # maze_config: which maze layout to use (original/swapped/flipped)
        # use_8_actions: whether to allow diagonal movements
        # cost_model: uniform (-1 all moves) or non-uniform (-1 cardinal, -sqrt(2) diagonal)
        # discount_factor: gamma for discounting future rewards
        """
        self.grid_size = grid_size
        self.maze_config = maze_config
        self.use_8_actions = use_8_actions
        self.cost_model = cost_model
        self.gamma = discount_factor
        
        # initialize maze configuration (walls, start, goal)
        self._initialize_maze(maze_config)
        
        # define action space based on configuration
        if use_8_actions:
            # 8-action space includes diagonals
            self.action_space = list(Action)
        else:
            # 4-action space only cardinal directions
            self.action_space = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
        
        # initialize state space as all valid (non-wall) cells
        self.state_space = []
        for r in range(grid_size):
            for c in range(grid_size):
                if (r, c) not in self.walls:
                    self.state_space.append(State(r, c))
        
        # define reward structure
        # using positive goal reward and negative penalties to shape behavior
        self.goal_reward = 50.0      # large positive reward for reaching goal
        self.wall_penalty = -5.0     # penalty for hitting walls
        
        # calculate step costs based on action space and cost model
        self.step_costs = self._calculate_step_costs()
    
    def _calculate_step_costs(self) -> dict:
        """
        # calculate step costs for each action based on cost model
        # uniform: all moves cost -1
        # non-uniform: cardinal -1, diagonal -sqrt(2) for true euclidean distance
        """
        costs = {}
        
        for action in self.action_space:
            if self.cost_model == CostModel.UNIFORM:
                # uniform cost model: all moves cost -1
                costs[action] = -1.0
            else:  # non-uniform cost model
                if action in [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]:
                    # cardinal moves cost -1
                    costs[action] = -1.0
                else:
                    # diagonal moves cost -sqrt(2) (approx -1.414)
                    # this reflects true geometric distance
                    costs[action] = -np.sqrt(2)
        
        return costs
    
    def _initialize_maze(self, config: str):
        """
        # set up specific maze configuration
        # defines walls, start, and goal for each maze type
        """
        if config == "original":
            self._original_maze()
        elif config == "swapped":
            self._swapped_maze()
        elif config == "flipped":
            self._flipped_maze()
        else:
            raise ValueError(f"unknown maze configuration: {config}")
    
    def _original_maze(self):
        """
        # original maze configuration with exact wall positions
        # walls create obstacles between start (7,0) and goal (0,7)
        """
        self.walls = {
            (0, 5), (1, 2), (2, 4), (2, 7),
            (3, 1), (3, 4), (3, 5),
            (5, 0), (5, 2), (5, 6),
            (6, 4), (7, 2)
        }
        self.start = State(7, 0)  # bottom-left corner
        self.goal = State(0, 7)   # top-right corner
    
    def _swapped_maze(self):
        """
        # original maze with start and goal positions swapped
        # tests if pathfinding is symmetric in both directions
        """
        # use exact same walls as original
        self.walls = {
            (0, 5), (1, 2), (2, 4), (2, 7),
            (3, 1), (3, 4), (3, 5),
            (5, 0), (5, 2), (5, 6),
            (6, 4), (7, 2)
        }
        # swap start and goal positions
        self.start = State(0, 7)  # was goal, now start
        self.goal = State(7, 0)   # was start, now goal
    
    def _flipped_maze(self):
        """
        # horizontally flipped version of original maze
        # tests geometric symmetry of policies
        """
        # get original walls and flip horizontally
        original_walls = {
            (0, 5), (1, 2), (2, 4), (2, 7),
            (3, 1), (3, 4), (3, 5),
            (5, 0), (5, 2), (5, 6),
            (6, 4), (7, 2)
        }
        # apply horizontal flip: (r, c) -> (r, 7-c)
        self.walls = {(r, 7 - c) for r, c in original_walls}
        self.start = State(7, 7)  # flipped from (7, 0)
        self.goal = State(0, 0)   # flipped from (0, 7)
    
    def is_valid_state(self, state: State) -> bool:
        """
        # check if state is valid (within bounds and not a wall)
        # returns true if agent can occupy this state
        """
        # check grid boundaries
        if state.row < 0 or state.row >= self.grid_size:
            return False
        if state.col < 0 or state.col >= self.grid_size:
            return False
        # check if position is a wall
        if (state.row, state.col) in self.walls:
            return False
        return True
    
    def transition(self, state: State, action: Action) -> Tuple[State, float, bool]:
        """
        # deterministic transition function p(s'|s,a)
        # returns: (next_state, reward, done)
        # deterministic means actions always have same effect
        """
        # calculate potential next state based on action
        next_state = State(
            state.row + action.row_delta,
            state.col + action.col_delta
        )
        
        # check if move is valid
        if not self.is_valid_state(next_state):
            # invalid move - agent stays in place and receives wall penalty
            return state, self.wall_penalty, False
        
        # check if goal is reached
        if next_state == self.goal:
            # goal reached - episode ends with goal reward
            return next_state, self.goal_reward, True
        
        # normal move - agent moves to next state with step cost
        return next_state, self.step_costs[action], False
    
    def get_valid_actions(self, state: State) -> List[Action]:
        """
        # get list of actions that lead to valid states
        # used for smarter action selection in some policies
        """
        valid_actions = []
        for action in self.action_space:
            next_state = State(
                state.row + action.row_delta,
                state.col + action.col_delta
            )
            if self.is_valid_state(next_state):
                valid_actions.append(action)
        return valid_actions
    
    def calculate_heuristic(self, state: State, heuristic_type: Heuristic) -> float:
        """
        # calculate heuristic distance from state to goal
        # different heuristics optimal for different action spaces/cost models
        """
        dr = abs(state.row - self.goal.row)  # row difference
        dc = abs(state.col - self.goal.col)  # col difference
        
        if heuristic_type == Heuristic.MANHATTAN:
            # manhattan distance: sum of absolute differences
            # optimal for 4-action space
            return dr + dc
        
        elif heuristic_type == Heuristic.EUCLIDEAN:
            # euclidean distance: straight-line distance
            return np.sqrt(dr**2 + dc**2)
        
        elif heuristic_type == Heuristic.CHEBYSHEV:
            # chebyshev distance: max of absolute differences
            # optimal for 8-action uniform cost (all moves cost same)
            return max(dr, dc)
        
        elif heuristic_type == Heuristic.OCTILE:
            # octile distance: accounts for diagonal cost sqrt(2)
            # optimal for 8-action non-uniform cost
            # formula: max(dr,dc) + (sqrt(2)-1)*min(dr,dc)
            return max(dr, dc) + (np.sqrt(2) - 1) * min(dr, dc)
        
        else:
            raise ValueError(f"unknown heuristic: {heuristic_type}")
    
    def get_optimal_heuristic(self) -> Heuristic:
        """
        # return the optimal heuristic for current configuration
        # depends on action space and cost model
        """
        if not self.use_8_actions:
            # 4-action space: manhattan is optimal
            return Heuristic.MANHATTAN
        elif self.cost_model == CostModel.UNIFORM:
            # 8-action uniform cost: chebyshev is optimal
            return Heuristic.CHEBYSHEV
        else:
            # 8-action non-uniform cost: octile is optimal
            return Heuristic.OCTILE
    
    def get_maze_info(self) -> dict:
        """
        # return information about maze configuration
        # useful for analysis and reporting
        """
        return {
            'name': self.maze_config,
            'grid_size': self.grid_size,
            'start': self.start.to_tuple(),
            'goal': self.goal.to_tuple(),
            'num_walls': len(self.walls),
            'num_valid_states': len(self.state_space),
            'action_space': '8-action' if self.use_8_actions else '4-action',
            'cost_model': self.cost_model.value,
            'optimal_heuristic': self.get_optimal_heuristic().value,
            'discount_factor': self.gamma
        }