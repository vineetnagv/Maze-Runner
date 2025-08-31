# policies.py
# implementation of different agent policies for maze navigation

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional
from state_action import State, Action, Heuristic

class Policy(ABC):
    """
    # abstract base class for all policies
    # defines interface that all policies must implement
    """
    
    @abstractmethod
    def select_action(self, env, state: State, episode: int = 0) -> Action:
        """
        # select action given current state and environment
        # must be implemented by all concrete policy classes
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        # return name of policy for display purposes
        """
        pass
    
    def reset(self):
        """
        # reset any internal state of policy
        # override if policy maintains internal state
        """
        pass

class RandomPolicy(Policy):
    """
    # baseline random policy - selects actions uniformly at random
    # establishes lower bound on performance for comparison
    """
    
    def select_action(self, env, state: State, episode: int = 0) -> Action:
        """
        # select random action from action space
        # baseline to compare against informed policies
        """
        # get valid actions to avoid repeatedly hitting walls
        valid_actions = env.get_valid_actions(state)
        if valid_actions:
            return np.random.choice(valid_actions)
        # if no valid actions (shouldn't happen), choose any action
        return np.random.choice(env.action_space)
    
    def get_name(self) -> str:
        return "Random"

class GreedyPolicy(Policy):
    """
    # greedy policy using heuristic function
    # always chooses action that minimizes heuristic distance to goal
    # heuristic choice depends on action space and cost model
    """
    
    def __init__(self, heuristic: Heuristic):
        """
        # initialize with specified heuristic function
        # manhattan for 4-action, chebyshev for 8-action uniform, 
        # octile for 8-action non-uniform
        """
        self.heuristic = heuristic
    
    def select_action(self, env, state: State, episode: int = 0) -> Action:
        """
        # select action that moves closest to goal according to heuristic
        # evaluates each action and chooses one with minimum heuristic value
        """
        valid_actions = env.get_valid_actions(state)
        if not valid_actions:
            # no valid moves - return random action
            return np.random.choice(env.action_space)
        
        # evaluate each valid action
        best_actions = []
        best_value = float('inf')
        
        for action in valid_actions:
            # calculate next state
            next_state = State(
                state.row + action.row_delta,
                state.col + action.col_delta
            )
            # calculate heuristic value for next state
            value = env.calculate_heuristic(next_state, self.heuristic)
            
            # keep track of best action(s)
            if value < best_value:
                best_value = value
                best_actions = [action]
            elif value == best_value:
                best_actions.append(action)
        
        # break ties randomly if multiple actions have same heuristic value
        return np.random.choice(best_actions)
    
    def get_name(self) -> str:
        return f"Greedy ({self.heuristic.value})"

class EpsilonGreedyPolicy(Policy):
    """
    # epsilon-greedy policy for exploration-exploitation balance
    # with probability epsilon, takes random action; otherwise acts greedily
    # addresses greedy policy's weakness of getting stuck in local optima
    """
    
    def __init__(self, epsilon: float = 0.1, heuristic: Heuristic = Heuristic.MANHATTAN):
        """
        # initialize with exploration rate and heuristic
        # epsilon controls exploration vs exploitation trade-off
        """
        self.epsilon = epsilon
        self.heuristic = heuristic
        # reuse greedy and random policies for clarity
        self.greedy_policy = GreedyPolicy(heuristic)
        self.random_policy = RandomPolicy()
    
    def select_action(self, env, state: State, episode: int = 0) -> Action:
        """
        # select action using epsilon-greedy strategy
        # explores with probability epsilon, exploits otherwise
        """
        if np.random.random() < self.epsilon:
            # explore - take random action
            return self.random_policy.select_action(env, state, episode)
        else:
            # exploit - take greedy action
            return self.greedy_policy.select_action(env, state, episode)
    
    def get_name(self) -> str:
        return f"ε-Greedy (ε={self.epsilon})"

class DecayingEpsilonGreedyPolicy(Policy):
    """
    # epsilon-greedy with exponentially decaying exploration rate
    # starts with high exploration and gradually shifts to exploitation
    # mimics natural learning process where agent explores initially then exploits knowledge
    """
    
    def __init__(self, initial_epsilon: float = 1.0, decay_rate: float = 0.1, 
                 min_epsilon: float = 0.01, heuristic: Heuristic = Heuristic.MANHATTAN):
        """
        # initialize with decay parameters
        # uses exponential decay to gradually reduce exploration
        # initial_epsilon: starting exploration rate (1.0 = pure random)
        # decay_rate: how fast epsilon decays
        # min_epsilon: minimum exploration to maintain
        """
        self.initial_epsilon = initial_epsilon
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon
        self.heuristic = heuristic
        self.greedy_policy = GreedyPolicy(heuristic)
        self.random_policy = RandomPolicy()
    
    def select_action(self, env, state: State, episode: int = 0) -> Action:
        """
        # select action with decaying exploration rate
        # epsilon decreases exponentially with episode number
        # allows agent to learn over time, exploring less as it gains experience
        """
        # calculate current epsilon with exponential decay
        # formula: epsilon = max(min_eps, initial_eps * e^(-decay_rate * episode))
        epsilon = max(self.min_epsilon, 
                     self.initial_epsilon * np.exp(-self.decay_rate * episode))
        
        if np.random.random() < epsilon:
            # explore
            return self.random_policy.select_action(env, state, episode)
        else:
            # exploit
            return self.greedy_policy.select_action(env, state, episode)
    
    def get_name(self) -> str:
        return f"Decaying ε-Greedy"