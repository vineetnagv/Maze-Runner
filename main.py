# main.py
# main execution script for mdp maze world simulation

import numpy as np
import sys
import json
from datetime import datetime
from environment import MazeWorld
from policies import (RandomPolicy, GreedyPolicy, EpsilonGreedyPolicy,
                     DecayingEpsilonGreedyPolicy)
from simulator import MazeSimulator
from visualizer import MazeVisualizer
from state_action import Heuristic, CostModel

class OutputLogger:
    """
    # captures terminal output and saves to file
    # allows us to save all experiment results
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        # open file with utf-8 encoding to handle special characters
        self.log = open(filename, 'w', encoding='utf-8')
    
    def write(self, message):
        # write to both terminal and file
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # ensure immediate write
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

def save_tracking_data(all_results: dict, filename: str):
    """
    # save all tracking data to json file
    # includes trajectories, rewards, steps for every episode
    """
    # convert state objects to tuples for json serialization
    tracking_data = {}
    
    for config_key, results in all_results.items():
        tracking_data[config_key] = {}
        
        for policy_name, policy_data in results.items():
            # save summary statistics
            # convert numpy types to python native types for json
            tracking_data[config_key][policy_name] = {
                'summary': {
                    'avg_steps': float(policy_data['avg_steps']),
                    'min_steps': int(policy_data['min_steps']),
                    'max_steps': int(policy_data['max_steps']),
                    'std_steps': float(policy_data['std_steps']),
                    'avg_total_reward': float(policy_data['avg_reward']),
                    'min_reward': float(policy_data['min_reward']),
                    'max_reward': float(policy_data['max_reward']),
                    'std_reward': float(policy_data['std_reward']),
                    'avg_discounted_return': float(policy_data['avg_return']),
                    'success_rate': float(policy_data['success_rate'])
                },
                'episodes': []
            }
            
            # save each episode's data
            for ep_idx, episode in enumerate(policy_data['episodes']):
                episode_data = {
                    'episode_number': ep_idx + 1,
                    'steps': int(episode['steps']),
                    'total_undiscounted_reward': float(episode['total_reward']),
                    'total_discounted_return': float(episode['discounted_return']),
                    'reached_goal': bool(episode['reached_goal']),
                    'trajectory': [state.to_tuple() for state in episode['trajectory']],
                    'actions': [action.name for action in episode['actions']],
                    'rewards': [float(r) for r in episode['rewards']]
                }
                tracking_data[config_key][policy_name]['episodes'].append(episode_data)
    
    # save to json file
    with open(filename, 'w') as f:
        json.dump(tracking_data, f, indent=2)
    
    print(f"\ntracking data saved to: {filename}")

def run_single_maze_experiment(maze_config: str, use_8_actions: bool = False,
                              cost_model: CostModel = CostModel.UNIFORM,
                              visualize: bool = True):
    """
    # run experiment on single maze configuration
    # tests all policies and compares performance
    # tracks steps, total reward, and discounted return
    """
    action_desc = "8-action" if use_8_actions else "4-action"
    cost_desc = f"({cost_model.value} cost)" if use_8_actions else ""
    full_action_desc = f"{action_desc} {cost_desc}".strip()
    
    print("\n" + "="*80)
    print(f"MAZE: {maze_config.upper()} - {full_action_desc}")
    print("="*80)
    
    # create environment with specified configuration
    env = MazeWorld(
        grid_size=8, 
        maze_config=maze_config,
        use_8_actions=use_8_actions,
        cost_model=cost_model,
        discount_factor=0.9
    )
    
    # print maze information
    info = env.get_maze_info()
    print(f"\nmaze information:")
    print(f"  grid size: {info['grid_size']}x{info['grid_size']}")
    print(f"  start position: {info['start']}")
    print(f"  goal position: {info['goal']}")
    print(f"  number of walls: {info['num_walls']}")
    print(f"  valid states: {info['num_valid_states']}")
    print(f"  action space: {info['action_space']}")
    print(f"  cost model: {info['cost_model']}")
    print(f"  optimal heuristic: {info['optimal_heuristic']}")
    print(f"  discount factor (γ): {info['discount_factor']}")
    
    # get optimal heuristic for this configuration
    optimal_heuristic = env.get_optimal_heuristic()
    
    # define policies to test
    # use optimal heuristic based on action space and cost model
    policies = [
        RandomPolicy(),
        GreedyPolicy(optimal_heuristic),
        EpsilonGreedyPolicy(epsilon=0.1, heuristic=optimal_heuristic),
        DecayingEpsilonGreedyPolicy(
            initial_epsilon=1.0, 
            decay_rate=0.1,
            min_epsilon=0.01,
            heuristic=optimal_heuristic
        )
    ]
    
    # run simulations
    simulator = MazeSimulator(env, max_steps=500)
    
    # run experiments with verbose output for first episode
    results = simulator.run_experiments(policies, num_episodes=20, verbose=True)
    
    # print distribution statistics for verification
    print("\n" + "="*60)
    print("DISTRIBUTION OF METRICS OVER 20 EPISODES")
    print("="*60)
    for policy_name in results:
        stats = results[policy_name]
        print(f"\n{policy_name}:")
        print(f"  steps distribution:")
        print(f"    minimum: {stats['min_steps']:.0f} steps")
        print(f"    maximum: {stats['max_steps']:.0f} steps")
        print(f"    average: {stats['avg_steps']:.1f} steps")
        print(f"    std dev: {stats['std_steps']:.1f} steps")
        print(f"  reward distribution:")
        print(f"    minimum: {stats['min_reward']:.1f}")
        print(f"    maximum: {stats['max_reward']:.1f}")
        print(f"    average: {stats['avg_reward']:.1f}")
        print(f"    std dev: {stats['std_reward']:.1f}")
        print(f"  success rate: {stats['success_rate']*100:.1f}%")
    
    # create filename for summary table
    table_filename = f"summary_{maze_config}_{action_desc.replace('-', '_')}"
    if use_8_actions:
        table_filename += f"_{cost_model.value}"
    table_filename += ".png"
    
    # print and save summary table
    simulator.print_summary_table(
        results, 
        maze_name=maze_config,
        action_space=full_action_desc,
        save_path=table_filename
    )
    
    # verify we're tracking the required metrics
    print("\n" + "-"*60)
    print("verification of tracked metrics:")
    for policy_name in list(results.keys())[:1]:  # show for first policy
        stats = results[policy_name]
        print(f"\n{policy_name}:")
        print(f"  average number of steps: {stats['avg_steps']:.1f}")
        print(f"  average total undiscounted reward: {stats['avg_reward']:.1f}")
        print(f"  average total discounted return: {stats['avg_return']:.1f}")
        print(f"  -> discounted return is primary performance metric")
    
    # analyze results
    simulator.analyze_results(results)
    
    # visualize if requested
    if visualize:
        visualizer = MazeVisualizer(env)
        
        print("\n" + "-"*60)
        print("generating visualizations...")
        
        # find best performing policy based on discounted return
        best_policy = max(results.keys(), 
                         key=lambda k: results[k]['avg_return'])
        best_episode = max(results[best_policy]['episodes'], 
                          key=lambda e: e['discounted_return'])
        
        # create filename with action space info
        filename_suffix = f"{maze_config}_{action_desc.replace('-', '_')}"
        if use_8_actions:
            filename_suffix += f"_{cost_model.value}"
        
        # visualize maze with best trajectory
        visualizer.visualize_maze(
            trajectory=best_episode['trajectory'],
            title=f"{maze_config.upper()} maze - best path ({best_policy}) - {full_action_desc}",
            save_path=f"maze_{filename_suffix}_best.png",
            show=False
        )
        
        # create comparison visualization
        visualizer.visualize_all_trajectories(
            results,
            save_path=f"maze_{filename_suffix}_compare.png",
            show=False
        )
        
        # create distribution visualization
        visualizer.visualize_performance_distributions(
            results,
            maze_name=maze_config,
            action_space=full_action_desc,
            save_path=f"distribution_{filename_suffix}.png"
        )

        # create histogram visualization
        visualizer.visualize_distributions_histogram(
            results,
            maze_name=maze_config,
            action_space=full_action_desc,
            save_path=f"histogram_{filename_suffix}.png"
        )

        # create learning curve visualization
        visualizer.visualize_learning_curves(
            results,
            maze_name=maze_config,
            action_space=full_action_desc,
            save_path=f"learning_curves_{filename_suffix}.png"
        )
    
    return results

def run_symmetry_analysis():
    """
    # analyze directional and geometric symmetry of mazes
    # tracks all required metrics across episodes
    """
    print("\n" + "="*80)
    print("SYMMETRY ANALYSIS")
    print("="*80)
    
    # test all three maze configurations
    configs = ["original", "swapped", "flipped"]
    
    # test both 4-action and 8-action spaces
    action_configs = [
        (False, CostModel.UNIFORM, "4-action"),
        (True, CostModel.UNIFORM, "8-action uniform"),
        (True, CostModel.NON_UNIFORM, "8-action non-uniform")
    ]
    
    # store results for analysis
    all_results = {}
    
    for use_8, cost_model, desc in action_configs:
        print(f"\n{'-'*60}")
        print(f"testing {desc} space...")
        
        for config in configs:
            results = run_single_maze_experiment(
                config, 
                use_8_actions=use_8,
                cost_model=cost_model,
                visualize=True
            )
            key = f"{config}_{desc}"
            all_results[key] = results
    
    # save all tracking data to file
    save_tracking_data(all_results, "complete_tracking_data.json")
    
    # perform symmetry analysis
    print("\n" + "="*80)
    print("SYMMETRY COMPARISON")
    print("="*80)
    
    for _, _, desc in action_configs:
        print(f"\n{desc.upper()} SPACE:")
        print("-" * 60)
        
        # directional symmetry: original vs swapped
        print("\ndirectional symmetry (original vs swapped):")
        print("  positive % = swapped is harder (more steps needed)")
        print("  negative % = swapped is easier (fewer steps needed)")
        print("  near 0% = symmetric difficulty in both directions\n")
        
        orig_key = f"original_{desc}"
        swap_key = f"swapped_{desc}"
        
        if orig_key in all_results and swap_key in all_results:
            for policy_name in all_results[orig_key]:
                orig_steps = all_results[orig_key][policy_name]['avg_steps']
                swap_steps = all_results[swap_key][policy_name]['avg_steps']
                diff = (swap_steps - orig_steps) / orig_steps * 100 if orig_steps > 0 else 0
                
                print(f"  {policy_name:30s}: {diff:+6.1f}%")
                
                # explain what the difference means
                if abs(diff) < 5:
                    print(f"    -> nearly symmetric, similar difficulty both ways")
                elif diff > 0:
                    print(f"    -> harder when navigating from (0,7) to (7,0)")
                else:
                    print(f"    -> easier when navigating from (0,7) to (7,0)")
        
        # geometric symmetry: original vs flipped
        print("\ngeometric symmetry (original vs flipped):")
        print("  positive % = flipped is harder (more steps needed)")
        print("  negative % = flipped is easier (fewer steps needed)")
        print("  near 0% = policies have no left/right bias\n")
        
        flip_key = f"flipped_{desc}"
        
        if orig_key in all_results and flip_key in all_results:
            for policy_name in all_results[orig_key]:
                orig_steps = all_results[orig_key][policy_name]['avg_steps']
                flip_steps = all_results[flip_key][policy_name]['avg_steps']
                diff = (flip_steps - orig_steps) / orig_steps * 100 if orig_steps > 0 else 0
                
                print(f"  {policy_name:30s}: {diff:+6.1f}%")
                
                # explain what the difference means
                if abs(diff) < 5:
                    print(f"    -> no directional bias, policies work equally well")
                elif diff > 0:
                    print(f"    -> policies perform worse on horizontally flipped maze")
                else:
                    print(f"    -> policies perform better on horizontally flipped maze")
    
    return all_results

def main():
    """
    # main entry point for program
    # runs comprehensive experiments and saves all data
    """
    # set random seed for reproducibility
    np.random.seed(42)
    
    # create output logger to save terminal output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = OutputLogger(f"terminal_output_{timestamp}.txt")
    sys.stdout = logger
    
    print("\n" + "="*80)
    print("MDP MAZE WORLD - COMPREHENSIVE ANALYSIS")
    print("="*80)
    print(f"\ntimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nexperiment tests different policies on three maze configurations:")
    print("1. original: standard maze with walls creating obstacles")
    print("2. swapped: original maze with start/goal reversed")
    print("3. flipped: horizontally mirrored original maze")
    print("\naction spaces tested:")
    print("- 4-action: cardinal movements only (up/down/left/right)")
    print("- 8-action uniform: includes diagonals, all moves cost -1")
    print("- 8-action non-uniform: diagonals cost -sqrt(2) for true distance")
    print("\nperformance metrics tracked:")
    print("- number of steps taken per episode")
    print("- total undiscounted reward per episode")
    print("- total discounted return per episode (primary metric)")
    print("  formula: Gt = Rt+1 + gamma*Rt+2 + gamma^2*Rt+3 + ...")
    print("  with discount factor gamma = 0.9")
    
    # run symmetry analysis which includes all experiments
    results = run_symmetry_analysis()
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print("\nfiles generated:")
    print("- summary_*.png: summary tables for each configuration")
    print("- maze_*.png: trajectory visualizations")
    print("- complete_tracking_data.json: all episode data")
    print(f"- terminal_output_{timestamp}.txt: complete terminal output")
    print("="*80)
    
    # close logger
    logger.close()
    sys.stdout = logger.terminal
    
    print(f"\nall output saved to terminal_output_{timestamp}.txt")

if __name__ == "__main__":
    main()