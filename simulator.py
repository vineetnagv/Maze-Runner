# simulator.py
# simulation engine for running episodes and collecting statistics

import numpy as np
from typing import Dict, List, Optional
from tabulate import tabulate
from state_action import State, Action

class MazeSimulator:
    """
    # simulation engine for running maze navigation episodes
    # collects trajectories, rewards, and performance statistics
    """
    
    def __init__(self, env, max_steps: int = 500):
        """
        # initialize simulator with environment and step limit
        # max_steps prevents infinite loops if agent gets stuck
        """
        self.env = env
        self.max_steps = max_steps
    
    def run_episode(self, policy, episode_num: int = 0, 
                   verbose: bool = False) -> Dict:
        """
        # run single episode with given policy
        # returns complete statistics for episode
        """
        # reset policy if it maintains internal state
        policy.reset()
        
        # initialize episode
        state = self.env.start
        trajectory = [state]
        actions = []
        rewards = []
        done = False
        steps = 0
        
        # run episode until goal reached or max steps
        while not done and steps < self.max_steps:
            # select action using policy
            action = policy.select_action(self.env, state, episode_num)
            actions.append(action)
            
            # execute action in environment
            next_state, reward, done = self.env.transition(state, action)
            rewards.append(reward)
            trajectory.append(next_state)
            
            # update state
            state = next_state
            steps += 1
        
        # calculate episode statistics
        # total undiscounted reward: simple sum of all rewards
        total_reward = sum(rewards)
        
        # total discounted return: gt = rt+1 + γ*rt+2 + γ^2*rt+3 + ...
        # this is the primary performance metric as specified
        discounted_return = 0.0
        for t, reward in enumerate(rewards):
            discounted_return += reward * (self.env.gamma ** t)
        
        # compile episode data with all required metrics
        episode_data = {
            'steps': steps,  # number of steps taken
            'total_reward': total_reward,  # total undiscounted reward
            'discounted_return': discounted_return,  # total discounted return (primary metric)
            'trajectory': trajectory,
            'actions': actions,
            'rewards': rewards,
            'success': done,
            'reached_goal': state == self.env.goal
        }
        
        # print trajectory if verbose mode
        if verbose:
            self._print_trajectory(episode_data, policy.get_name())
        
        return episode_data
    
    def run_experiments(self, policies: List, num_episodes: int = 20,
                       verbose: bool = False) -> Dict:
        """
        # run experiments with multiple policies
        # collects comprehensive statistics for comparison
        """
        results = {}
        
        print("\n" + "="*80)
        print("RUNNING SIMULATION EXPERIMENTS")
        print("="*80)
        
        for policy in policies:
            print(f"\ntesting {policy.get_name()} policy...")
            policy_stats = []
            
            # run episodes for this policy
            for episode in range(num_episodes):
                # run episode (verbose only for first episode)
                episode_data = self.run_episode(
                    policy, episode, 
                    verbose=(verbose and episode == 0)
                )
                policy_stats.append(episode_data)
                
                # progress indicator
                if (episode + 1) % 5 == 0:
                    print(f"  completed {episode + 1}/{num_episodes} episodes")
            
            # calculate aggregate statistics
            steps_list = [ep['steps'] for ep in policy_stats]
            rewards_list = [ep['total_reward'] for ep in policy_stats]
            returns_list = [ep['discounted_return'] for ep in policy_stats]
            success_list = [ep['success'] for ep in policy_stats]
            
            # store results for this policy
            results[policy.get_name()] = {
                'episodes': policy_stats,
                'steps_list': steps_list,
                'rewards_list': rewards_list,
                'avg_steps': np.mean(steps_list),
                'min_steps': np.min(steps_list),
                'max_steps': np.max(steps_list),
                'std_steps': np.std(steps_list),
                'avg_reward': np.mean(rewards_list),
                'min_reward': np.min(rewards_list),
                'max_reward': np.max(rewards_list),
                'std_reward': np.std(rewards_list),
                'avg_return': np.mean(returns_list),
                'success_rate': np.mean(success_list)
            }
        
        return results
    
    def _print_trajectory(self, episode_data: Dict, policy_name: str):
        """
        # print detailed trajectory for an episode
        # shows complete path from start to goal with states and actions
        """
        print(f"\n{'-'*60}")
        print(f"SAMPLE TRAJECTORY - {policy_name} Policy")
        print(f"{'-'*60}")
        
        trajectory = episode_data['trajectory']
        actions = episode_data['actions']
        rewards = episode_data['rewards']
        
        # print complete trajectory information
        print(f"\nSTATES VISITED (complete list):")
        print(f"start: {trajectory[0]}")
        
        # print all states and actions taken
        states_str = " -> ".join([str(s) for s in trajectory[:15]])  # first 15 states
        if len(trajectory) > 15:
            states_str += f" -> ... ({len(trajectory)-15} more states) -> {trajectory[-1]}"
        print(f"path: {states_str}")
        
        print(f"\nACTIONS TAKEN (step-by-step):")
        # print detailed steps
        for i, action in enumerate(actions[:10]):  # show first 10 steps in detail
            print(f"  step {i+1:3d}: state {trajectory[i]} -> action {action.name:10s} -> state {trajectory[i+1]} (reward={rewards[i]:+.1f})")
        
        if len(actions) > 10:
            print(f"  ... ({len(actions) - 10} more steps)")
            # show last few steps
            for i in range(max(10, len(actions)-3), len(actions)):
                print(f"  step {i+1:3d}: state {trajectory[i]} -> action {actions[i].name:10s} -> state {trajectory[i+1]} (reward={rewards[i]:+.1f})")
        
        # print episode summary
        print(f"\nEPISODE SUMMARY:")
        print(f"  goal reached: {'YES' if episode_data['reached_goal'] else 'NO'}")
        print(f"  total steps taken: {episode_data['steps']}")
        print(f"  total undiscounted reward: {episode_data['total_reward']:.1f}")
        print(f"  total discounted return: {episode_data['discounted_return']:.1f}")
    
    def print_summary_table(self, results: Dict, maze_name: str = "", 
                           action_space: str = "", save_path: str = None):
        """
        # print summary statistics in tabular format
        # creates clean table for easy comparison of policies
        # includes total discounted return as primary metric
        # saves table as png image if path provided
        """
        print("\n" + "="*80)
        print("SUMMARY STATISTICS (20 EPISODES)")
        if maze_name or action_space:
            print(f"maze: {maze_name.upper()} | action space: {action_space}")
        print("="*80)
        
        # prepare data for table including discounted return
        table_data = []
        headers = ["Policy", "Avg Steps", "Min/Max Steps", 
                  "Avg Total Reward", "Avg Discounted Return", "Success Rate"]
        
        for policy_name, stats in results.items():
            row = [
                policy_name,
                f"{stats['avg_steps']:.1f} ± {stats['std_steps']:.1f}",
                f"{stats['min_steps']:.0f}/{stats['max_steps']:.0f}",
                f"{stats['avg_reward']:.1f} ± {stats['std_reward']:.1f}",
                f"{stats['avg_return']:.1f}",  # primary performance metric
                f"{stats['success_rate']*100:.1f}%"
            ]
            table_data.append(row)
        
        # print table
        table_str = tabulate(table_data, headers=headers, tablefmt="grid")
        print(table_str)
        
        # save table as image if path provided
        if save_path:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.axis('tight')
            ax.axis('off')
            
            # create table
            table = ax.table(cellText=[headers] + table_data,
                           cellLoc='center',
                           loc='center',
                           colWidths=[0.2, 0.2, 0.15, 0.2, 0.15, 0.1])
            
            # style header row
            for i in range(len(headers)):
                table[(0, i)].set_facecolor('#40466e')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # style data rows
            for i in range(1, len(table_data) + 1):
                for j in range(len(headers)):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#f0f0f0')
            
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # add title
            title = f"Summary Statistics - {maze_name.upper()} Maze"
            if action_space:
                title += f"\nAction Space: {action_space}"
            plt.title(title, fontsize=14, fontweight='bold', pad=20)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  summary table saved to: {save_path}")
    
    def analyze_results(self, results: Dict):
        """
        # perform comparative analysis of results
        # identifies best performing policies based on data
        """
        print("\n" + "="*80)
        print("COMPARATIVE ANALYSIS")
        print("="*80)
        
        # find best policy by average steps
        best_policy = min(results.keys(), 
                         key=lambda k: results[k]['avg_steps'])
        worst_policy = max(results.keys(), 
                          key=lambda k: results[k]['avg_steps'])
        
        print(f"\nbest performing policy (fewest steps): {best_policy}")
        print(f"  average steps: {results[best_policy]['avg_steps']:.1f}")
        print(f"  success rate: {results[best_policy]['success_rate']*100:.1f}%")
        
        print(f"\nworst performing policy (most steps): {worst_policy}")
        print(f"  average steps: {results[worst_policy]['avg_steps']:.1f}")
        print(f"  success rate: {results[worst_policy]['success_rate']*100:.1f}%")
        
        # calculate improvement
        if results[worst_policy]['avg_steps'] > 0:
            improvement = ((results[worst_policy]['avg_steps'] - 
                          results[best_policy]['avg_steps']) / 
                          results[worst_policy]['avg_steps'] * 100)
            print(f"\nimprovement: {improvement:.1f}% reduction in steps")
        
        # analyze exploration benefits if epsilon-greedy present
        epsilon_policies = [p for p in results if "ε-Greedy" in p]
        greedy_policies = [p for p in results if "Greedy" in p and "ε" not in p]
        
        if epsilon_policies and greedy_policies:
            print("\n" + "-"*60)
            print("exploration vs exploitation analysis:")
            for greedy in greedy_policies:
                print(f"\n{greedy}:")
                print(f"  success rate: {results[greedy]['success_rate']*100:.1f}%")
                print(f"  avg steps: {results[greedy]['avg_steps']:.1f}")
            
            for epsilon in epsilon_policies:
                print(f"\n{epsilon}:")
                print(f"  success rate: {results[epsilon]['success_rate']*100:.1f}%")
                print(f"  avg steps: {results[epsilon]['avg_steps']:.1f}")
        
        # check for learning behavior in decaying epsilon
        if "Decaying ε-Greedy" in results:
            episodes = results["Decaying ε-Greedy"]['episodes']
            early_steps = np.mean([ep['steps'] for ep in episodes[:5]])
            late_steps = np.mean([ep['steps'] for ep in episodes[-5:]])
            
            print("\n" + "-"*60)
            print("learning analysis (decaying epsilon):")
            print(f"  early episodes (1-5): {early_steps:.1f} avg steps")
            print(f"  late episodes (16-20): {late_steps:.1f} avg steps")
            
            if late_steps < early_steps:
                improvement = (early_steps - late_steps) / early_steps * 100
                print(f"  improvement: {improvement:.1f}% reduction")
            else:
                print(f"  no improvement observed")