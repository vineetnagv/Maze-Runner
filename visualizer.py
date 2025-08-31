# visualizer.py
# Visualization tools for maze and trajectories

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import List, Optional, Dict
from state_action import State

class MazeVisualizer:
    """
    # Visualization tools for maze environment and agent trajectories
    # I create clear visual representations of the maze and solution paths
    """
    
    def __init__(self, env):
        """
        # Initialize visualizer with environment reference
        """
        self.env = env
    
    def visualize_maze(self, trajectory: Optional[List[State]] = None,
                      title: Optional[str] = None, 
                      save_path: Optional[str] = None,
                      show: bool = True):
        """
        # Visualize the maze with optional trajectory overlay
        # I draw walls, start, goal, and the agent's path if provided
        """
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw grid lines
        for i in range(self.env.grid_size + 1):
            ax.axhline(y=i, color='gray', linewidth=0.5, alpha=0.3)
            ax.axvline(x=i, color='gray', linewidth=0.5, alpha=0.3)
        
        # Draw walls as black squares
        for wall in self.env.walls:
            rect = patches.Rectangle((wall[1], wall[0]), 1, 1,
                                    linewidth=1, edgecolor='black',
                                    facecolor='black')
            ax.add_patch(rect)
        
        # Draw start position as green circle
        start_circle = patches.Circle((self.env.start.col + 0.5, 
                                      self.env.start.row + 0.5),
                                     0.3, color='green', zorder=5)
        ax.add_patch(start_circle)
        ax.text(self.env.start.col + 0.5, self.env.start.row + 0.5, 'S',
               ha='center', va='center', color='white', 
               fontweight='bold', fontsize=14, zorder=6)
        
        # Draw goal position as red circle
        goal_circle = patches.Circle((self.env.goal.col + 0.5, 
                                     self.env.goal.row + 0.5),
                                    0.3, color='red', zorder=5)
        ax.add_patch(goal_circle)
        ax.text(self.env.goal.col + 0.5, self.env.goal.row + 0.5, 'G',
               ha='center', va='center', color='white', 
               fontweight='bold', fontsize=14, zorder=6)
        
        # Draw trajectory if provided
        if trajectory and len(trajectory) > 1:
            # Draw path as arrows between consecutive states
            for i in range(len(trajectory) - 1):
                x1 = trajectory[i].col + 0.5
                y1 = trajectory[i].row + 0.5
                x2 = trajectory[i+1].col + 0.5
                y2 = trajectory[i+1].row + 0.5
                
                # Use color gradient to show progression
                color = plt.cm.viridis(i / (len(trajectory) - 1))
                
                # Draw arrow from current to next position
                ax.arrow(x1, y1, x2 - x1, y2 - y1,
                        head_width=0.15, head_length=0.1,
                        fc=color, ec=color, alpha=0.7,
                        length_includes_head=True, zorder=3)
        
        # Set axis properties
        ax.set_xlim(0, self.env.grid_size)
        ax.set_ylim(0, self.env.grid_size)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Invert y-axis so (0,0) is top-left
        
        # Add labels and title
        ax.set_xlabel('Column', fontsize=12)
        ax.set_ylabel('Row', fontsize=12)
        
        # Set title
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title(f'Maze: {self.env.maze_config.upper()}', 
                        fontsize=14, fontweight='bold')
        
        # Add grid coordinates
        ax.set_xticks(range(self.env.grid_size))
        ax.set_yticks(range(self.env.grid_size))
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Visualization saved to: {save_path}")
        
        # Show plot
        if show:
            plt.show()
        else:
            plt.close()
    
    def visualize_all_trajectories(self, results: Dict, 
                                  save_path: Optional[str] = None,
                                  show: bool = True):
        """
        # Visualize best trajectory for each policy side by side
        # I create a grid showing how different policies navigate the maze
        """
        policies = list(results.keys())
        num_policies = len(policies)
        
        # Calculate grid dimensions for subplots
        cols = min(3, num_policies)
        rows = (num_policies + cols - 1) // cols
        
        # Create figure with subplots
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        
        # Handle single policy case
        if num_policies == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Plot each policy's best trajectory
        for idx, policy_name in enumerate(policies):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            # Get best (shortest) trajectory for this policy
            episodes = results[policy_name]['episodes']
            successful_episodes = [ep for ep in episodes if ep['reached_goal']]
            
            if successful_episodes:
                # Find shortest successful trajectory
                best_episode = min(successful_episodes, key=lambda e: e['steps'])
                trajectory = best_episode['trajectory']
            else:
                # If no successful episodes, use first episode
                trajectory = episodes[0]['trajectory'] if episodes else []
            
            # Draw maze for this subplot
            self._draw_maze_subplot(ax, trajectory, policy_name)
        
        # Hide unused subplots
        for idx in range(num_policies, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')
        
        # Add main title
        fig.suptitle(f'Policy Comparison - {self.env.maze_config.upper()} Maze',
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Comparison saved to: {save_path}")
        
        # Show plot
        if show:
            plt.show()
        else:
            plt.close()
    
    def _draw_maze_subplot(self, ax, trajectory: List[State], title: str):
        """
        # Helper method to draw maze in a subplot
        # I reuse this for creating comparison visualizations
        """
        # Draw grid lines
        for i in range(self.env.grid_size + 1):
            ax.axhline(y=i, color='gray', linewidth=0.3, alpha=0.3)
            ax.axvline(x=i, color='gray', linewidth=0.3, alpha=0.3)
        
        # Draw walls
        for wall in self.env.walls:
            rect = patches.Rectangle((wall[1], wall[0]), 1, 1,
                                    linewidth=0.5, edgecolor='black',
                                    facecolor='black')
            ax.add_patch(rect)
        
        # Draw start
        ax.plot(self.env.start.col + 0.5, self.env.start.row + 0.5,
               'go', markersize=10, label='Start')
        
        # Draw goal
        ax.plot(self.env.goal.col + 0.5, self.env.goal.row + 0.5,
               'ro', markersize=10, label='Goal')
        
        # Draw trajectory
        if trajectory and len(trajectory) > 1:
            for i in range(len(trajectory) - 1):
                x1 = trajectory[i].col + 0.5
                y1 = trajectory[i].row + 0.5
                x2 = trajectory[i+1].col + 0.5
                y2 = trajectory[i+1].row + 0.5
                
                color = plt.cm.plasma(i / (len(trajectory) - 1))
                ax.plot([x1, x2], [y1, y2], color=color, 
                       linewidth=2, alpha=0.7)
        
        # Set axis properties
        ax.set_xlim(0, self.env.grid_size)
        ax.set_ylim(0, self.env.grid_size)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    def visualize_performance_distributions(self, results: Dict, 
                                           maze_name: str, action_space: str,
                                           save_path: Optional[str] = None):
        """
        # Visualize distribution of performance metrics (steps, rewards)
        # I use boxplots to show variability and central tendency
        """
        policies = list(results.keys())
        num_policies = len(policies)
        
        # Create figure with subplots for steps and rewards
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Data for boxplots
        steps_data = [results[p]['steps_list'] for p in policies]
        rewards_data = [results[p]['rewards_list'] for p in policies]
        
        # Boxplot for steps
        ax1.boxplot(steps_data, labels=policies, vert=False, patch_artist=True)
        ax1.set_title('Distribution of Steps per Episode', fontsize=12)
        ax1.set_xlabel('Number of Steps', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # Boxplot for rewards
        ax2.boxplot(rewards_data, labels=policies, vert=False, patch_artist=True)
        ax2.set_title('Distribution of Total Reward per Episode', fontsize=12)
        ax2.set_xlabel('Total Reward', fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        # Main title
        fig.suptitle(f'Performance Distributions - {maze_name.upper()} ({action_space})',
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Distribution plot saved to: {save_path}")
        
        plt.close()

    def visualize_distributions_histogram(self, results: Dict,
                                           maze_name: str, action_space: str,
                                           save_path: Optional[str] = None):
        """
        # Visualize distribution of performance metrics (steps, rewards) using histograms
        """
        policies = list(results.keys())
        num_policies = len(policies)
        
        # Create figure with subplots for steps and rewards
        fig, axes = plt.subplots(num_policies, 2, figsize=(12, 4 * num_policies), sharex='col')
        
        if num_policies == 1:
            axes = [axes]

        for i, policy_name in enumerate(policies):
            steps_data = results[policy_name]['steps_list']
            rewards_data = results[policy_name]['rewards_list']

            # Histogram for steps
            axes[i][0].hist(steps_data, bins=20, color='skyblue', edgecolor='black')
            axes[i][0].set_title(f'{policy_name} - Steps Distribution', fontsize=10)
            axes[i][0].set_ylabel('Frequency', fontsize=8)
            if i == num_policies - 1:
                axes[i][0].set_xlabel('Number of Steps', fontsize=8)

            # Histogram for rewards
            axes[i][1].hist(rewards_data, bins=20, color='salmon', edgecolor='black')
            axes[i][1].set_title(f'{policy_name} - Rewards Distribution', fontsize=10)
            if i == num_policies - 1:
                axes[i][1].set_xlabel('Total Reward', fontsize=8)

        fig.suptitle(f'Performance Histograms - {maze_name.upper()} ({action_space})',
                     fontsize=16, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Histogram plot saved to: {save_path}")
        
        plt.close()

    def visualize_learning_curves(self, results: Dict,
                                   maze_name: str, action_space: str,
                                   save_path: Optional[str] = None):
        """
        # Visualize learning curves for all policies
        """
        plt.figure(figsize=(12, 8))
        
        for policy_name, data in results.items():
            episodes = data['episodes']
            steps_per_episode = [ep['steps'] for ep in episodes]
            
            # Calculate moving average for smoother curve
            window = 5
            if len(steps_per_episode) >= window:
                moving_avg = np.convolve(steps_per_episode, np.ones(window)/window, mode='valid')
                plt.plot(range(window-1, len(steps_per_episode)), moving_avg, label=f'{policy_name} (Avg)')
            else:
                plt.plot(steps_per_episode, label=policy_name, alpha=0.6)

        plt.title(f'Learning Curves - {maze_name.upper()} ({action_space})', fontsize=16, fontweight='bold')
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Number of Steps', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Learning curves saved to: {save_path}")
        
        plt.close()