# Maze Runner MDP Analysis

This project provides a framework for analyzing and comparing various pathfinding policies within a maze environment, modeled as a Markov Decision Process (MDP). It explores how different agent strategies perform under various maze configurations and action spaces, offering detailed analytics and visualizations to understand their behavior.

## Features

- **Multiple Maze Configurations:** Test policies on three different maze layouts to analyze symmetry:
  - `original`: The standard maze configuration.
  - `swapped`: The start and goal positions are reversed to test for directional symmetry.
  - `flipped`: The maze is horizontally mirrored to test for geometric bias in policies.

- **Flexible Action Spaces:** Supports both 4-action and 8-action movement spaces:
  - **4-Action:** Cardinal directions only (Up, Down, Left, Right).
  - **8-Action (Uniform Cost):** Includes diagonal movements, with all moves having a uniform cost.
  - **8-Action (Non-Uniform Cost):** Diagonal moves have a higher cost (sqrt(2)), reflecting true Euclidean distance.

- **Variety of Policies:** Compare the performance of several classic reinforcement learning policies:
  - **Random Policy:** Acts as a baseline by choosing actions randomly.
  - **Greedy Policy:** Always chooses the action that it heuristically estimates is best.
  - **Epsilon-Greedy Policy:** A simple exploration/exploitation strategy.
  - **Decaying Epsilon-Greedy Policy:** A more advanced policy that starts with high exploration and gradually becomes more greedy over time, mimicking a learning process.

- **Comprehensive Simulation and Tracking:** The simulation engine runs policies for a specified number of episodes and tracks key performance metrics:
  - Number of steps per episode.
  - Total undiscounted reward.
  - Total discounted return (the primary metric for performance).

- **Rich Visualizations:** The project generates a suite of visualizations to provide a clear understanding of policy performance:
  - **Best Path Trajectory:** A plot of the most successful trajectory for the best-performing policy.
  - **Policy Comparison:** A side-by-side view of the best paths taken by each policy.
  - **Performance Distributions (Box Plots):** Shows the distribution of steps and rewards for each policy across all episodes.
  - **Performance Histograms:** Provides a frequency-based view of the distribution of steps and rewards.
  - **Learning Curves:** A line chart that plots the number of steps per episode, visualizing the learning trend of each policy.

- **In-Depth Symmetry Analysis:** The script automatically performs and reports on:
  - **Directional Symmetry:** Compares performance on the `original` vs. `swapped` mazes.
  - **Geometric Symmetry:** Compares performance on the `original` vs. `flipped` mazes.

- **Persistent Results:** All experimental data is saved for future analysis:
  - All generated plots are saved as `.png` files.
  - A `complete_tracking_data.json` file stores the detailed results of every episode.
  - The complete terminal output of the script is logged to a timestamped text file.

## File Structure

```
.
├── environment.py
├── main.py
├── policies.py
├── simulator.py
├── state_action.py
├── visualizer.py
├── requirements.txt
└── README.md
```

- `main.py`: The main entry point for running the entire suite of experiments and analyses.
- `environment.py`: Defines the maze world, including the grid, walls, states, actions, and reward structure of the MDP.
- `policies.py`: Implements all the different agent policies (Random, Greedy, etc.).
- `simulator.py`: The core simulation engine responsible for running episodes, collecting data, and calculating statistics.
- `state_action.py`: Contains the core data structures for the project, such as `State` and `Action`.
- `visualizer.py`: The visualization toolkit for generating all the plots and charts.
- `requirements.txt`: A list of all the Python libraries required to run the project.
- `README.md`: This file.

## How to Run

1.  **Install Dependencies:** Make sure you have Python installed. Then, install the required libraries using pip:

    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Experiment:** Execute the main script from your terminal:

    ```bash
    python main.py
    ```

## Understanding the Output

After running the script, the directory you run the script in will contain several new files:

- **Trajectory Visualizations (`maze_*.png`):** These images show the paths taken by the agents.
- **Comparison Visualizations (`comparison_*.png`):** These images show side-by-side comparisons of the different policies.
- **Summary Tables (`summary_*.png`):** These images contain tables summarizing the performance statistics for each experiment.
- **Distribution Plots (`distribution_*.png`, `histogram_*.png`):** These images show the distribution of performance metrics.
- **Learning Curves (`learning_curves_*.png`):** These images show the learning progress of the policies over time.
- **`complete_tracking_data.json`:** This file contains the raw data from every episode of every experiment, allowing for further, more detailed analysis if desired.
- **`terminal_output_*.txt`:** A log of everything that was printed to the console during the execution of the script.
