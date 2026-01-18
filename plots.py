# plot_metrics.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)




def plot_hourly_model_vs_baseline(metrics_hourly_path, baseline_df, save_dir='runs_ddqn_fixed'):
    hourly_model = pd.read_csv(metrics_hourly_path)
    baseline_mean = baseline_df.groupby('hour')['avg_count'].mean().reset_index()

    plt.figure(figsize=(10,6))
    plt.plot(baseline_mean['hour'], baseline_mean['avg_count'], label='Baseline Avg Density', marker='o')
    plt.plot(hourly_model['hour'], hourly_model['model_avg_density'], label='Model Avg Density', marker='x')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Density per Segment')
    plt.title('Hourly Model vs Baseline Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'baseline_vs_model_hourly_density.png'))
    plt.close()


def plot_training_metrics(metrics_csv_path, baseline_df=None, save_dir=None):
    """
    Plot learning and traffic metrics from training logs.
    If baseline_df is provided (from build_hourly_baselines), it compares model vs baseline for density.

    Args:
        metrics_csv_path: str, path to metrics_final.csv
        baseline_df: pandas.DataFrame or None
        save_dir: optional directory to save figures (defaults to metrics_csv folder)
    """
    if not os.path.exists(metrics_csv_path):
        raise FileNotFoundError(f"{metrics_csv_path} not found")

    df = pd.read_csv(metrics_csv_path)
    if save_dir is None:
        save_dir = os.path.dirname(metrics_csv_path)
    _ensure_dir(save_dir)

    # --- Basic Training Metrics ---
    plt.figure(figsize=(8,5))
    plt.plot(df['episode'], df['episode_reward'], label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Episode Reward Over Time')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'episode_reward.png'))
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(df['episode'], df['avg_reward_per_agent'], label='Avg Reward/Agent', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Avg Reward per Agent')
    plt.title('Average Agent Reward')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'avg_agent_reward.png'))
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(df['episode'], df['loss'], label='Loss', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss.png'))
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(df['episode'], df['collision_count'], label='Collisions', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Collisions per Episode')
    plt.title('Collision Count')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'collisions.png'))
    plt.close()

    # --- Average Density & Congestion Index ---
    plt.figure(figsize=(8,5))
    plt.plot(df['episode'], df['avg_density'], label='Model Avg Density', color='purple')
    plt.xlabel('Episode')
    plt.ylabel('Avg Density')
    plt.title('Average Density per Episode')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'avg_density.png'))
    plt.close()

    # Congestion index = normalized density (0 to 1)
    cong_index = (df['avg_density'] - df['avg_density'].min()) / (df['avg_density'].max() - df['avg_density'].min() + 1e-9)
    plt.figure(figsize=(8,5))
    plt.plot(df['episode'], cong_index, label='Congestion Index', color='brown')
    plt.xlabel('Episode')
    plt.ylabel('Congestion Index')
    plt.title('Congestion Index Trend')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'congestion_index.png'))
    plt.close()

    # --- Compare Model Density vs Baseline ---
    if baseline_df is not None:
        # compute hourly baseline avg
        baseline_hourly = baseline_df.groupby('hour')['avg_count'].mean()
        model_mean_density = df['avg_density'].mean()

        plt.figure(figsize=(8,5))
        plt.plot(baseline_hourly.index, baseline_hourly.values, label='Baseline Avg Density', marker='o')
        plt.hlines(model_mean_density, xmin=0, xmax=23, colors='purple', linestyles='--', label='Model Mean Density')
        plt.xlabel('Hour of Day')
        plt.ylabel('Density (avg)')
        plt.title('Model vs Baseline Density Comparison')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'baseline_vs_model_density.png'))
        plt.close()

    print(f"✅ Plots saved in: {save_dir}")


def summarize_results(metrics_csv_path, baseline_df=None):
    """
    Print a brief summary of training performance and comparison to baseline.
    """
    df = pd.read_csv(metrics_csv_path)
    print("\n=== Training Summary ===")
    print(f"Episodes: {len(df)}")
    print(f"Final Avg Reward per Agent: {df['avg_reward_per_agent'].iloc[-1]:.3f}")
    print(f"Final Loss: {df['loss'].iloc[-1]:.4f}")
    print(f"Mean Collisions: {df['collision_count'].mean():.2f}")
    print(f"Mean Density: {df['avg_density'].mean():.3f}")
    if baseline_df is not None:
        baseline_mean = baseline_df['avg_count'].mean()
        print(f"Baseline Mean Density: {baseline_mean:.3f}")
        diff = df['avg_density'].mean() - baseline_mean
        print(f"Δ Model vs Baseline Density: {diff:+.3f}")
    print("=========================\n")
