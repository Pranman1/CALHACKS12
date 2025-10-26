#!/usr/bin/env python3
"""
Quick Training Script for Skateboard Balancing
Optimized for fast convergence and testing
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run quick training with optimized parameters."""
    
    # Set up paths
    project_root = Path(__file__).parent
    unitree_path = project_root / "unitree_rl_lab"
    train_script = unitree_path / "scripts" / "rsl_rl" / "train.py"
    
    # Set environment variables
    env = os.environ.copy()
    env["ISAACLAB_PATH"] = str(unitree_path)
    env["PYTHONPATH"] = f"{unitree_path}/source:{env.get('PYTHONPATH', '')}"
    
    # Training parameters optimized for speed
    training_args = [
        sys.executable, str(train_script),
        "--task", "Unitree-G1-29DOF-Skateboard-v0",
        "--num_envs", "8192",           # More environments for faster data collection
        "--max_iterations", "1000",      # Reduced for quick testing
        "--batch_size", "16384",         # Smaller batch for faster updates
        "--mini_batch_size", "512",      # Smaller mini-batch
        "--learning_rate", "3e-4",       # Standard PPO learning rate
        "--save_interval", "100",         # Save every 100 iterations
        "--log_interval", "10",          # Log every 10 iterations
        "--headless",                    # No GUI for faster training
        "--seed", "42",                  # Reproducible results
    ]
    
    print("🚀 Starting quick training for skateboard balancing...")
    print(f"📁 Project root: {project_root}")
    print(f"🤖 Task: Unitree-G1-29DOF-Skateboard-v0")
    print(f"🌍 Environments: 8192")
    print(f"🔄 Max iterations: 1000")
    print(f"📊 Batch size: 16384")
    print(f"📈 Learning rate: 3e-4")
    print("=" * 50)
    
    try:
        # Run training
        result = subprocess.run(training_args, env=env, check=True)
        print("✅ Training completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error: {e}")
        return 1
    except KeyboardInterrupt:
        print("⏹️ Training interrupted by user")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
