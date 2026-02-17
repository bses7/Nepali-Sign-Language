import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

class NSLLogger:
    def __init__(self, log_dir="experiments/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history = []

    def log_epoch(self, epoch, train_pos, train_vel, train_bone, val_loss, lr):
        data = {
            'epoch': epoch,
            'train_pos': train_pos,
            'train_vel': train_vel,
            'train_bone': train_bone,
            'val_loss': val_loss,
            'lr': lr
        }
        self.history.append(data)
        pd.DataFrame(self.history).to_csv(self.log_dir / "training_history.csv", index=False)
        self._plot()

    def _plot(self):
        df = pd.DataFrame(self.history)
        # 3 Subplots: Loss, Velocity, and Bone Integrity
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

        ax1.plot(df['epoch'], df['train_pos'], label='Train Pos')
        ax1.plot(df['epoch'], df['val_loss'], label='Val Loss')
        ax1.set_yscale('log')
        ax1.set_title('Learning Curves')
        ax1.legend(); ax1.grid(True)

        ax2.plot(df['epoch'], df['train_vel'], label='Motion Jitter (Vel)', color='orange')
        ax2.set_title('Smoothness Loss')
        ax2.legend(); ax2.grid(True)

        ax3.plot(df['epoch'], df['train_bone'], label='Bone Stretch', color='green')
        ax3.set_title('Anatomic Integrity (Should stay low)')
        ax3.legend(); ax3.grid(True)

        plt.tight_layout()
        plt.savefig(self.log_dir / "detailed_training_metrics.png")
        plt.close()