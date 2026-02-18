import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

class RecLogger:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.output_dir / "rec_history.csv"

    def plot_progress(self, history_list):
        df = pd.DataFrame(history_list)
        df.to_csv(self.history_path, index=False)

        plt.figure(figsize=(12, 5))

        # Loss Plot
        plt.subplot(1, 2, 1)
        plt.plot(df['epoch'], df['loss'], label='Train Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.legend()

        # Accuracy Plot
        plt.subplot(1, 2, 2)
        plt.plot(df['epoch'], df['train_acc'], label='Train Acc')
        plt.plot(df['epoch'], df['val_acc'], label='Val Acc')
        plt.title('Accuracy Over Epochs')
        plt.xlabel('Epoch')
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / "training_curves.png")
        plt.close()