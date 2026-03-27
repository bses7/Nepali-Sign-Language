import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import pandas as pd

class NSLEvaluator:
    def __init__(self, model, dataloader, tokenizer, device):
        self.model = model
        self.dataloader = dataloader
        self.tokenizer = tokenizer
        self.device = device
        
        plt.rcParams['font.family'] = ['Nirmala UI', 'sans-serif']

    def generate_report(self, output_dir="experiments/evaluation"):
        self.model.eval()
        all_preds, all_labels = [], []
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        print("🧪 Collecting predictions for final evaluation...")
        with torch.no_grad():
            for features, labels in self.dataloader:
                features = features.to(self.device)
                outputs = self.model(features)
                preds = outputs.argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)

        present_label_ids = sorted(list(set(all_labels) | set(all_preds)))
        target_names = [self.tokenizer.idx2char[str(i)] if isinstance(i, int) else self.tokenizer.idx2char[i] for i in present_label_ids]

        print("🎨 Generating Confusion Matrix...")
        cm = confusion_matrix(all_labels, all_preds, labels=present_label_ids)
        plt.figure(figsize=(15, 12))
        sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=target_names, yticklabels=target_names)
        plt.title('Nepali Sign Language - Confusion Matrix')
        plt.tight_layout()
        plt.savefig(out_path / "confusion_matrix.png", dpi=300)
        plt.close()
        
        print("📝 Generating Classification Report...")
        report = classification_report(
            all_labels, 
            all_preds, 
            labels=present_label_ids, 
            target_names=target_names, 
            output_dict=True,
            zero_division=0 
        )
        
        df_report = pd.DataFrame(report).transpose()
        df_report.to_csv(out_path / "classification_report.csv")
        
        print("📊 Generating Per-Class Accuracy Chart...")
        self._plot_class_accuracies(report, target_names, out_path)

        print(f"✅ Evaluation Finished. check {out_path} for report files.")
        return report['accuracy']

    def _plot_class_accuracies(self, report_dict, target_names, out_path):
        classes = []
        f1_scores = []
        
        for name in target_names:
            if name in report_dict:
                classes.append(name)
                f1_scores.append(report_dict[name]['f1-score'])
        
        sorted_indices = np.argsort(f1_scores)
        classes = [classes[i] for i in sorted_indices]
        f1_scores = [f1_scores[i] for i in sorted_indices]

        plt.figure(figsize=(10, 14))
        colors = plt.cm.RdYlGn(np.array(f1_scores)) 
        plt.barh(classes, f1_scores, color=colors)
        plt.axvline(x=np.mean(f1_scores), color='blue', linestyle='--', label=f'Mean F1 ({np.mean(f1_scores):.2f})')
        plt.xlabel('F1-Score')
        plt.title('Performance per Nepali Character')
        plt.xlim(0, 1.0)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path / "class_performance.png", dpi=300)
        plt.close()