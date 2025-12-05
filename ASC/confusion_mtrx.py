import pandas as pd
from sklearn.metrics import confusion_matrix,precision_score, recall_score, f1_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

#class labels
class_labels = ["beach", "bus", "cafe/restaurant", "car", "city_center", "forest_path", "grocery_store", "home", "library", "metro_station", "office", "park", "residential_area", "train", "tram"]


df = pd.read_csv('/home/jacobala@alabsad.fau.de/AOT/1baseevaluationsnr15.csv')


true_labels = df['original_class'].values
predicted_labels = df['predicted_class_name'].values


precision = precision_score(true_labels, predicted_labels, average='weighted')
print(f"Precision: {precision:.4f}")

recall = recall_score(true_labels, predicted_labels, average='weighted')
print(f"Recall: {recall:.4f}")


f1 = f1_score(true_labels, predicted_labels, average='weighted')
print(f"F1-Score: {f1:.4f}")

accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Accuracy: {accuracy:.4f}")

report = classification_report(true_labels, predicted_labels)


print(report)


conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=class_labels)

plt.figure(figsize=(21, 15))
ax = sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_labels,
    yticklabels=class_labels,
    annot_kws={"size": 16}   # Annotation numbers bigger
)

# Enlarge axis labels
plt.title("Confusion Matrix", fontsize=24)
plt.xlabel("Predicted Class", fontsize=20)
plt.ylabel("True Class", fontsize=20)

# Enlarge the tick label font sizes
plt.xticks(fontsize=20,  ha='right')
plt.yticks(fontsize=20)

plt.tight_layout()
plt.savefig('/home/jacobala@alabsad.fau.de/AOT/confusion_matrix_snr15.png')
plt.show()
