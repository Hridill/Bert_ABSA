"""
Simple demonstration of how F1, Precision, Recall, and Accuracy are calculated.
This shows the formulas and provides a clear example.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Example: 3-class classification (Negative=0, Neutral=1, Positive=2)
# Let's say we have 20 samples

# True labels (what the data actually is)
true_labels = np.array([
    0, 0, 0, 0, 0,        # 5 negatives
    1, 1, 1, 1, 1, 1, 1,  # 7 neutrals
    2, 2, 2, 2, 2, 2, 2, 2  # 8 positives
])

# Predicted labels (what the model predicted)
predicted_labels = np.array([
    0, 0, 0, 1, 0,        # 4 correct negatives, 1 wrong (predicted neutral)
    1, 1, 0, 1, 1, 1, 2,  # 5 correct neutrals, 1 wrong (predicted negative), 1 wrong (predicted positive)
    2, 2, 1, 2, 2, 2, 2, 2  # 7 correct positives, 1 wrong (predicted neutral)
])

print("=" * 60)
print("METRICS CALCULATION EXPLANATION")
print("=" * 60)

# 1. Confusion Matrix
print("\n1. CONFUSION MATRIX")
print("-" * 60)
cm = confusion_matrix(true_labels, predicted_labels, labels=[0, 1, 2])
print("Confusion Matrix (rows=actual, cols=predicted):")
print("              Predicted")
print("            Neg  Neu  Pos")
print(f"Actual Neg  {cm[0,0]:3d}  {cm[0,1]:3d}  {cm[0,2]:3d}")
print(f"       Neu  {cm[1,0]:3d}  {cm[1,1]:3d}  {cm[1,2]:3d}")
print(f"       Pos  {cm[2,0]:3d}  {cm[2,1]:3d}  {cm[2,2]:3d}")

# 2. Accuracy
print("\n2. ACCURACY")
print("-" * 60)
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Accuracy = Correct Predictions / Total Predictions")
print(f"         = {np.sum(true_labels == predicted_labels)} / {len(true_labels)}")
print(f"         = {accuracy:.4f} ({accuracy*100:.2f}%)")

# 3. Per-Class Metrics
print("\n3. PER-CLASS METRICS")
print("-" * 60)

for class_idx, class_name in enumerate(['Negative', 'Neutral', 'Positive']):
    print(f"\n{class_name} (Class {class_idx}):")
    
    # True Positives: correctly predicted as this class
    tp = cm[class_idx, class_idx]
    
    # False Positives: predicted as this class but actually something else
    fp = np.sum(cm[:, class_idx]) - tp
    
    # False Negatives: actually this class but predicted as something else
    fn = np.sum(cm[class_idx, :]) - tp
    
    # True Negatives: correctly predicted as NOT this class
    tn = np.sum(cm) - tp - fp - fn
    
    print(f"  TP (True Positives)  = {tp:2d} (correctly predicted as {class_name})")
    print(f"  FP (False Positives) = {fp:2d} (predicted {class_name} but was actually something else)")
    print(f"  FN (False Negatives) = {fn:2d} (actually {class_name} but predicted something else)")
    print(f"  TN (True Negatives)  = {tn:2d} (correctly predicted as NOT {class_name})")
    
    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    print(f"\n  Precision = TP / (TP + FP)")
    print(f"            = {tp} / ({tp} + {fp})")
    print(f"            = {precision:.4f}")
    print(f"  → Of all samples predicted as {class_name}, {precision*100:.1f}% were actually {class_name}")
    
    # Recall
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"\n  Recall = TP / (TP + FN)")
    print(f"         = {tp} / ({tp} + {fn})")
    print(f"         = {recall:.4f}")
    print(f"  → Of all actual {class_name} samples, {recall*100:.1f}% were correctly identified")
    
    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f"\n  F1 Score = 2 × (Precision × Recall) / (Precision + Recall)")
    print(f"           = 2 × ({precision:.4f} × {recall:.4f}) / ({precision:.4f} + {recall:.4f})")
    print(f"           = {f1:.4f}")
    print(f"  → Harmonic mean of precision and recall")

# 4. Overall Metrics (Weighted Average)
print("\n4. OVERALL METRICS (Weighted Average)")
print("-" * 60)
print("Weighted average: each class's metric is weighted by the number of true instances")

weighted_precision = precision_score(true_labels, predicted_labels, average='weighted')
weighted_recall = recall_score(true_labels, predicted_labels, average='weighted')
weighted_f1 = f1_score(true_labels, predicted_labels, average='weighted')

print(f"\nWeighted Precision = {weighted_precision:.4f}")
print(f"Weighted Recall     = {weighted_recall:.4f}")
print(f"Weighted F1 Score   = {weighted_f1:.4f}")

# Show the calculation
class_counts = np.bincount(true_labels)
per_class_precision = precision_score(true_labels, predicted_labels, average=None)
per_class_recall = recall_score(true_labels, predicted_labels, average=None)

print(f"\nCalculation:")
print(f"  Weighted Precision = Σ(Precision_i × Count_i) / Total")
for i, (name, prec, count) in enumerate(zip(['Negative', 'Neutral', 'Positive'], 
                                              per_class_precision, class_counts)):
    print(f"    {name}: {prec:.4f} × {count} = {prec * count:.4f}")

print("\n" + "=" * 60)
print("KEY TAKEAWAYS:")
print("=" * 60)
print("• Accuracy: Overall correctness")
print("• Precision: When model says 'positive', how often is it right?")
print("• Recall: Of all actual positives, how many did we catch?")
print("• F1 Score: Balances precision and recall (harmonic mean)")
print("• Weighted Average: Accounts for class imbalance")
print("=" * 60)


