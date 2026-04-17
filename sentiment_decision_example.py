"""
Simple demonstration of how the model decides between negative, neutral, and positive.
Run this to see the decision process step-by-step.
"""

import numpy as np
import torch
import torch.nn.functional as F

print("=" * 70)
print("HOW THE MODEL DECIDES: NEGATIVE, NEUTRAL, OR POSITIVE")
print("=" * 70)

# Simulate what happens inside the model
def demonstrate_decision(text_description, raw_logits):
    """
    Shows the complete decision process from raw model output to final label.
    """
    print(f"\n📝 Example: {text_description}")
    print("-" * 70)
    
    # Step 1: Raw model output (logits)
    print(f"\n1️⃣  RAW MODEL OUTPUT (logits):")
    print(f"   Negative:  {raw_logits[0]:.2f}")
    print(f"   Neutral:   {raw_logits[1]:.2f}")
    print(f"   Positive:  {raw_logits[2]:.2f}")
    
    # Step 2: Convert to probabilities using softmax
    logits_tensor = torch.tensor([raw_logits], dtype=torch.float32)
    probabilities = F.softmax(logits_tensor, dim=1).numpy()[0]
    
    print(f"\n2️⃣  PROBABILITIES (after softmax):")
    print(f"   Negative:  {probabilities[0]:.4f} ({probabilities[0]*100:.1f}%)")
    print(f"   Neutral:   {probabilities[1]:.4f} ({probabilities[1]*100:.1f}%)")
    print(f"   Positive:  {probabilities[2]:.4f} ({probabilities[2]*100:.1f}%)")
    print(f"   ✓ Sum:     {probabilities.sum():.4f} (must equal 1.0)")
    
    # Step 3: Find the highest probability (argmax)
    predicted_index = np.argmax(probabilities)
    labels = ["negative", "neutral", "positive"]
    predicted_label = labels[predicted_index]
    
    print(f"\n3️⃣  DECISION (argmax - highest probability wins):")
    print(f"   Index {predicted_index} has highest probability: {probabilities[predicted_index]:.4f}")
    print(f"   → Predicted Sentiment: {predicted_label.upper()}")
    
    # Visual bar
    max_bar_length = 50
    bars = []
    for i, prob in enumerate(probabilities):
        bar_length = int(prob * max_bar_length)
        bar = "█" * bar_length
        bars.append(f"   {labels[i]:8s}: {bar} {prob*100:5.1f}%")
    
    print(f"\n4️⃣  VISUAL COMPARISON:")
    for bar in bars:
        print(bar)
    
    return predicted_label, probabilities

# Example scenarios
examples = [
    ("Strong Positive", [0.5, 1.0, 4.5]),
    ("Weak Positive (might be neutral)", [1.2, 2.1, 2.3]),
    ("Clear Negative", [3.8, 0.8, 0.5]),
    ("Ambiguous (close scores)", [1.5, 1.6, 1.4]),
    ("Strong Neutral", [0.8, 3.2, 1.1]),
]

print("\n" + "=" * 70)
print("EXAMPLE SCENARIOS")
print("=" * 70)

results = []
for desc, logits in examples:
    label, probs = demonstrate_decision(desc, logits)
    results.append((desc, label, probs))
    print()

# Summary
print("\n" + "=" * 70)
print("KEY INSIGHTS")
print("=" * 70)
print("""
1. The model outputs 3 raw scores (logits) for each class
2. Softmax converts these to probabilities (sum = 1.0)
3. Argmax picks the class with highest probability
4. Even if scores are close (e.g., 0.34 vs 0.33), the higher one wins
5. The model learned these patterns from training data

⚠️  IMPORTANT: The model uses a simple "highest wins" rule.
   There's NO confidence threshold - even 0.34 beats 0.33!
""")

print("\n" + "=" * 70)
print("WHY 'DOCTOR IS GOOD' MIGHT BE NEUTRAL")
print("=" * 70)
print("""
Example probabilities: [0.30, 0.42, 0.28]

Reasons:
1. Short text → limited context for model
2. "good" is positive but not strong enough
3. Model saw many neutral examples with mild language
4. All three probabilities are close (model is uncertain)

The model picks neutral (0.42) because it's highest,
even though it's only slightly higher than negative (0.30)!
""")

print("\n" + "=" * 70)
print("CODE LOCATION IN YOUR PROJECT")
print("=" * 70)
print("""
📍 Model Output:    src/model.py line 32 (nn.Linear(256, 3))
📍 Softmax:         src/app.py line 285 (torch.softmax)
📍 Decision:        src/app.py line 324 (predictions.argmax())
📍 Label Mapping:   src/train.py line 33 (negative=0, neutral=1, positive=2)
""")


