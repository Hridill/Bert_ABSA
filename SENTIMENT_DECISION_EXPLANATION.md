# How the Model Decides Negative, Neutral, or Positive

## 🎯 Quick Answer

The model outputs **3 probability scores** (one for each class), and whichever score is **highest** determines the final label.

---

## 📊 Step-by-Step Process

### **1. Model Output (Raw Logits)**
The BERT model outputs **3 raw scores** (called "logits"):
```python
# Example output from model:
raw_outputs = [2.1, 0.8, 1.5]  # [negative_score, neutral_score, positive_score]
```

### **2. Convert to Probabilities (Softmax)**
These raw scores are converted to probabilities that sum to 1.0:

```python
# In app.py line 285:
outputs = torch.softmax(outputs, dim=1)

# Example after softmax:
probabilities = [0.45, 0.20, 0.35]  # [negative_prob, neutral_prob, positive_prob]
# These sum to 1.0: 0.45 + 0.20 + 0.35 = 1.0
```

**What Softmax Does:**
- Converts raw scores into probabilities (0.0 to 1.0)
- Ensures all probabilities add up to 1.0
- Higher raw scores → higher probabilities

### **3. Find the Highest Probability (Argmax)**
The class with the **highest probability** wins:

```python
# In app.py line 324:
predicted_sentiment = ["negative", "neutral", "positive"][predictions.argmax()]

# Example:
probabilities = [0.45, 0.20, 0.35]
argmax = 0  # Index 0 has highest value (0.45)
result = ["negative", "neutral", "positive"][0] = "negative"
```

### **4. Label Mapping**
The model uses this fixed mapping (from `train.py` line 33):
```python
sentiment_map = {
    "negative": 0,  # Index 0
    "neutral": 1,   # Index 1
    "positive": 2   # Index 2
}
```

---

## 🔍 Real Example

Let's trace through an example:

**Input Text:** "The doctor was very professional and explained everything clearly."

**Step 1: Text Processing**
- Tokenized by BERT tokenizer
- Padded/truncated to 128 tokens
- Converted to tensor format

**Step 2: Model Forward Pass**
- Text goes through BERT layers
- Gets processed by custom head (attention, FC layers)
- Outputs 3 raw logits: `[0.5, 1.2, 3.8]`

**Step 3: Softmax**
```python
raw = [0.5, 1.2, 3.8]
softmax = [0.05, 0.12, 0.83]  # Probabilities
# Interpretation:
# - 5% chance it's negative
# - 12% chance it's neutral  
# - 83% chance it's positive
```

**Step 4: Argmax Decision**
```python
argmax = 2  # Index 2 has highest probability (0.83)
predicted = ["negative", "neutral", "positive"][2] = "positive"
```

**Result:** ✅ **Positive** (because 0.83 > 0.12 > 0.05)

---

## 🧠 What the Model Learned During Training

### **Training Data**
The model was trained on labeled examples like:
```python
# From train.py line 33:
sentiment_map = {"negative": 0, "neutral": 1, "positive": 2}

# Training examples:
"The doctor was rude" → negative (0)
"The appointment was okay" → neutral (1)
"The treatment was excellent" → positive (2)
```

### **What BERT Learned**
1. **Word Patterns**: 
   - "rude", "terrible", "awful" → negative
   - "okay", "fine", "average" → neutral
   - "excellent", "great", "amazing" → positive

2. **Context Understanding**:
   - "not good" → negative (understands negation)
   - "very good" → positive (understands intensifiers)
   - "could be better" → neutral (understands uncertainty)

3. **Sentence Structure**:
   - Complex sentences with multiple sentiments
   - Aspect-specific language (doctor, treatment, staff, etc.)

---

## ⚠️ Why "Neutral" Sometimes Wins

**Example:** "DOCTOR IS GOOD"

**Why it might be neutral:**
```python
probabilities = [0.30, 0.42, 0.28]
# neutral (0.42) > negative (0.30) > positive (0.28)
```

**Reasons:**
1. **Weak Signal**: "good" is positive but not strong enough
2. **Short Text**: Limited context for the model
3. **Training Data Bias**: Model saw many neutral examples with mild language
4. **Close Scores**: All three probabilities are close (model is uncertain)

**The model picks the highest, even if it's only slightly higher!**

---

## 🎛️ Decision Threshold (Current Implementation)

**Current System: Simple Argmax**
```python
# Always picks the highest probability, no matter how close
if probabilities = [0.34, 0.33, 0.33]:
    result = "negative"  # Even though it's very close!
```

**Alternative (Not Currently Used):**
You could add a confidence threshold:
```python
max_prob = probabilities.max()
if max_prob < 0.5:  # If no class is confident
    result = "neutral"  # Default to neutral
else:
    result = argmax(probabilities)
```

---

## 📈 Model Architecture Output

**Final Layer** (`model.py` line 32):
```python
self.out = nn.Linear(256, 3)  # Outputs 3 values
```

**What This Means:**
- Input: 256-dimensional feature vector (from BERT + custom layers)
- Output: 3 raw scores (one per class)
- These scores are then softmaxed to get probabilities

---

## 🔬 How Training Shapes the Decision

### **Loss Function** (`engine.py` line 46):
```python
loss = nn.CrossEntropyLoss()(outputs, sentiment)
```

**What This Does:**
- Compares model's probability distribution to true label
- Penalizes wrong predictions more heavily
- Teaches model to maximize correct class probability

### **Training Process:**
1. Model sees: "The doctor was rude" → should output high probability for negative
2. If model outputs high probability for positive instead → large penalty
3. Model adjusts weights to fix this
4. After many examples, model learns patterns

---

## 💡 Key Takeaways

1. **Decision Rule**: Highest probability wins (argmax)
2. **No Threshold**: Even 0.34 vs 0.33 → first one wins
3. **Probabilities Sum to 1.0**: All three probabilities always add up to 1.0
4. **Based on Training**: Model learned from labeled healthcare sentiment data
5. **Context Matters**: Same word can mean different things in different contexts

---

## 🔍 Example Scenarios

### **Scenario 1: Clear Positive**
```
Input: "The treatment was extremely effective and I feel amazing!"
Probabilities: [0.05, 0.10, 0.85]
Decision: positive (0.85 is clearly highest)
```

### **Scenario 2: Ambiguous (Close Scores)**
```
Input: "The doctor was okay, nothing special."
Probabilities: [0.25, 0.40, 0.35]
Decision: neutral (0.40 wins, but all are close)
```

### **Scenario 3: Weak Positive**
```
Input: "DOCTOR IS GOOD"
Probabilities: [0.30, 0.42, 0.28]
Decision: neutral (model is uncertain, neutral slightly wins)
```

### **Scenario 4: Strong Negative**
```
Input: "The staff was completely unprofessional and rude."
Probabilities: [0.92, 0.05, 0.03]
Decision: negative (very confident)
```

---

## 🛠️ Code Location Reference

| Step | File | Line | Description |
|------|------|------|-------------|
| Model Output | `model.py` | 32 | `nn.Linear(256, 3)` - outputs 3 logits |
| Softmax | `app.py` | 285 | `torch.softmax(outputs, dim=1)` |
| Argmax Decision | `app.py` | 324 | `predictions.argmax()` |
| Label Mapping | `train.py` | 33 | `{"negative": 0, "neutral": 1, "positive": 2}` |

---

## 🎯 Summary

**The model decides based on:**
1. ✅ **Highest probability** (argmax of softmax probabilities)
2. ✅ **Training data patterns** (learned from labeled examples)
3. ✅ **BERT's understanding** (context, word relationships, sentiment patterns)
4. ✅ **No confidence threshold** (always picks highest, even if close)

**The decision is purely mathematical:** whichever of the 3 probabilities is highest wins, regardless of how close the scores are!


