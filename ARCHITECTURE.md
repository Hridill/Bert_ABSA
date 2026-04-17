# Frontend & Backend Architecture Details

## 📋 Table of Contents
1. [Backend Architecture](#backend-architecture)
2. [Frontend Architecture](#frontend-architecture)
3. [Data Flow](#data-flow)
4. [Key Features](#key-features)

---

## 🔧 BACKEND ARCHITECTURE

### **Technology Stack**
- **Framework**: Flask (Python web framework)
- **ML Framework**: PyTorch
- **NLP Models**: 
  - BERT (bert-base-uncased) for sentiment analysis
  - BART-Large-MNLI for zero-shot aspect classification
  - DistilBERT for trained aspect classifier
- **Libraries**: Transformers, NumPy, Pandas, scikit-learn

### **Main Backend File: `src/app.py`**

#### **1. Application Setup**
```python
- Flask app initialization
- Rate limiting: 200 requests/day, 50/hour per IP
- Global model variables (MODEL, DEVICE, ZERO_SHOT, ASPECT_MODEL)
```

#### **2. Core Components**

##### **A. Model Loading (Lines 341-361)**
- Loads pre-trained BERT sentiment model from `model.bin`
- Initializes zero-shot classifier (BART-Large-MNLI) for aspect detection
- Optionally loads trained aspect classifier if available
- Moves models to GPU if available, otherwise CPU

##### **B. Aspect Detection System (Lines 73-123)**
Three-tier fallback system:
1. **Trained Classifier** (if available): Uses DistilBERT-based aspect classifier
2. **Zero-Shot Classifier**: Uses BART-Large-MNLI for aspect classification
3. **Keyword Heuristic**: Falls back to keyword matching if models fail

**Aspects Supported:**
- `doctor` - Doctor's Communication
- `wait` - Wait Times
- `facility` - Facility & Cleanliness
- `treatment` - Treatment Effectiveness
- `staff` - Staff Behavior
- `overall` - Overall Experience

##### **C. Text Processing Pipeline (Lines 141-286)**
1. **Input Sanitization** (`sanitize_input`):
   - Removes HTML tags
   - Filters special characters
   - Limits to 500 characters
   
2. **Aspect Extraction** (`extract_aspect_relevant_text`):
   - Splits text into segments
   - Filters segments containing aspect keywords
   - Returns relevant portions only

3. **Text Enhancement** (`enhance_text_with_aspect`):
   - Adds aspect-specific context
   - Injects relevant keywords if missing

4. **Prediction** (`sentence_prediction`):
   - Tokenizes text using BERT tokenizer
   - Pads/truncates to MAX_LEN (128 tokens)
   - Runs through BERT model
   - Returns probability distribution over 3 classes: [negative, neutral, positive]

##### **D. Example Override System (Lines 152-247)**
- Predefined example sentences mapped to expected sentiments
- Ensures UI example buttons always return correct labels
- Normalizes text for matching (lowercase, sanitized, space-collapsed)

##### **E. Caching (Line 249)**
- `@lru_cache(maxsize=1000)`: Caches predictions for frequently analyzed texts
- Reduces computation for repeated queries

#### **3. API Endpoints**

##### **GET `/` (Line 288)**
- Serves the main HTML template (`index.html`)
- Entry point for the web interface

##### **GET `/predict` (Line 292)**
**Rate Limited**: 10 requests per minute

**Query Parameters:**
- `sentence` (required): Text to analyze
- `aspect` (optional): Healthcare aspect (default: "overall")
  - Options: `auto`, `overall`, `doctor`, `wait`, `facility`, `treatment`, `staff`

**Response Format:**
```json
{
  "response": {
    "predicted_sentiment": "positive|negative|neutral",
    "sentence": "original text",
    "aspect": "detected aspect",
    "time_taken": "0.123"
  }
}
```

**Processing Flow:**
1. Validate and sanitize input
2. Check if text matches predefined example (override)
3. Auto-detect aspect if `aspect='auto'`
4. Run prediction through BERT model
5. Return JSON response with sentiment and metadata

#### **4. Error Handling**
- Input validation (empty text, invalid characters)
- Model loading errors (graceful fallbacks)
- Exception logging to file and console
- Returns HTTP 400/500 with error messages

---

## 🎨 FRONTEND ARCHITECTURE

### **Technology Stack**
- **HTML5**: Structure
- **CSS3**: Custom styling with CSS variables for theming
- **JavaScript (Vanilla)**: No frameworks, pure JS
- **Bootstrap 5.1.3**: UI components and grid system
- **Font Awesome 6.0**: Icons

### **Main Frontend File: `src/templates/index.html`**

#### **1. Structure & Layout**

##### **A. Header Section (Lines 147-149)**
- Title: "Sentiment Analysis"
- Dark mode toggle button (top-right)

##### **B. Analysis Mode Selection (Lines 151-159)**
- Radio button group: Single Analysis vs Compare Feedback
- Toggles between two different UI layouts

##### **C. Single Analysis Mode (Lines 161-194)**
**Components:**
- Text input area (textarea, max 500 chars)
- Character counter (real-time)
- Healthcare aspect dropdown (7 options)
- Example buttons (6 predefined examples)
- Analyze & Clear buttons

##### **D. Compare Feedback Mode (Lines 196-301)**
**Components:**
- Two-column layout (side-by-side)
- First Feedback textarea + context field
- Second Feedback textarea + context field
- Shared aspect dropdown
- Comparison example buttons (10 examples: 5 improvements + 5 deteriorations)
- Analyze & Clear buttons

##### **E. Results Section (Lines 315-363)**
- Single Analysis Results:
  - Healthcare aspect display
  - Predicted sentiment (color-coded alert)
  - Processing time
  
- Compare Results:
  - Side-by-side sentiment comparison
  - Context labels for each feedback
  - Comparison insight message
  - Processing time

#### **2. Styling & Theming**

##### **CSS Variables (Lines 10-24)**
```css
Light Theme:
- --bg-color: #f8f9fa
- --text-color: #212529
- --card-bg: #ffffff
- --header-bg: #007bff

Dark Theme:
- --bg-color: #212529
- --text-color: #f8f9fa
- --card-bg: #343a40
- --header-bg: #0d6efd
```

##### **Sentiment Color Coding (Lines 121-137)**
- **Positive**: Green (#d4edda background, #155724 text)
- **Negative**: Red (#f8d7da background, #721c24 text)
- **Neutral**: Gray (#e2e3e5 background, #383d41 text)

##### **Responsive Design**
- Bootstrap grid system
- Mobile-friendly layout
- Max-width container (800px)

#### **3. JavaScript Functionality**

##### **A. Character Counting (Lines 369-401)**
- `updateCharCount()`: Single analysis mode
- `updateCharCount1()` / `updateCharCount2()`: Compare mode
- Auto-truncates at 500 characters
- Real-time counter update

##### **B. Example Population (Lines 403-416)**
- `useExample(text, aspect)`: Fills single analysis form
- `useComparisonExample(text1, text2, context1, context2)`: Fills compare form
- Auto-updates character counters

##### **C. Form Management (Lines 418-448)**
- `clearForm()`: Resets form based on active mode
- `toggleTheme()`: Switches between light/dark mode
- Mode switching: Shows/hides appropriate sections

##### **D. API Communication (Lines 467-603)**

**Single Analysis Flow:**
```javascript
1. Validate input (non-empty)
2. Show loading spinner
3. Fetch GET /predict?sentence=...&aspect=...
4. Parse JSON response
5. Update UI with results
6. Hide loading spinner
```

**Compare Analysis Flow:**
```javascript
1. Validate both inputs
2. Show loading spinner
3. Parallel fetch both predictions (Promise.all)
4. Parse both responses
5. Display side-by-side results
6. Generate comparison insight
7. Hide loading spinner
```

**Error Handling:**
- Network errors: Alert user
- Invalid responses: Console log + alert
- Empty inputs: Validation alerts

##### **E. Dynamic UI Updates**
- Aspect name mapping (display names)
- Sentiment color coding (data attributes)
- Auto-detected aspect display
- Comparison insights generation

#### **4. User Experience Features**

##### **A. Loading States**
- Spinner during API calls
- Results hidden until ready
- Smooth transitions

##### **B. Input Validation**
- Character limit enforcement
- Empty input checks
- Real-time feedback

##### **C. Example Buttons**
- Quick-fill functionality
- Pre-configured aspects
- One-click testing

##### **D. Dark Mode**
- Persistent theme toggle
- Smooth color transitions
- Icon changes (moon/sun)

---

## 🔄 DATA FLOW

### **Single Analysis Flow:**
```
User Input → Frontend Validation → GET /predict → Backend Processing
    ↓
Sanitization → Aspect Detection → Text Enhancement → BERT Model
    ↓
Prediction → Override Check → JSON Response → Frontend Display
```

### **Compare Analysis Flow:**
```
Two Inputs → Parallel API Calls → Two Predictions → Side-by-Side Display
    ↓
Comparison Logic → Insight Generation → Results Display
```

### **Aspect Detection Flow:**
```
Text Input → Trained Classifier (if available)
    ↓ (if fails)
Zero-Shot Classifier (BART-MNLI)
    ↓ (if fails)
Keyword Heuristic
    ↓
Return Detected Aspect
```

---

## ✨ KEY FEATURES

### **Backend Features:**
1. ✅ Multi-model sentiment analysis (BERT-based)
2. ✅ Intelligent aspect detection (3-tier fallback)
3. ✅ Text preprocessing and enhancement
4. ✅ Example override system (guaranteed correct labels)
5. ✅ Response caching (LRU cache)
6. ✅ Rate limiting (prevent abuse)
7. ✅ Comprehensive logging
8. ✅ Error handling and validation
9. ✅ GPU/CPU auto-detection
10. ✅ Mixed precision support

### **Frontend Features:**
1. ✅ Dual analysis modes (Single & Compare)
2. ✅ Real-time character counting
3. ✅ Dark/Light theme toggle
4. ✅ Responsive design
5. ✅ Color-coded sentiment display
6. ✅ Predefined examples (16 total)
7. ✅ Loading states
8. ✅ Input validation
9. ✅ Comparison insights
10. ✅ Auto aspect detection display

### **Security Features:**
- Input sanitization (XSS prevention)
- Rate limiting (DoS prevention)
- Character limits (buffer overflow prevention)
- HTML tag removal
- Special character filtering

---

## 📁 File Structure

```
src/
├── app.py              # Main Flask backend (362 lines)
├── model.py            # BERT model architecture (95 lines)
├── config.py           # Configuration settings
├── engine.py           # Training/evaluation functions
├── dataset.py          # Data loading and augmentation
├── train.py            # Training script
├── logger.py           # Logging setup
├── visualization.py    # Plot generation
├── model_versioning.py # Model version management
├── aspect_model.py     # Aspect classifier model
├── aspect_utils.py     # Aspect utilities
└── templates/
    └── index.html      # Frontend UI (606 lines)
```

---

## 🚀 Performance Optimizations

### **Backend:**
- Model caching (loaded once at startup)
- Prediction caching (LRU, 1000 entries)
- GPU acceleration (if available)
- Mixed precision inference
- Efficient tokenization

### **Frontend:**
- Parallel API calls (Compare mode)
- Minimal DOM manipulation
- CSS transitions (hardware accelerated)
- Efficient event listeners

---

## 🔐 API Security

1. **Rate Limiting**: 10 requests/minute per IP
2. **Input Sanitization**: Removes HTML, special chars
3. **Length Limits**: Max 500 characters
4. **Error Handling**: No sensitive data in errors
5. **Logging**: All requests logged (no PII)

---

## 📊 Model Details

### **Sentiment Model:**
- **Architecture**: BERT-base-uncased + custom head
- **Input**: 128 tokens max
- **Output**: 3 classes (negative, neutral, positive)
- **Features**: 
  - Multi-head attention
  - Layer normalization
  - Dropout (0.3, 0.2)
  - GELU activation
  - Last 4 hidden states concatenated

### **Aspect Classifier:**
- **Primary**: DistilBERT-based (if trained)
- **Fallback**: BART-Large-MNLI (zero-shot)
- **Heuristic**: Keyword matching

---

## 🎯 Future Enhancements

**Potential Improvements:**
- WebSocket support for real-time updates
- Batch prediction endpoint
- Model confidence scores in UI
- Export results functionality
- User authentication
- Historical analysis tracking
- Advanced visualizations
- Multi-language support

---

**Last Updated**: Based on current codebase analysis
**Version**: 1.0


