# Bert_ABSA

# BERT Sentiment Analysis Project

This project implements a sentiment analysis system using BERT (Bidirectional Encoder Representations from Transformers) for text classification. The system can classify text as positive or negative sentiment and provides a web interface for real-time predictions.

## Features

- BERT-based sentiment analysis model
- Real-time prediction through REST API
- Web interface for easy interaction
- Model training pipeline with validation
- Performance metrics and evaluation

## Project Structure

```
├── src/
│   ├── app.py              # Flask web application
│   ├── config.py           # Configuration settings
│   ├── dataset.py          # Dataset handling
│   ├── engine.py           # Training and evaluation engine
│   ├── model.py            # BERT model implementation
│   ├── train.py            # Training script
│   └── templates/          # Web templates
├── tests/                  # Unit tests
├── logs/                   # Log files
└── Input/                  # Input data directory
```

## Setup and Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the pre-trained BERT model:
```bash
python src/download_model.py
```

## Usage

### Training the Model

```bash
python src/train.py
```

### Running the Web Interface

```bash
python src/app.py
```

The web interface will be available at `http://localhost:9999`

### API Usage

Send a GET request to `/predict` with a `sentence` parameter:

```bash
curl "http://localhost:9999/predict?sentence=This%20is%20a%20great%20product"
```

Response format:
```json
{
    "response": {
        "positive": "0.95",
        "negative": "0.05",
        "sentence": "This is a great product",
        "time_taken": "0.123"
    }
}
```

## Model Performance

The model achieves the following metrics on the validation set:
- Accuracy: [Your accuracy score]
- F1 Score: [Your F1 score]
- Precision: [Your precision score]
- Recall: [Your recall score]

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

