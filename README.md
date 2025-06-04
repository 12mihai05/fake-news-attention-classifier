# Fake News Detection Using Attention-Based Deep Learning

## Overview

This project implements a deep learning-based fake news detection system using an attention mechanism. The web application allows users to input news articles and receive real-time predictions on their authenticity, along with confidence scores.

## Features

- Real-time fake news detection
- Web-based user interface
- Text preprocessing and cleaning
- Confidence scoring
- Binary classification (Fake/Real)
- Support for long-form articles

## Technical Architecture

### Frontend

- Flask web framework
- HTML/CSS for user interface
- Responsive design
- Loading animation for user feedback

### Backend

- Python 3.x
- TensorFlow 2.x
- NLTK for text processing
- Custom attention mechanism

## Installation

1. Clone the repository

```bash
git clone https://github.com/yourusername/fake-news-attention-classifier.git
cd fake-news-attention-classifier
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Download required files

- Download GloVe embeddings (glove.6B.300d.txt)
- Place in the model directory

## Usage

1. Start the Flask server:

```bash
python app.py
```

2. Open browser and navigate to `http://localhost:5000`
3. Enter news text in the input field
4. Click "Classify" to get predictions

## Model Details

### Architecture

1. **Input Layer**

   - Maximum sequence length: 500 tokens
   - Vocabulary size: 15000
2. **Embedding Layer**

   - GloVe embeddings (300 dimensions)
   - Trainable embeddings
3. **Bidirectional LSTM Layers**

   - First layer: 64 units
   - Second layer: 32 units
   - Dropout: 0.3
   - Recurrent dropout: 0.3
4. **Attention Layer**

   - Custom implementation
   - Learns importance weights for words
5. **Dense Layers**

   - Dense(64, activation='relu') with L2 regularization
   - Layer Normalization
   - Dropout(0.5)
   - Dense(32, activation='relu')
   - Dense(1, activation='sigmoid')

### Training

- Optimizer: Adam (learning rate: 1e-4)
- Loss: Binary crossentropy
- Batch size: 128
- Early stopping with patience=5
- Class weight balancing

## Dataset

- FakeNewsNet dataset
- Kaggle Fake/Real news dataset
- Augmented real news using back-translation

## Project Structure

```
fake-news-attention-classifier/
├── app.py                    # Flask application
├── model/
│   ├── NN_model.py          # Model architecture and training
│   ├── test.py              # Model evaluation
│   ├── back-translation.py  # Data augmentation
│   └── glove.6B.300d.txt    # Word embeddings
├── static/                   # CSS files
├── templates/                # HTML templates
└── README.md                # Documentation
```

## Dependencies

- TensorFlow
- Flask
- NLTK
- NumPy
- Pandas
- Scikit-learn
- deep-translator (for data augmentation)
