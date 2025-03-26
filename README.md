# 🧠 Natural Language Processing Projects – Classification, Sentiment Analysis & Sarcasm Detection

## 📌 Overview

This repository contains two rich, real-world NLP projects designed to demonstrate expertise in building **classification models**, performing **sentiment analysis**, and detecting **sarcasm** using **RNNs**, **LSTM**, and **GloVe embeddings** in **TensorFlow**.

---

## 📚 NLP Project 1: Blog Author Multi-Label Text Classification

### 🧪 Domain: Digital Content Management

Using the **Blog Authorship Corpus** (681k+ posts from 19,320 bloggers), this project classifies a blog post into multiple labels like **age group**, **gender**, and **topic**.

### 📦 Dataset:
- Blog posts text labeled with gender, age group, industry, and astrological sign
- Age categories: 10s (13–17), 20s (23–27), 30s (33–47)

### 🧠 Objectives:
- Predict multiple attributes about the blog author based on blog text
- Perform end-to-end NLP pipeline: cleaning, preprocessing, vectorization, model building, and tuning

### 🛠️ Key Components:
- Language detection and filtering (e.g., `langdetect`)
- Text cleaning: lowercasing, stopword removal, punctuation cleanup
- Multi-label classification using:
  - CountVectorizer / TF-IDF / Word Embeddings
  - Models: Logistic Regression, SVM, Naive Bayes, etc.
- Performance tuning using hyperparameter optimization
- Evaluation using Accuracy, Precision, Recall, and ROC-AUC

---

## 🎬 NLP Project 2

### Part A: IMDB Sentiment Classification

#### 🧪 Domain: Digital Content & Entertainment

Using the IMDB movie review dataset (50,000 reviews), this model classifies sentiment as **positive** or **negative** using **RNN with LSTM layers**.

### 📦 Dataset:
- Pre-tokenized dataset via `keras.datasets.imdb` with word indices

### 🧠 Objectives:
- Build a sequential NLP classifier to predict sentiment
- Use the top 10,000 frequent words, truncate sequences to 20 words
- Train RNN-based models with LSTM architecture
- Visualize model performance and decode predictions

---

### Part B: Sarcasm Detection in News Headlines

#### 🧪 Domain: Social Media & News Analytics

Using a curated dataset of **news headlines** from [TheOnion](https://www.theonion.com/) (sarcastic) and [HuffPost](https://www.huffpost.com/) (non-sarcastic), the goal is to detect **sarcasm** using **GloVe embeddings** and **Bidirectional LSTM**.

### 📦 Dataset:
- Headlines with labels: `is_sarcastic`, `headline`, `article_link`
- High-quality, well-labeled sarcastic vs. non-sarcastic text

### 🧠 Objectives:
- Clean and preprocess textual data
- Tokenize and pad sequences
- Build vocabulary and word index mappings
- Use GloVe word embeddings to initialize weight matrix
- Train a BiLSTM model with appropriate dropout and activation
- Evaluate using validation accuracy and predictions

---

## ⚙️ Tools, Libraries & Skills Used

- **Languages**: Python
- **Libraries**: TensorFlow, Keras, NLTK, Scikit-learn, GloVe, langdetect
- **Concepts & Techniques**:
  - Word Embeddings (GloVe)
  - Recurrent Neural Networks (RNN)
  - LSTM & Bidirectional LSTM
  - Sequence Padding & Tokenization
  - Multi-Label Classification
  - Sentiment Analysis
  - Sarcasm Detection
  - Hyperparameter Tuning
  - Performance Evaluation (ROC-AUC, Precision, Recall)

---

## 📁 Repository Structure

<pre>
.
├── Natural Language Processing Project - 1/
│   ├── code/
│   │   ├── Ishant_Kundra_NLP.ipynb              # Main notebook
│   │   └── Ishant_Kundra_NLP.html               # Exported HTML
│   └── problem statement/
│       └── NLP-1_Problem Statement.pdf

├── Natural Language Processing Project - 2/
│   ├── code/
│   │   ├── NLP-2.ipynb                          # Sentiment + Sarcasm notebook
│   │   └── NLP-2.html                           # Exported HTML
│   └── problem statement/
│       └── NLP-2_Problem Statement.pdf

├── README.md                                     # This file
└── .gitignore
</pre>

---

## 💡 Key Learnings

- Built robust NLP pipelines for multi-label and binary classification
- Compared traditional ML vectorizers with modern deep learning embeddings
- Designed LSTM-based models and interpreted performance
- Understood the complexity of sarcasm detection using contextual text
- Mastered data preprocessing techniques tailored to NLP tasks

---

## ✍️ Author

**Ishant Kundra**  
📧 [ishantkundra9@gmail.com](mailto:ishantkundra9@gmail.com)  
🎓 Master’s in Computer Science | AIML Track
