# Lexicon

LU CDS527 Project Lexicon Team

# CDS527 Big Data Analytics

Group Name:            Lexicon

Director:              Prof. CHIU Hon-wing Billy

Project:        Decision Support System based on Business textual dataset

| Student No. | English Name | Email Address (Lingnan Email) |
|-------------|--------------|-------------------------------|
| 3160708     | LUO Suhai    | suhailuo@ln.hk                |
| 3160291     | -            | fansenwei@ln.hk               |
| 3160320     | -            | junrongli@ln.hk               |
| 3160734     | -            | jiaweiwang2@ln.hk             |
| 3160722     | -            | zihaowang3@ln.hk              |
| 3160069     | GUAN Yuqi    | yuqiguan@ln.hk                |

## Project Overview

This project consists of two main tasks:

- **System Development (30%)**: Develop a decision support system for text classification.
- **Case Study (10%)**: Analyze and propose solutions for a real-world business case assigned by the instructor.

Additionally, each group will deliver an oral presentation (20%).

## Task Details

### Task 1: System Development

Your team will receive a business textual dataset. Tasks include:

- **Baseline Model**: Develop a basic model without hyperparameter tuning (e.g., Logistic Regression).
- **Model Comparison**: Evaluate various data analytics models (Gradient Boosting Tree, Decision Tree, etc.) and word
  embedding models (BERT, GloVe, Word2Vec).
- **Hyperparameter Tuning**: Optimize models and report the best performance.
- **Visualization**: Apply statistical visualization methods (charts, word clouds, correlations, central tendencies,
  dispersion).
- **Model Improvement**: Explore novel techniques to improve the baseline model.

**Submission**: Jupyter Notebook containing code, model details, visualization, and a short descriptive summary (max 300
words).

### Task 2: Case Study

Your group will analyze an assigned company case, addressing:

- Company background and current business/data issues.
- Your big data solution approach (statistical analysis, visual analysis, machine learning, semantic analysis).
- A critical evaluation of your proposed solution, including relevant KPIs (e.g., SMART criteria).

**Submission**: Written report in MS Word (max 3 pages).

## Oral Presentation

- Each student presents (~15 minutes per group).
- Contents must align with the submitted written report and notebook.

## Deliverables

**Due date:** April 25, 2025 (23:59 HK Time) on Moodle

Submit the following files:

- **Jupyter Notebook**: System development code and visualizations.
- **MS Word Report**: Case study analysis and evaluation.
- **PowerPoint Presentation**: Summary highlights from Tasks 1 & 2.

Each student must individually submit a separate Word file detailing their personal contribution and work distribution
within the group.

## Assessment Criteria

- **System Implementation & Evaluation (20%)**: Performance and thorough evaluation of the developed system.
- **Problem Definition (10%)**: Clear understanding, constraints identification, and proposed alternatives.
- **Creative Solution (10%)**: Innovation and contextual relevance of the big data solution.
- **Presentation (20%)**: Time management, clarity, organization, body language, audience engagement, and language
  accuracy.

## Notes

- Late submissions incur a penalty of 1% per day.
- Plagiarism will be strictly penalized.

## 🔖 Project Structure

```
project/
│
├── dataset/
│   ├── traindata7.csv
│   ├── testdata7.csv
│   ├── cleaned_traindata7.csv
│   └── cleaned_testdata7.csv
│
├── embeddings/
│   └── glove.6B.100d.txt (Download manually)
│
├── notebooks/
│   └── sentiment_analysis.ipynb
│
├── output/
│   ├── confusion_matrix.png
│   ├── label_distribution.png
│   └── model_performance_comparison.csv
│
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd project
```

### 2. Set Up Python Environment

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download External Embeddings (Optional)

- Download GloVe embeddings (`glove.6B.100d.txt`) [here](https://nlp.stanford.edu/projects/glove/) and save it in
  the `embeddings` directory.

### 4. Run Jupyter Notebook

```bash
jupyter notebook notebooks/sentiment_analysis.ipynb
```

## 🚧 Data Preprocessing Steps

- **Text Cleaning**: Removing punctuation, URLs, mentions, and digits; converting text to lowercase.
- **Stopwords Removal & Lemmatization**: Using NLTK's stopwords and WordNet Lemmatizer.

## 📚 Feature Extraction Methods

- **TF-IDF**: Traditional text feature extraction.
- **BERT**: Contextual embeddings using pre-trained transformers.
- **GloVE**: Pre-trained word embeddings (100-dimensional vectors).
- **Word2Vec**: Custom word embeddings trained on provided data.

## 🚀 Models Evaluated

- Logistic Regression
- Decision Tree
- Gradient Boosting
- Random Forest
- Linear SVM
- Multinomial Naive Bayes

## 📈 Results

- Results including model accuracy, classification reports, confusion matrix, and label distributions are saved
  under `output/`.

## 🗂️ Files & Output

- `cleaned_traindata7.csv`, `cleaned_testdata7.csv`: Processed dataset.
- `model_performance_comparison.csv`: Performance comparison for different ML models.

## 💡 Recommendations for Improvements

- Explore more advanced deep learning methods (e.g., LSTM, CNN).
- Perform more extensive hyperparameter tuning.
- Evaluate using additional metrics like precision, recall, F1-score for specific class sensitivity.