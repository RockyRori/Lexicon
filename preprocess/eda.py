"""
data loading
"""

print("_______________________________________________________________________________________________")
print("data loading")

import pandas as pd

# 加载数据
train_df = pd.read_csv('../dataset/traindata7.csv')
test_df = pd.read_csv('../dataset/testdata7.csv')

print("loaded from ../dataset/traindata7.csv")

"""
data cleaning
"""

print("_______________________________________________________________________________________________")
print("data cleaning")

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')


# 定义清洗函数
def clean_text(text):
    text = text.lower()  # 转小写
    text = re.sub(r'@\w+', ' ', text)  # 去除@用户
    text = re.sub(r'http\S+', ' ', text)  # 去除URL链接
    text = re.sub(r'[^a-z\s]', ' ', text)  # 去除标点符号、数字
    text = re.sub(r'\s+', ' ', text).strip()  # 去除多余空格
    return text


# 应用清洗函数
train_df['cleaned_Phrase'] = train_df['Phrase'].apply(clean_text)
test_df['cleaned_Phrase'] = test_df['Phrase'].apply(clean_text)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def tokenize_lemmatize(text):
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)


# 应用去除停用词和词形还原
train_df['Processed'] = train_df['cleaned_Phrase'].apply(tokenize_lemmatize)
test_df['Processed'] = test_df['cleaned_Phrase'].apply(tokenize_lemmatize)

# 查看清洗后的数据
print(train_df[['Phrase', 'Processed']].head())

# 仅保存清洗后的文本和Sentiment列
train_df[['Processed', 'Sentiment']].to_csv('../dataset/cleaned_traindata7.csv', index=False)
test_df[['Processed', 'Sentiment']].to_csv('../dataset/cleaned_testdata7.csv', index=False)

"""
feature engineering TFIDF
"""

print("_______________________________________________________________________________________________")
print("feature engineering TFIDF")

from sklearn.feature_extraction.text import TfidfVectorizer

# 使用清洗后的文本数据进行特征提取
X_train = train_df['Processed']
y_train = train_df['Sentiment']
X_test = test_df['Processed']
y_test = test_df['Sentiment']

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

"""
feature engineering BERT
"""

print("_______________________________________________________________________________________________")
print("feature engineering BERT")

import torch
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def bert_encode(texts, tokenizer, model):
    inputs = tokenizer(texts, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()


# 提取BERT特征
X_train_bert = bert_encode(X_train.tolist(), tokenizer, model)
X_test_bert = bert_encode(X_test.tolist(), tokenizer, model)

"""
feature engineering GloVE
"""

print("_______________________________________________________________________________________________")
print("feature engineering GloVE")

import os
import numpy as np

# 指定GloVe文件路径
glove_path = "../dataset/glove.6B.100d.txt"
embeddings_index = {}

if os.path.exists(glove_path):
    print("loading GloVe model")
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], 'float32')
            embeddings_index[word] = vector
else:
    print(
        f"⚠️ {glove_path} not found, use zero vector instead. Goto and download then unzip glove.6B.zip https://nlp.stanford.edu/projects/glove/")


def glove_encode(texts, embeddings_index, dim=100):
    encoded = []
    for text in texts:
        words = text.split()
        vectors = [embeddings_index.get(word, np.zeros(dim)) for word in words]
        if vectors:
            vectors_mean = np.mean(vectors, axis=0)
        else:
            vectors_mean = np.zeros(dim)
        encoded.append(vectors_mean)
    return np.array(encoded)


# 如果embeddings_index为空，则所有特征都会是零向量
X_train_glove = glove_encode(X_train, embeddings_index)
X_test_glove = glove_encode(X_test, embeddings_index)

"""
feature engineering Word2Vec
"""

print("_______________________________________________________________________________________________")
print("feature engineering Word2Vec")

from gensim.models import Word2Vec
import numpy as np

# 准备数据（tokenize）
train_tokens = [text.split() for text in X_train]

# 训练Word2Vec模型
w2v_model = Word2Vec(sentences=train_tokens, vector_size=100, window=5, min_count=2, workers=4)


def word2vec_encode(texts, model, dim=100):
    encoded = []
    for text in texts:
        vectors = [model.wv[word] for word in text.split() if word in model.wv]
        if vectors:
            vectors_mean = np.mean(vectors, axis=0)
        else:
            vectors_mean = np.zeros(dim)
        encoded.append(vectors_mean)
    return np.array(encoded)


# 提取Word2Vec特征
X_train_w2v = word2vec_encode(X_train, w2v_model)
X_test_w2v = word2vec_encode(X_test, w2v_model)

"""
baseline model regression
"""

print("_______________________________________________________________________________________________")
print("baseline model regression")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 训练基线模型
baseline_model = LogisticRegression(max_iter=200)
baseline_model.fit(X_train_tfidf, y_train)

# 测试基线模型性能
baseline_preds = baseline_model.predict(X_test_tfidf)
baseline_accuracy = accuracy_score(y_test, baseline_preds)
print(f'Baseline Logistic Regression Accuracy: {baseline_accuracy:.4f}')
print(classification_report(y_test, baseline_preds))

"""
different data models
"""

print("_______________________________________________________________________________________________")
print("different data models")

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import os

# 创建output文件夹（若不存在）
output_dir = '../output'
os.makedirs(output_dir, exist_ok=True)

# 定义各个特征集
feature_sets = {
    'TF-IDF': (X_train_tfidf, X_test_tfidf),
    'BERT': (X_train_bert, X_test_bert),
    'GloVE': (X_train_glove, X_test_glove),
    'Word2Vec': (X_train_w2v, X_test_w2v)
}

# 模型列表定义
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=28),
    'Gradient Boosting': GradientBoostingClassifier(random_state=28),
    'Random Forest': RandomForestClassifier(random_state=28),
    'Linear SVM': LinearSVC(random_state=28, max_iter=100),
    'Multinomial Naive Bayes': MultinomialNB()
}

# 存储模型准确率
model_accuracies = {}

# 循环对比每种特征的效果
for feature_name, (X_tr, X_te) in feature_sets.items():
    print(f"\n feature: {feature_name}")

    for model_name, model in models.items():
        # 跳过不兼容的组合 (BERT, GloVE, Word2Vec 不适合 MultinomialNB)
        if model_name == 'Multinomial Naive Bayes' and feature_name in ['BERT', 'GloVE', 'Word2Vec']:
            print(f"skip {model_name} with {feature_name} for feature involve negative numbers")
            continue

        model.fit(X_tr, y_train)
        predictions = model.predict(X_te)
        accuracy = accuracy_score(y_test, predictions)
        model_accuracies[model_name + " with " + feature_name] = accuracy
        print(f"{model_name} with {feature_name}: Accuracy = {accuracy:.4f}")
        print(classification_report(y_test, predictions))

# 比较各个模型的准确率
print("\nModel Performance Comparison:")
for model_name, accuracy in model_accuracies.items():
    print(f"{model_name}: {accuracy:.4f}")

# 将模型性能保存到csv文件中
performance_df = pd.DataFrame(list(model_accuracies.items()), columns=['Model', 'Accuracy'])
performance_df.to_csv(os.path.join(output_dir, 'model_performance_comparison.csv'), index=False)

print(f"\nresults saved to {os.path.join(output_dir, 'model_performance_comparison.csv')}")

"""
hyperparameter_tuning
"""
# TODO: every model above need hyper-parameter tuning

print("_______________________________________________________________________________________________")
print("hyperparameter_tuning")

from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'saga']
}

grid_search = GridSearchCV(LogisticRegression(max_iter=200), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_tfidf, y_train)

print(f'Best parameters: {grid_search.best_params_}')
print(f'Best cross-validation accuracy: {grid_search.best_score_:.4f}')

# 使用最佳参数的模型进行预测
best_model = grid_search.best_estimator_
best_preds = best_model.predict(X_test_tfidf)
best_accuracy = accuracy_score(y_test, best_preds)
print(f'Tuned Logistic Regression Accuracy: {best_accuracy:.4f}')
print(classification_report(y_test, best_preds))

"""
visualization
"""
# TODO: every model above need visualization

print("_______________________________________________________________________________________________")
print("visualization")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# 创建output文件夹（若不存在）
output_dir = '../output'
os.makedirs(output_dir, exist_ok=True)

# 混淆矩阵（最佳模型）
cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Logistic Regression (Tuned)')
plt.xlabel('Predicted Sentiment')
plt.ylabel('Actual Sentiment')
plt.tight_layout()

# 保存混淆矩阵图像
confusion_matrix_path = os.path.join(output_dir, 'confusion_matrix.png')
plt.savefig(confusion_matrix_path)
plt.show()

# 标签分布（训练数据）
plt.figure(figsize=(8, 5))
sns.countplot(x='Sentiment', data=train_df)
plt.title('Sentiment Distribution in Training Data')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.tight_layout()

# 保存标签分布图像
label_distribution_path = os.path.join(output_dir, 'label_distribution.png')
plt.savefig(label_distribution_path)
plt.show()

print(f"Visualization images saved to {output_dir}")
