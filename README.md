# Fake News Detection Using Machine Learning

This project demonstrates how machine learning and natural language processing (NLP) techniques can be used to classify news articles as **real** or **fake**. Using a labeled dataset of news content, the model is trained to detect misinformation based on textual features.

---

## Dataset

The dataset includes two CSV files:

* `Fake.csv` — Contains fake news articles
* `True.csv` — Contains real news articles

**Source:** [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

---

## Project Workflow

### Step 1: Load and Explore the Dataset

* Load both datasets using Pandas.
* Add labels (`1` for fake, `0` for real).
* Concatenate and shuffle the combined dataset to ensure randomness.

### Step 2: Data Preprocessing

* Convert text to lowercase.
* Remove URLs, punctuation, and stopwords.
* Apply lemmatization using NLTK's `WordNetLemmatizer`.
* Create a new `content` column by combining cleaned title and body text.

### Step 3: Feature Extraction

* Use **TF-IDF Vectorization** to convert text into numerical features.
* Limit to the top 5000 most significant terms for efficiency.

### Step 4: Train-Test Split

* Split the dataset into 80% training and 20% testing sets using `train_test_split`.

### Step 5: Model Training

Multiple classification models were trained and compared:

* **Passive Aggressive Classifier** (best performance)
* Naive Bayes
* Random Forest Classifier
* Linear Support Vector Classifier (SVM)

### Step 6: Model Evaluation

* Evaluate models using metrics such as accuracy, precision, recall, and F1-score.
* Display confusion matrices for visual comparison.

### Step 7: Save the Best Model

* Use `joblib` to serialize and save the best-performing model and the TF-IDF vectorizer for future use.

### Step 8: Prediction Function

* A utility function is created to load the saved model and vectorizer.
* Accepts raw input text, processes it, and predicts whether the content is real or fake.

---

## Technology Stack

* Python 3
* Pandas
* Scikit-learn
* NLTK (Natural Language Toolkit)
* Matplotlib and Seaborn
* Jupyter Notebook / Kaggle Notebook

---

## Example Prediction

```python
sample_news = """
The president announced a new policy that will change everything.
This is completely false information designed to mislead.
"""

print(predict_news(sample_news))  # Output: Fake News

