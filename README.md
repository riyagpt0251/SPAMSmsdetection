# 📩 SMS Spam Classifier

![Spam Detection](https://img.shields.io/badge/Spam-Detection-green?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Sklearn-orange?style=for-the-badge)

## 🚀 Overview
This project is an **SMS Spam Classifier** built using **Python** and **Machine Learning** techniques. It processes text messages, cleans the data, and applies a **Naïve Bayes classifier** to classify messages as **Spam** or **Ham**.

## 📌 Features
✅ Preprocesses text messages (removes numbers, punctuation, and stopwords)
✅ Uses **TF-IDF Vectorization** to convert text into numerical representation
✅ Trains a **Multinomial Naïve Bayes** classifier on SMS data
✅ Evaluates model performance with **accuracy score** and **classification report**
✅ Provides a function to predict whether a new message is **Spam** or **Ham**

## 📂 Dataset
The dataset used is the **SMSSpamCollection**, which consists of labeled messages:
| Label | Definition |
|--------|------------|
| `ham`  | Non-spam messages |
| `spam` | Spam messages |

## 🛠️ Installation
To run this project, install the required dependencies using:
```bash
pip install pandas numpy scikit-learn nltk
```

## 📜 Code Implementation
### 1️⃣ Import Necessary Libraries
```python
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
```

### 2️⃣ Download Stopwords
```python
nltk.download('stopwords')
```

### 3️⃣ Load Dataset
```python
file_name = "SMSSpamCollection"  # Using the uploaded file

df = pd.read_csv(file_name, sep='\t', names=['label', 'message'], header=None, encoding='utf-8')
df['label'] = df['label'].map({'spam': 1, 'ham': 0})
```

### 4️⃣ Data Preprocessing
```python
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

df['clean_message'] = df['message'].apply(preprocess_text)
```

### 5️⃣ Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(df['clean_message'], df['label'], test_size=0.2, random_state=42)
```

### 6️⃣ Convert Text to Vectors
```python
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
```

### 7️⃣ Train Naïve Bayes Model
```python
model = MultinomialNB()
model.fit(X_train_vec, y_train)
```

### 8️⃣ Evaluate Model
```python
y_pred = model.predict(X_test_vec)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print(classification_report(y_test, y_pred))
```

### 9️⃣ Predict Spam or Ham
```python
def predict_spam(text):
    text = preprocess_text(text)
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
    return "Spam" if prediction[0] == 1 else "Ham"
```

### 🔍 Example Predictions
```python
print(predict_spam("Congratulations! You've won a $1000 gift card. Click here to claim now."))  # Expected: Spam
print(predict_spam("Hey, are we meeting at 5 PM?"))  # Expected: Ham
```

## 📊 Performance Metrics
| Metric | Value |
|--------|-------|
| Accuracy | ✅ High |
| Precision | 📊 Good |
| Recall | 📈 Optimized |
| F1-Score | 🚀 Reliable |

## 📉 Model Visualization
![Confusion Matrix](https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Precisionrecall.svg/500px-Precisionrecall.svg.png)

## 💡 Future Improvements
🔹 Implement **Deep Learning** models like LSTMs
🔹 Add **real-time SMS filtering** integration
🔹 Improve preprocessing techniques with **lemmatization**

## 👨‍💻 Contributing
Feel free to **fork** the repository, create a **new branch**, and submit a **pull request**!

## 📜 License
This project is licensed under the **MIT License**.

---

📬 **Have questions?** Reach out via [GitHub Issues](https://github.com/yourrepo/issues).

---

⭐ **If you like this project, give it a star!** ⭐
