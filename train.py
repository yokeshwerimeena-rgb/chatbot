import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

import joblib
file_name = "question.xlsx"
df = pd.read_excel(file_name)
print("Dataset Preview:")
print(df)
X = df['QUESTION']
y = df['ANSWER']
model = Pipeline([
    
    ('tfidf', TfidfVectorizer()),
    
   
    ('classifier', MultinomialNB())
])

model.fit(X, y)
print("\nModel Training Completed!")
joblib.dump(model, 'chatbot_model.pkl')
print("Model Saved Successfully!")
