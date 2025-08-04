import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
data = {
    'text': [
        'I love this', 'Amazing movie', 'Best film ever',
        'Horrible experience', 'Worst movie', 'Terrible acting'
    ],
    'label': [1, 1, 1, 0, 0, 0]
}

df = pd.DataFrame(data)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))  # Should be 1.0 with this data
print("Report:\n", classification_report(y_test, y_pred))
new_message=['I love this']
new_features=vectorizer.transform(new_message)
predictions=model.predict(new_features)
