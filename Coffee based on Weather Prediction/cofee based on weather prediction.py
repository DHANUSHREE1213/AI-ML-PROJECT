import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Original dataset
data = {
    'Weather': ['Sunny', 'Rainy', 'Overcast', 'Sunny', 'Rainy', 'Sunny', 'Overcast', 'Rainy', 'Sunny', 'Rainy'],
    'TimeOfDay': ['Morning', 'Morning', 'Afternoon', 'Afternoon', 'Evening', 'Morning', 'Morning', 'Afternoon', 'Evening', 'Morning'],
    'SleepQuality': ['Poor', 'Good', 'Poor', 'Good', 'Poor', 'Good', 'Poor', 'Good', 'Good', 'Poor'],
    'Mood': ['Tired', 'Fresh', 'Tired', 'Energetic', 'Tired', 'Fresh', 'Tired', 'Tired', 'Energetic', 'Tired'],
    'BuyCoffee': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)

# Encoding
encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    encoders[column] = le

X = df.drop('BuyCoffee', axis=1)
y = df['BuyCoffee']

# Train Decision Tree (ID3 using entropy)
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X, y)

# Streamlit UI
st.title("Buy Coffee Predictor")
st.markdown("Predict whether a person will buy coffee based on their mood and conditions.")

# User input
weather = st.selectbox("Weather", encoders['Weather'].classes_)
timeofday = st.selectbox("Time of Day", encoders['TimeOfDay'].classes_)
sleep = st.selectbox("Sleep Quality", encoders['SleepQuality'].classes_)
mood = st.selectbox("Mood", encoders['Mood'].classes_)

if st.button("Predict"):
    sample = pd.DataFrame([{
        'Weather': encoders['Weather'].transform([weather])[0],
        'TimeOfDay': encoders['TimeOfDay'].transform([timeofday])[0],
        'SleepQuality': encoders['SleepQuality'].transform([sleep])[0],
        'Mood': encoders['Mood'].transform([mood])[0]
    }])
    prediction = model.predict(sample)[0]
    result = encoders['BuyCoffee'].inverse_transform([prediction])[0]
    st.success(f"Prediction: The person will **{'BUY' if result == 'Yes' else 'NOT BUY'}** coffee.")

# Plot the decision tree
st.subheader("Decision Tree")
fig, ax = plt.subplots(figsize=(12, 5))
plot_tree(model, feature_names=X.columns, class_names=encoders['BuyCoffee'].classes_, filled=True)
st.pyplot(fig)
