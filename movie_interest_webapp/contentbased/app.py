

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Book Recommender - Content Based")

books = pd.read_csv(r"C:\Users\Dhanushree\books.csv")
books['features'] = books['Title'] + " " + books['Author'] + " " + books['Genre']

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(books['features'])
content_similarity = cosine_similarity(tfidf_matrix)

book_titles = books['Title'].tolist()
selected = st.selectbox("Choose a Book", book_titles)
index = books[books['Title'] == selected].index[0]

if st.button("Recommend"):
    similar_books = content_similarity[index].argsort()[::-1][1:4]
    st.write("### Recommended Books:")
    for i in similar_books:
        st.write("- " + books.iloc[i]['Title'])



## 💻 Streamlit Code (Unique Version)
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
books = pd.read_csv(r"C:\Users\Dhanushree\books.csv")
ratings = pd.read_csv(r"C:\Users\Dhanushree\ratings.csv")

# Preprocessing for Content-Based Filtering
books['features'] = books['Title'] + " " + books['Author'] + " " + books['Genre']
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(books['features'])
content_similarity = cosine_similarity(tfidf_matrix)

# Preprocessing for Collaborative Filtering
user_item_matrix = ratings.pivot_table(index='User_ID', columns='Book_ID', values='Rating').fillna(0)
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# --- Web Layout ---
st.title("📖 Priyadharshini Book Suggestor")

# Mode selector
mode = st.selectbox("Choose Recommendation Mode:", ["🔍 Content-Based", "👥 Collaborative", "🧬 Hybrid", "🎲 Surprise Picks", "📈 Popular Books"])

# Shared selections
book_titles = books['Title'].tolist()
user_ids = user_item_matrix.index.tolist()

# --- Content-Based Mode ---
if mode == "🔍 Content-Based":
    selected_book = st.selectbox("Pick a Book Title", book_titles)
    if st.button("Get Suggestions"):
        index = books[books['Title'] == selected_book].index[0]
        similar_books = content_similarity[index].argsort()[::-1][1:4]
        st.subheader("Books with Similar Style:")
        for i in similar_books:
            st.write("✅", books.iloc[i]['Title'])

# --- Collaborative Mode ---
elif mode == "👥 Collaborative":
    selected_user = st.selectbox("Pick a User ID", user_ids)
    if st.button("Fetch Recommendations"):
        sim_users = user_similarity_df.loc[selected_user].sort_values(ascending=False)[1:4].index
        sim_ratings = user_item_matrix.loc[sim_users].mean().sort_values(ascending=False)
        user_rated_books = user_item_matrix.loc[selected_user][user_item_matrix.loc[selected_user] > 0].index
        recs = sim_ratings.drop(user_rated_books).head(3)

        st.subheader("Readers Like You Also Enjoyed:")
        for book_id in recs.index:
            title = books[books['Book_ID'] == book_id]['Title'].values[0]
            st.write("✨", title)

# --- Hybrid Mode ---
elif mode == "🧬 Hybrid":
    user = st.selectbox("Select User ID", user_ids)
    book = st.selectbox("Select Book", book_titles)

    if st.button("Hybrid Suggest"):
        book_id = books[books['Title'] == book]['Book_ID'].values[0]
        book_index = books[books['Book_ID'] == book_id].index[0]
        content_scores = content_similarity[book_index]
        user_ratings = user_item_matrix.loc
