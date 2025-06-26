#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# ------------------ LOAD DATA ------------------ #

netflix_df = pd.read_csv("netflix_titles.csv")
imdb_df = pd.read_csv("netflix_imdb.csv")

# Rename and merge
imdb_df.rename(columns={'imdb_score': 'imdb_rating'}, inplace=True)
df = pd.merge(netflix_df, imdb_df[['title', 'imdb_rating']], on='title', how='left')

# ------------------ CLEANING ------------------ #

df['type'] = df['type'].fillna("Unknown")
df['imdb_rating'] = df['imdb_rating'].fillna(0)
df['duration'] = df['duration'].fillna("")

# Extract seasons and minutes
df['season_count'] = df['duration'].str.extract(r'(\d+)').astype(float)
df['duration_mins'] = df['duration'].apply(
    lambda x: float(x.split()[0]) if 'min' in x else None
)

# ------------------ MODEL ------------------ #

le = LabelEncoder()
df["type_encoded"] = le.fit_transform(df["type"])
X = df[["imdb_rating"]]
y = df["type_encoded"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if not os.path.exists("model.pkl"):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, "model.pkl")
else:
    model = joblib.load("model.pkl")

model_accuracy = model.score(X_test, y_test) * 100

# ------------------ STREAMLIT UI ------------------ #

st.title("\U0001F3AC Netflix Smart Recommender")

st.subheader("\U0001F50D Model Accuracy")
st.success(f"{model_accuracy:.2f}%")

st.markdown("---")

# ------------------ TIME-BASED QUESTIONS ------------------ #

st.subheader("⏱️ Let's Find You Something to Watch!")

time_available = st.slider("How many hours do you have?", 0.5, 20.0, step=0.5)

if time_available <= 4:
    st.markdown("### \U0001F3A5 You don't have much time — let's find a movie!")
    short_movies = df[(df['type'] == 'Movie') & (df['duration_mins'].notnull())]
    short_movies = short_movies[short_movies['duration_mins'] <= time_available * 60]
    top_movies = short_movies.sort_values(by='imdb_rating', ascending=False).head(5)

    for _, row in top_movies.iterrows():
        st.write(f"**{row['title']}** — IMDb Rating: {row['imdb_rating']} ({int(row['duration_mins'])} mins)")

else:
    st.markdown("### \U0001F4FA You’ve got time — let’s look at TV shows!")
    season_option = st.selectbox("How many seasons are you comfortable with?", ['1', '2', '3+'])

    # Estimate time per season (~5 hrs/season)
    if season_option == '1':
        max_seasons = 1
    elif season_option == '2':
        max_seasons = 2
    else:
        max_seasons = 100  # Any number of seasons

    tv_df = df[(df['type'] == 'TV Show') & (df['season_count'].notnull())]
    tv_df["estimated_total_time"] = tv_df["season_count"] * 5
    tv_df = tv_df[(tv_df["season_count"] <= max_seasons) & (tv_df["estimated_total_time"] <= time_available)]

    top_tv = tv_df.sort_values(by='imdb_rating', ascending=False).dropna(subset=['title']).head(5)

    st.markdown(f"### \U0001F9E0 Top TV Shows With {season_option} Season(s):")
    for _, row in top_tv.iterrows():
        st.write(f"**{row['title']}** — IMDb Rating: {row['imdb_rating']} (Est. {int(row['estimated_total_time'])} hrs)")

st.markdown("---")

# ------------------ PREDICT TYPE AND RECOMMEND ------------------ #

st.markdown("### 🤖 Predict Content Type from IMDb Rating")

user_rating = st.slider("Select an IMDb Rating", 1.0, 10.0, 6.0, step=0.01)

if st.button("Predict Type"):
    prediction = model.predict([[user_rating]])
    predicted_type = le.inverse_transform(prediction)[0]

    st.markdown(f"Based on rating **{user_rating:.2f}**, this would likely be a **{predicted_type}**.")

    # Get top 5 results near this rating from predicted type
    similar_items = df[
        (df['type'] == predicted_type) &
        (df['imdb_rating'].between(user_rating - 0.5, user_rating + 0.5))
    ].sort_values(by='imdb_rating', ascending=False).head(5)

    if not similar_items.empty:
        st.markdown(f"#### \U0001F3AC Top {predicted_type}s around IMDb Rating {user_rating:.2f}:")
        for _, row in similar_items.iterrows():
            title = row['title']
            rating = row['imdb_rating']

            if predicted_type == "Movie":
                duration = f"{int(row['duration_mins'])} mins" if pd.notnull(row['duration_mins']) else "Unknown duration"
            else:
                est_time = int(row['estimated_total_time']) if pd.notnull(row['estimated_total_time']) else "?"
                duration = f"Est. {est_time} hrs"

            st.markdown(f"**{title}** — IMDb Rating: {rating} ({duration})")
    else:
        st.warning("No similar titles found near this rating.")

st.caption("✨ Built with Netflix + IMDb Data | @Utsav Kumar")


# In[ ]:




