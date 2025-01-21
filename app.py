import streamlit as st
import pickle
import pandas as pd
import requests
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

CLIENT_ID = "1bb83b4d552941bf9ad2140b5872aaae"
CLIENT_SECRET = "05f06c2d79d74329802329ce04b0aebb"


client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

music_dict = pickle.load(open(r"C:\Users\saniy\musicrec.pkl", 'rb'))
music = pd.DataFrame(music_dict)
similarity = pickle.load(open(r"C:\Users\saniy\similarities.pkl", 'rb'))

def fetch_poster(music_title):
    try:
        results = sp.search(q=music_title, type="track", limit=1)

        if results['tracks']['items']:
            track = results['tracks']['items'][0]
            album_cover_url = track['album']['images'][0]['url']
            return album_cover_url
        else:
            raise ValueError("No track found with the name: " + music_title)
    except (KeyError, IndexError, requests.exceptions.RequestException, ValueError) as e:
        print(f"Error fetching poster for {music_title}: {e}")
        return "https://i.postimg.cc/0QNxYz4V/social.png"

def recommend(music, similarity, selected_music_name=None, mood=None, language=None):

    if not selected_music_name:
        return [], []

    try:
        m_i = music[music['title'] == selected_music_name].index[0]
        d = similarity[m_i]
        m_l = sorted(list(enumerate(d)), reverse=True, key=lambda x: x[1])[1:6]
    except IndexError:
        return [], []


        m_l = []
        for idx in music_filtered.index:
            d = similarity[idx]
            m_l.extend(sorted(list(enumerate(d)), reverse=True, key=lambda x: x[1])[1:6])

        m_l = sorted(set(m_l), key=lambda x: x[1], reverse=True)[:5]

    recommend_music = []
    recommend_music_poster = []
    for i in m_l:
        if i[0] < len(music):  # Validate index bounds
            music_title = music.iloc[i[0]]['title']
            recommend_music.append(music_title)
            poster_url = fetch_poster(music_title)
            recommend_music_poster.append(poster_url)

    return recommend_music, recommend_music_poster

st.title('Music Recommendation System')

recommendation_method = st.radio("Choose recommendation method", ["Search by Song"])

if recommendation_method == "Search by Song":
    selected_music_name = st.selectbox('Select a music you like', music['title'].values)
    if st.button('Recommend'):
        names, posters = recommend(music, similarity, selected_music_name=selected_music_name)
        if names:
            # Create columns for displaying recommendations in a row
            cols = st.columns(5)  # You can adjust the number of columns based on how many recommendations you want to display
            for i in range(len(names)):
                with cols[i]:
                    st.text(names[i])
                    st.image(posters[i], use_container_width=True)  # Updated to use use_container_width
        else:
            st.write("No recommendations found.")

# elif recommendation_method == "Choose by Mood and Language":
#     mood_options = ['happy', 'Romantic']
#     language_options = ['hindi']
#
#     mood = st.selectbox("Select your mood", mood_options)
#     language = st.selectbox("Select language", language_options)
#
#     if st.button('Recommend'):
#         names, posters = recommend(music, similarity, mood=mood, language=language)
#         if names:
#             # Create columns for displaying recommendations in a row
#             cols = st.columns(5)  # You can adjust the number of columns based on how many recommendations you want to display
#             for i in range(len(names)):
#                 with cols[i]:
#                     st.text(names[i])
#                     st.image(posters[i], use_container_width=True)  # Updated to use use_container_width
#         else:
#             st.write("No recommendations found.")
