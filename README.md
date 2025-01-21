# C-Suite2
ENVISION-HACKATHON ROUND : 3

TEAM NAME : C-Suite
TEAM MEMBERS 
1.SANIYA 
2.NANDANI 
3.KOMAL 
4.Shrawni

Abstract : This project creates a Music Recommendation System by collecting data from the Spotify Web API using the Spotipy library. It gathers details on 1500+ pop tracks, including song names, artists, albums, and genres, to help generate personalized music recommendations. The data is organized with Pandas and stored in a CSV file, forming the basis for future user-driven recommendations based on preferences. project aims to help by providing personalized music recommendations based on user preferences, focusing on genre and artist information. By collecting data from the Spotify Web API, it helps users discover new songs tailored to their tastes. In the future, the system can evolve to adapt to individual listening habits, enhancing the overall music discovery experience.

Work Done: report on music recommendation system , tranined model and the app
Tool used : juypter , streamlit, spotify and pythram 

Model Used : XGBoost
Accuracy : 93.75%

The files :
finalmusic contanin the cleaned dataset after webscrapping ,
xgboost_model.pkl has the trained model ,
musicrec and similarities file are used for app,
envision file contain the report on the project ,
music recom system this file contain all the code which made the project .

localhost link : http://localhost:8501/
  Local URL: http://localhost:8501
  Network URL: http://192.168.1.7:8501
where my modle is performing



code:
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import pandas as pd

X = df.drop(columns=['Genre'])  
y = df['Genre']  

for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])

X['Mood'] = pd.to_numeric(X['Mood'], errors='coerce')

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_decoded = label_encoder.inverse_transform(y_pred)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred, labels=valid_labels)
class_report = classification_report(
    y_test, y_pred, labels=valid_labels, target_names=[str(label) for label in valid_labels]
)
print(f'Accuracy: {accuracy * 100:.2f}%') 
print('Confusion Matrix:')
print(conf_matrix) 
print('Classification Report:')
print(class_report)

OUTPUT:
Accuracy: 93.15%
Confusion Matrix:
[[31  0  0  0  0  0  0  0  0  0]
 [ 0  1  0  0  0  0  0  0  0  0]
 [ 0  0  4  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0 17  0  0  0  0  0]
 [ 0  0  0  0  0  2  0  0  0  0]
 [ 0  0  0  0  0  0 38  0  0  0]
 [ 0  0  0  0  0  0  0  5  0  0]
 [ 0  0  0  0  0  0  0  0 31  0]
 [ 0  1  0  0  0  0  0  0  0  1]]
Classification Report:
              precision    recall  f1-score   support

          11       0.91      0.91      0.91        34
          12       0.50      1.00      0.67         1
          13       1.00      1.00      1.00         4
          14       0.00      0.00      0.00         0
          15       1.00      1.00      1.00        17
          16       1.00      1.00      1.00         2
          17       1.00      1.00      1.00        38
          18       1.00      1.00      1.00         5
          19       1.00      1.00      1.00        31
          20       1.00      0.50      0.67         2

   micro avg       0.97      0.97      0.97       134
   macro avg       0.84      0.84      0.82       134
weighted avg       0.97      0.97      0.97       134

app.py file contains :
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



final work : 
![Screenshot 2025-01-21 065437](https://github.com/user-attachments/assets/4e8060de-1262-42c8-8678-0afc38dc46b7)
