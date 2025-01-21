# C-Suite2
ENVISION-HACKATHON ROUND : 3

TEAM NAME : C-Suite
TEAM MEMBERS 1.SANIYA 2.NANDANI 3.KOMAL 4.Shrawni

Abstract : This project creates a Music Recommendation System by collecting data from the Spotify Web API using the Spotipy library. It gathers details on 1500+ pop tracks, including song names, artists, albums, and genres, to help generate personalized music recommendations. The data is organized with Pandas and stored in a CSV file, forming the basis for future user-driven recommendations based on preferences. project aims to help by providing personalized music recommendations based on user preferences, focusing on genre and artist information. By collecting data from the Spotify Web API, it helps users discover new songs tailored to their tastes. In the future, the system can evolve to adapt to individual listening habits, enhancing the overall music discovery experience.

Work Done: report on music recommendation system , tranined model and the app
Tool used : juypter , streamlit, spotify and pythram 

Model Used : XGBoost
Accuracy : 93.75%

The files :
finalmusic contanin the cleaned dataset 
xgboost_model.pkl has the trained model 
musicrec and similarities file are used for app
envusion file contain the report on the project 

localhost link : http://localhost:8501/
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
