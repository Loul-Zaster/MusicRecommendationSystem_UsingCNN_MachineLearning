import pickle
import streamlit as st
import spotipy
import requests
from io import BytesIO
import numpy as np
from spotipy.oauth2 import SpotifyClientCredentials
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models

# ================== Spotify API Setup ================== #
CLIENT_ID = "YOUR ID"
CLIENT_SECRET = "YOUR SECRET"

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET))

# ================== Function: Get Album Cover ================== #
def get_song_album_cover_url(song_name, artist_name):
    search_query = f"track:{song_name} artist:{artist_name}"
    results = sp.search(q=search_query, type="track", limit=1)

    if results and results["tracks"]["items"]:
        track = results["tracks"]["items"][0]
        album_cover_url = track["album"]["images"][0]["url"]
        track_url = track['external_urls']['spotify']
        return album_cover_url, track_url, track['id']
    return None, None, None

# ================== Function: Extract Image Features ================== #
def extract_image_features(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content)).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove final classification layer
    model.eval()

    with torch.no_grad():
        img_embedding = model(img_tensor)
    
    img_embedding = torch.nn.functional.avg_pool2d(img_embedding, img_embedding.shape[2:]).squeeze()  # Average pooling
    
    return img_embedding.numpy()


# ================== Function: Get Song Suggestions from Spotify ================== #
def search_songs_on_spotify(query):
    results = sp.search(q=query, type="track", limit=5)  # Get top 5 songs
    song_list = []
    
    for track in results["tracks"]["items"]:
        song_name = track["name"]
        artist_name = track["artists"][0]["name"]  # Lấy tên nghệ sĩ đầu tiên
        song_list.append(f"{song_name} - {artist_name}")  # Gộp tên bài hát + nghệ sĩ
    
    return song_list


# ================== Load Data ================== #
st.header('🎵 Music Recommender System')
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(45deg, #9ecbff, #b0e0e6);
            background-attachment: fixed;
            background-size: cover;
            color: black; /* Để chữ dễ nhìn trên nền */
        }
        .stButton>button {
            background: linear-gradient(45deg, #ff7eb3, #ff758c);
            color: black; /* Màu chữ mặc định */
            font-size: 18px;
            font-weight: bold;
            border-radius: 10px;
            padding: 10px 20px;
            transition: 0.3s ease-in-out;
            border: none;
        }

        .stButton>button:hover {
            background: linear-gradient(45deg, #ff758c, #ff7eb3);
            transform: scale(1.05);
            box-shadow: 0px 4px 10px rgba(255, 118, 136, 0.6);
            color: black; /* Khi hover, chữ đổi thành màu đen */
        }
        /* Hiệu ứng hover cho ảnh */
        .song-image {
            border-radius: 10px;
            transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
        }
        .song-image:hover {
            transform: scale(1.1);
            box-shadow: 0px 4px 15px rgba(255, 118, 136, 0.6);
        }

        /* Căn giữa toàn bộ nội dung */
        .stApp {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)
music = pickle.load(open('dataset\df_new_with_album_url.pkl', 'rb'))
similarity = pickle.load(open('similarity_matrix.pkl', 'rb'))
combined_embedding = pickle.load(open('train_model\combined_embeddings_final.pkl', 'rb'))

# ================== Song Search with Auto-suggestions ================== #
search_term = st.text_input("🔍 Search for a song", "")

if search_term:
    suggested_songs = search_songs_on_spotify(search_term)
    selected_song = st.selectbox("🎼 Select a song", suggested_songs)
else:
    selected_song = None

# ================== Function: Recommend Songs ================== #
def recommend(song):
    recommended_music_names = []
    recommended_music_artists = []
    recommended_music_posters = []
    recommended_music_links = []
    recommended_music_ids = []

    if " - " in song:  # Nếu bài hát có cả nghệ sĩ trong selectbox
        song_name, artist_name = song.split(" - ", 1)
    else:
        song_name, artist_name = song, "Unknown"

    if song_name in music['name'].values:
        index = music[music['name'] == song_name].index[0]
        distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    else:
        album_cover_url, _, _ = get_song_album_cover_url(song_name, artist_name)
        if album_cover_url:
            song_embedding = extract_image_features(album_cover_url)
            if song_embedding.shape[0] != combined_embedding.shape[1]:
                song_embedding = np.resize(song_embedding, (combined_embedding.shape[1],))
        else:
            song_embedding = np.random.rand(combined_embedding.shape[1])

        if combined_embedding.shape[1] == song_embedding.shape[0]:
            distances = sorted(enumerate(np.linalg.norm(combined_embedding - song_embedding, axis=1)), key=lambda x: x[1])
        else:
            st.error("Lỗi: Kích thước embedding không khớp!")
            return [], [], [], [], []

    for i in range(min(5, len(distances))):
        idx = distances[i][0]
        song_title = music.iloc[idx]['name']
        artist = music.iloc[idx]['artists']
        album_cover_url, track_url, track_id = get_song_album_cover_url(song_title, artist)
        
        if not album_cover_url:
            album_cover_url = "https://i.postimg.cc/0QNxYz4V/social.png"

        recommended_music_names.append(song_title)
        recommended_music_artists.append(artist)
        recommended_music_posters.append(album_cover_url)
        recommended_music_links.append(track_url)
        recommended_music_ids.append(track_id)

    return recommended_music_names, recommended_music_artists, recommended_music_posters, recommended_music_links, recommended_music_ids


# ================== Show Recommendations ================== #
if st.button('🎶 Show Recommendation'):
    if selected_song:
        recommended_music_names, recommended_music_artists, recommended_music_posters, recommended_music_links, recommended_music_ids = recommend(selected_song)

        cols = st.columns(5)
        for i in range(len(recommended_music_names)):
            with cols[i]:
                st.image(recommended_music_posters[i], width=150)
                st.text(recommended_music_names[i])

                if recommended_music_links[i]:
                    st.markdown(f'<a href="{recommended_music_links[i]}" target="_blank">🎧 [Listen on Spotify]</a>', unsafe_allow_html=True)

