# Music Recommendation System Using CNN & Unsupervised Learning
## 📌 Overview
This project is a **music recommendation system** that utilizes **collaborative filtering** and **content-based filtering**, enhanced with **BERT embeddings** and **image analysis** to improve accuracy. The system leverages **unsupervised learning** techniques and is optimized for real-world application by integrating with **Spotify's API**.

## 🔗 Dataset
The primary dataset is sourced from:
- [Kaggle: Spotify Songs with Attributes and Lyrics](https://www.kaggle.com/datasets/bwandowando/spotify-songs-with-attributes-and-lyrics)
- Spotify API for **real-time metadata retrieval** (album covers, song details, track popularity, etc.).

### 🗂 Data Characteristics
The dataset includes:
- **id**: Unique identifier for each song.
- **name**: Song title.
- **album_name**: Album in which the song is featured.
- **artists**: Names of the performing artists.
- **lyrics**: Full song lyrics.
- **album_url**: Link to album cover image.
- **popularity**: Popularity score from Spotify.
- **audio features**: tempo, valence, danceability, energy, etc.
- **over 160,000 records**.

## 🚀 Technologies Used
- **Python** (NumPy, Pandas, Scikit-learn, TensorFlow, PyTorch)
- **Spotify API** for real-time song data retrieval
- **BERT embeddings** for lyrics-based similarity
- **CNN (Convolutional Neural Networks)** for album cover image analysis
- **Flask / FastAPI** for deployment
- **Docker** for containerization
- **PostgreSQL** / **MongoDB** for database storage

## 🏗 System Architecture
1. **Data Processing**
   - Preprocessing: Cleaning, tokenization, embedding
   - Feature Engineering: TF-IDF, Word2Vec, BERT embeddings
   - Image Processing: CNN-based album image similarity
2. **Model Training**
   - Unsupervised learning: KNN, matrix factorization, collaborative filtering
   - Hybrid model combining **content-based + collaborative filtering**
3. **Deployment**
   - API integration with Spotify
   - Web application with **ReactJS / Next.js** frontend
   - Model inference pipeline for real-time recommendations

## 📊 Results
- Evaluation Metrics: **Precision@K, Recall, F1-Score, RMSE**
- Comparative Analysis: **TF-IDF vs Word2Vec vs BERT** for lyrics embedding
- Image-based Similarity: Improved recommendation accuracy using CNNs

## 🎯 Future Improvements
- Enhance model accuracy with **Transformer-based models**
- Improve **real-time recommendation speed**
- Expand dataset to include more **global music trends**

## 📬 Contact
For any inquiries, feel free to open an issue or contact me at **your-email@example.com**.

