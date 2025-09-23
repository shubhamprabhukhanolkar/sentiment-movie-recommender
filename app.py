import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
import requests
from bs4 import BeautifulSoup
from imdb import IMDb
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page configuration
st.set_page_config(
    page_title="🎬 Mood-Based Movie Recommender",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main background and theme */
    .main {
        padding-top: 2rem;
    }
    
    /* Custom header styling */
    .main-header {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Emotion display styling */
    .emotion-display {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        color: white;
        font-size: 1.4rem;
        font-weight: bold;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    /* Movie card styling */
    .movie-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem;
        transition: all 0.3s ease;
        height: 420px;
        display: flex;
        flex-direction: column;
    }
    
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .movie-poster {
        width: 100%;
        height: 280px;
        object-fit: cover;
        border-radius: 10px;
        margin-bottom: 0.8rem;
    }
    
    .movie-title {
        font-weight: bold;
        font-size: 1rem;
        margin-bottom: 0.5rem;
        text-align: center;
        flex-grow: 1;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .movie-info {
        font-size: 0.85rem;
        color: #888;
        text-align: center;
        margin-top: auto;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #4ecdc4;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #ff6b6b;
        box-shadow: 0 0 0 0.2rem rgba(255, 107, 107, 0.25);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Sidebar styling */
    .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    
    /* Animation for loading */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .loading {
        animation: pulse 1.5s infinite;
    }
    
    /* Grid layout for movies */
    .movie-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load all necessary files
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('sa_text.h5')

@st.cache_resource
def load_tokenizer():
    with open('tokenizer.pickle', 'rb') as handle:
        return pickle.load(handle)

@st.cache_resource
def load_max_length():
    with open('max_length.pickle', 'rb') as handle:
        return pickle.load(handle)

@st.cache_data
def load_dataset():
    with open('dataset.pkl', 'rb') as f:
        return pickle.load(f)

# Load models and data
try:
    model = load_model()
    tokenizer = load_tokenizer()
    max_length = load_max_length()
    df_filtered1 = load_dataset()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"Error loading models: {str(e)}")

# Mappings
label_mapping = {
    0: "😢 Sadness", 
    1: "😂 Joy", 
    2: "❤️ Love", 
    3: "😡 Anger", 
    4: "😨 Fear", 
    5: "😲 Surprise"
}

mood_mapping = {
    "😂 Joy": 0, 
    "❤️ Love": 0, 
    "😢 Sadness": 1, 
    "😨 Fear": 1, 
    "😡 Anger": 1, 
    "😲 Surprise": 2
}

def normalize(text):
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    stop_words = {"i", "am", "the", "is", "and", "a", "to", "of", "in", "for", "on", "with", "as", "by", "at", "an"}
    word_tokens = text.split()
    return ' '.join(word for word in word_tokens if word not in stop_words)

def predict_emotion(text):
    if not models_loaded:
        return 0, "😊 Unknown"
    
    normalized_input = normalize(text)
    input_sequence = tokenizer.texts_to_sequences([normalized_input])
    input_padded = pad_sequences(input_sequence, maxlen=max_length)
    prediction = model.predict(input_padded)
    predicted_label = np.argmax(prediction, axis=1)[0]
    return predicted_label, label_mapping[predicted_label]

def recommend_movies(cluster_value):
    if not models_loaded:
        return pd.DataFrame({'title': ['Sample Movie 1', 'Sample Movie 2']})
    
    filtered_movies = df_filtered1[df_filtered1['cluster'] == cluster_value]
    top_movies = filtered_movies.head(16)
    return top_movies[['title']]

def get_movie_info(title):
    try:
        ia = IMDb()
        movies = ia.search_movie(title)
        if not movies:
            return {
                'title': title, 
                'poster': None, 
                'description': 'No description available', 
                'year': 'N/A', 
                'rating': 'N/A', 
                'imdb_link': '#'
            }

        movie = movies[0]
        ia.update(movie)

        # Try to get poster from OMDB or use placeholder
        poster_url = None
        try:
            query = title.replace(' ', '+') + '+movie+poster'
            url = f'https://www.google.com/search?q={query}&tbm=isch'
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(response.content, 'html.parser')
            images = soup.find_all('img')
            if len(images) > 1:
                poster_url = images[1].get('src')
        except:
            poster_url = None

        return {
            'title': title,
            'poster': poster_url,
            'description': movie.get('plot outline', ['No plot available'])[0] if movie.get('plot outline') else 'No plot available',
            'year': movie.get('year', 'N/A'),
            'rating': movie.get('rating', 'N/A'),
            'imdb_link': f"https://www.imdb.com/title/tt{movie.movieID}/"
        }
    except Exception as e:
        return {
            'title': title, 
            'poster': None, 
            'description': 'Information not available', 
            'year': 'N/A', 
            'rating': 'N/A', 
            'imdb_link': '#'
        }

# Main App Interface
def main():
    # Header
    st.markdown('<h1 class="main-header">🎭 Mood-Based Movie Recommender</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">✨ Tell us how you feel, and we\'ll recommend the perfect movies for your mood! ✨</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-content">
            <h3>🎬 How it works</h3>
            <p>1. Enter a sentence describing your current mood</p>
            <p>2. Our AI analyzes your emotions</p>
            <p>3. Get personalized movie recommendations</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""
        <div class="sidebar-content">
            <h4>🎯 Emotion Categories</h4>
            <p>😂 Joy • ❤️ Love • 😢 Sadness</p>
            <p>😡 Anger • 😨 Fear • 😲 Surprise</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Example inputs
        st.markdown("### 💡 Try these examples:")
        example_inputs = [
            "I'm feeling great and want to celebrate!",
            "I'm sad and need comfort",
            "I'm angry about everything today",
            "I'm scared and anxious",
            "I'm in love and everything is beautiful",
            "I'm surprised by recent events"
        ]
        
        for example in example_inputs:
            if st.button(f"🎲 {example[:30]}...", key=example, help=example):
                st.session_state.example_input = example

    # Main content area
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Text input options
        input_method = st.radio(
            "Choose input method:",
            ["✏️ Type your mood", "📝 Write a longer description"],
            horizontal=True
        )
        
        if input_method == "✏️ Type your mood":
            default_text = st.session_state.get('example_input', "I am feeling happy and excited today!")
            user_input = st.text_input(
                "🗣️ **Tell us about your mood:**", 
                value=default_text,
                placeholder="Type how you're feeling right now...",
                help="Be specific about your emotions for better recommendations!"
            )
        else:
            default_text = st.session_state.get('example_input', "Today has been an amazing day! I woke up feeling energetic and positive. Everything seems to be going my way and I want to watch something that matches this upbeat mood.")
            user_input = st.text_area(
                "📝 **Describe your mood in detail:**",
                value=default_text,
                height=100,
                placeholder="Write a detailed description of how you're feeling...",
                help="The more detail you provide, the better our recommendations!"
            )
        
        # Clear the session state after using it
        if 'example_input' in st.session_state:
            del st.session_state.example_input
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Predict button
        if st.button("🎯 **Analyze My Mood & Get Recommendations**", type="primary"):
            if user_input.strip() != "":
                # Emotion prediction
                with st.spinner('🤖 Analyzing your emotions...'):
                    predicted_label, predicted_emotion = predict_emotion(user_input)
                    mood = mood_mapping.get(predicted_emotion, 2)
                
                # Display detected emotion
                st.markdown(f"""
                <div class="emotion-display">
                    🎭 Detected Emotion: <strong>{predicted_emotion}</strong>
                </div>
                """, unsafe_allow_html=True)
                
                # Get movie recommendations
                with st.spinner('🎬 Finding perfect movies for your mood...'):
                    recommended_movies = recommend_movies(mood)
                
                st.markdown("---")
                st.markdown("## 🍿 **Your Personalized Movie Recommendations**")
                
                if len(recommended_movies) > 0:
                    # Create columns for movie grid
                    movies_per_row = 4
                    movie_data = []
                    
                    # Fetch movie information
                    progress_bar = st.progress(0)
                    for idx, title in enumerate(recommended_movies['title']):
                        movie_info = get_movie_info(title)
                        movie_data.append(movie_info)
                        progress_bar.progress((idx + 1) / len(recommended_movies))
                    
                    progress_bar.empty()
                    
                    # Display movies in grid
                    for i in range(0, len(movie_data), movies_per_row):
                        cols = st.columns(movies_per_row)
                        for j, col in enumerate(cols):
                            if i + j < len(movie_data):
                                movie = movie_data[i + j]
                                with col:
                                    # Fallback poster
                                    fallback_poster = 'https://via.placeholder.com/300x450/333333/ffffff?text=🎬+Movie+Poster'
                                    poster = movie['poster'] if movie['poster'] else fallback_poster
                                    
                                    st.markdown(f"""
                                    <div class="movie-card">
                                        <a href="{movie['imdb_link']}" target="_blank">
                                            <img src="{poster}" class="movie-poster" alt="{movie['title']}" onerror="this.src='{fallback_poster}'"/>
                                        </a>
                                        <div class="movie-title">
                                            <strong>{i+j+1}. {movie['title']}</strong>
                                        </div>
                                        <div class="movie-info">
                                            ⭐ {movie['rating']} | 📅 {movie['year']}
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                else:
                    st.warning("😕 No movie recommendations found. Please try a different mood description.")
            else:
                st.warning("🤔 Please enter some text describing your mood!")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>🎬 Made with ❤️ using Streamlit & TensorFlow</p>
        <p>🚀 Discover movies that match your emotional state perfectly!</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()