import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import base64

def load_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Page configuration
st.set_page_config(
    page_title="CineMatch - Find Your Next Movie",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)
banner_base64 = load_image_base64("banner.jpg")

# Professional Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700;800;900&family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    html {
        scroll-behavior: smooth;
    }
    
    .stApp {
        background: linear-gradient(180deg, #0d0d12 0%, #1a1a24 50%, #0d0d12 100%);
    }
    
    #MainMenu, footer, header {visibility: hidden;}
    
    .main {
        padding: 0 !important;
        max-width: 100% !important;
    }
    
    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
    
    /* Banner Section */
    
    
    .banner-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 10.5rem;
        font-weight: 900;
        color: #ffffff;
        text-shadow: 0 4px 20px rgba(0, 0, 0, 0.8);
        letter-spacing: -2px;
        margin-bottom: 0.5rem;
    }
    
    .banner-gradient {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .banner-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem;
        color: #d1d5db;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.8);
        font-weight: 400;
    }
    
    /* Navigation Cards */
    .nav-card,
    .nav-card:visited,
    .nav-card:hover,
    .nav-card:active,
    .nav-card:focus {
        text-decoration: none !important;
        outline: none !important;
        border-bottom: none !important;
    }

    .nav-card:focus-visible {
        outline: none !important;
    }

    .nav-cards-container {
        display: flex;
        justify-content: center;
        gap: 2.5rem;
        padding: 3.5rem 2rem;
        max-width: 1400px;
        margin: 0 auto;
    }
    
    .nav-card {
        padding: 10px 16px;
        font-size: 0.95rem;
        border-radius: 12px;
        min-width: auto;
        min-height: auto;
        line-height: 1.2;
        background: rgba(255, 255, 255, 0.03);
        border: 1.5px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
        flex: 1;
    } 
    
    .nav-card:hover {
        background: rgba(99, 102, 241, 0.15);
        border-color: #6366f1;
        transform: translateY(-8px);
        box-shadow: 0 20px 50px rgba(99, 102, 241, 0.3);
    }
    
    .nav-card-icon {
        font-size: 3.5rem;
        margin-bottom: 1.2rem;
    }
    
    .nav-card-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.6rem;
    }
    
    .nav-card-desc {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        color: #9ca3af;
        line-height: 1.5;
    }
    
    /* Search Section */
    .search-section {
        padding: 4rem 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .search-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin-bottom: 3rem;
    }
    
    /* Input styling with proper height */
    .stTextInput > div {
        height: 75px;
    }
    
    .stTextInput > div > div {
        height: 10px;
    }
    
    .stTextInput input {
        height: 52px !important;
        line-height: 52px !important;
        padding: 0 1.4rem !important;
        font-size: 1rem !important;
    }
    
    .stTextInput input {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 2px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 16px !important;
        padding: 0 2rem !important;
        color: #ffffff !important;
        font-size: 1.15rem !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
        height: 70px !important;
        line-height: 70px !important;
    }
    
    .stTextInput input:focus {
        background: rgba(255, 255, 255, 0.08) !important;
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.15) !important;
    }
    
    .stTextInput input::placeholder {
        color: #9ca3af !important;
        font-size: 1.05rem !important;
    }
    
    /* Select Box with proper height */
    .stSelectbox > div {
        height: 70px;
    }
    
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 2px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 16px !important;
        color: #ffffff !important;
        height: 70px !important;
        display: flex !important;
        align-items: center !important;
    }
    
    .stSelectbox [data-baseweb="select"] {
        background: rgba(255, 255, 255, 0.05) !important;
        height: 70px !important;
    }
    
    .stSelectbox [data-baseweb="select"] > div {
        height: 70px !important;
        padding: 0 1.5rem !important;
        font-size: 1.05rem !important;
    }
    
    /* Button with proper height */
    .stButton {
        height: 70px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #7c3aed 100%) !important;
        color: white !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        padding: 0 3rem !important;
        border-radius: 16px !important;
        border: none !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        height: 70px !important;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    .stButton button:hover {
        transform: translateY(-3px);
        box-shadow: 0 20px 40px rgba(99, 102, 241, 0.5) !important;
    }
    
    /* Search controls alignment */
    .search-controls {
        display: flex;
        gap: 1.5rem;
        align-items: stretch;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .search-wrapper {
        max-width: 40px;
        margin: 0 auto;
    }

    .search-wrapper .stTextInput {
        width: 100%;
    }
    
    /* Results Section */
    .results-header {
        padding: 4rem 2rem 2.5rem 2rem;
        text-align: center;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        margin-top: 3rem;
    }
    
    .results-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.8rem;
    }
    
    .results-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.15rem;
        color: #9ca3af;
    }
    
    .results-grid {
        padding: 2.5rem;
        max-width: 1800px;
        margin: 0 auto;
    }
    
    /* Movie Card */
    .movie-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        overflow: hidden;
        transition: all 0.4s ease;
        height: 100%;
    }
    
    .movie-card:hover {
        border-color: rgba(99, 102, 241, 0.5);
        transform: translateY(-10px);
        box-shadow: 0 30px 60px rgba(0, 0, 0, 0.6);
        background: rgba(255, 255, 255, 0.05);
    }
    
    .movie-poster {
        position: relative;
        aspect-ratio: 2/3;
        overflow: hidden;
        background: #111;
    }
    
    .movie-poster img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        transition: transform 0.4s ease;
    }
    
    .movie-card:hover .movie-poster img {
        transform: scale(1.08);
    }
    
    .movie-rank {
        position: absolute;
        top: 16px;
        left: 16px;
        background: rgba(99, 102, 241, 0.95);
        backdrop-filter: blur(10px);
        color: white;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 800;
        font-size: 1rem;
        padding: 0.6rem 1.1rem;
        border-radius: 12px;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.4);
    }
    
    .movie-info {
        padding: 1.5rem;
        background: rgba(0, 0, 0, 0.4);
    }
    
    .movie-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 1rem;
        line-height: 1.4;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        min-height: 3rem;
    }
    
    .match-score {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .match-bar-container {
        flex: 1;
        height: 8px;
        background: rgba(255, 255, 255, 0.15);
        border-radius: 10px;
        overflow: hidden;
    }
    
    .match-bar {
        height: 100%;
        background: linear-gradient(90deg, #6366f1, #a855f7);
        border-radius: 10px;
        transition: width 0.7s ease;
    }
    
    .match-percentage {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        font-weight: 700;
        color: #6366f1;
    }
    
    /* How It Works Section */
    .how-it-works-section {
        padding: 5rem 2rem;
        margin-top: 3rem;
        background: rgba(255, 255, 255, 0.02);
        border-top: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 1rem;
        color: #ffffff;
    }
    
    .section-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.15rem;
        color: #9ca3af;
        text-align: center;
        max-width: 720px;
        margin: 0.8rem auto 4rem;
        line-height: 1.7;
    }
    
    .features-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        gap: 2.5rem;
        max-width: 1400px;
        margin: 0 auto;
    }
    
    .feature-card {
        background: rgba(255, 255, 255, 0.03);
        padding: 2.5rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-6px);
        background: rgba(255, 255, 255, 0.05);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1.5rem;
    }
    
    .feature-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 1rem;
    }
    
    .feature-description {
        font-family: 'Inter', sans-serif;
        font-size: 1.05rem;
        color: #9ca3af;
        line-height: 1.7;
    }
    
    /* Alerts */
    .stAlert {
        background: rgba(239, 68, 68, 0.1) !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
        border-radius: 16px !important;
        color: #fca5a5 !important;
    }
    
    </style>
""", unsafe_allow_html=True)


st.markdown(f"""
<style>
.banner-section {{
    position: relative;
    width: 100%;
    height: 500px;
    background:
        linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.75)),
        url("data:image/jpeg;base64,{banner_base64}");
    background-size: cover;
    background-position: center;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_process_data():
    try:
        movies_data = pd.read_csv('movies_preprocessed.csv')
        selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
        
        for feature in selected_features:
            movies_data[feature] = movies_data[feature].fillna('')
        
        combined_features = (
            movies_data['genres'] + ' ' + 
            movies_data['keywords'] + ' ' + 
            movies_data['tagline'] + ' ' + 
            movies_data['cast'] + ' ' + 
            movies_data['director']
        )
        
        vectorizer = TfidfVectorizer()
        feature_vectors = vectorizer.fit_transform(combined_features)
        similarity = cosine_similarity(feature_vectors)
        
        return movies_data, similarity
    except FileNotFoundError:
        st.error("❌ movies.csv file not found. Please upload the dataset.")
        return None, None

def fetch_poster(movie_title):
    try:
        search_url = f"https://api.themoviedb.org/3/search/movie?api_key=0b78123bc2f438d6b6d11c94f882d34b&query={movie_title}"
        response = requests.get(search_url, timeout=5)
        data = response.json()
        
        if data['results']:
            poster_path = data['results'][0].get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
        
        return "https://via.placeholder.com/500x750?text=No+Poster"
    except:
        return "https://via.placeholder.com/500x750?text=No+Poster"

def get_recommendations(movie_name, movies_data, similarity, num_recommendations=10):
    try:
        list_of_all_titles = movies_data['title'].tolist()
        find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles, n=1, cutoff=0.6)
        
        if not find_close_match:
            return None, "Hmm, couldn't find that one. Try another movie title?"
        
        close_match = find_close_match[0]
        index_of_the_movie = movies_data[movies_data['title'] == close_match].index[0]
        similarity_score = list(enumerate(similarity[index_of_the_movie]))
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for i, movie in enumerate(sorted_similar_movies[1:num_recommendations+1]):
            index = movie[0]
            title = movies_data.iloc[index]['title']
            score = movie[1]
            recommendations.append({
                'title': title,
                'similarity_score': score,
                'rank': i + 1
            })
        
        return recommendations, close_match
    except Exception as e:
        return None, f"Oops, something went wrong: {str(e)}"

def main():
    # Banner Section
    st.markdown("""
        <div class="banner-section">
            <h1 class="banner-title">CINE<span class="banner-gradient">MATCH</span></h1>
            <p class="banner-subtitle">Find movies you'll actually want to watch</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load data
    movies_data, similarity = load_and_process_data()
    
    if movies_data is None or similarity is None:
        st.stop()
    
    # Navigation Cards
    st.markdown("""
        <div class="nav-cards-container">
            <a class="nav-card" href="#home">
                <div class="nav-card-icon">🏠</div>
                <div class="nav-card-title">Home</div>
                <div class="nav-card-desc">Start here</div>
            </a>
            <a class="nav-card" href="#search">
                <div class="nav-card-icon">🔍</div>
                <div class="nav-card-title">Search</div>
                <div class="nav-card-desc">Find movies</div>
            </a>
            <a class="nav-card" href="#how-it-works">
                <div class="nav-card-icon">💡</div>
                <div class="nav-card-title">How It Works</div>
                <div class="nav-card-desc">Learn more</div>
            </a>
        </div>
    """, unsafe_allow_html=True)
    
    # Search Section
    st.markdown('<div class="search-section" id="search">', unsafe_allow_html=True)
    st.markdown('<h2 class="search-title">What movie do you like?</h2>', unsafe_allow_html=True)
    
    # Using proper column ratios for better alignment
    col1, col2, col3 = st.columns([3, 1.2, 1.2], vertical_alignment="bottom")
    
    with col1:
        movie_input = st.text_input(
            "movie",
            placeholder="Type a movie you enjoyed... (like Inception or The Matrix)",
            label_visibility="collapsed",
            key="movie_search"
        )
    
    with col2:
        num_recommendations = st.selectbox(
            "Results",
            options=[5, 10, 15, 20],
            index=1,
            label_visibility="collapsed",
            key="num_results"
        )
    
    with col3:
        search_button = st.button("Find Movies", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Results
    if search_button and movie_input:
        with st.spinner("Looking through our collection..."):
            recommendations, result = get_recommendations(
                movie_input, 
                movies_data, 
                similarity, 
                num_recommendations
            )
        
        if recommendations:
            st.markdown(f"""
                <div class="results-header">
                    <div class="results-title">Because you liked "{result}"</div>
                    <div class="results-subtitle">Here are {len(recommendations)} movies you might enjoy</div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="results-grid">', unsafe_allow_html=True)
            
            cols_per_row = 5
            for i in range(0, len(recommendations), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(recommendations):
                        rec = recommendations[i + j]
                        with col:
                            poster_url = fetch_poster(rec['title'])
                            match_percent = int(rec['similarity_score'] * 100)
                            
                            st.markdown(f"""
                                <div class="movie-card">
                                    <div class="movie-poster">
                                        <img src="{poster_url}" alt="{rec['title']}">
                                        <div class="movie-rank">#{rec['rank']}</div>
                                    </div>
                                    <div class="movie-info">
                                        <div class="movie-title">{rec['title']}</div>
                                        <div class="match-score">
                                            <div class="match-bar-container">
                                                <div class="match-bar" style="width: {match_percent}%"></div>
                                            </div>
                                            <div class="match-percentage">{match_percent}%</div>
                                        </div>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error(result)
    elif search_button:
        st.warning("⚠️ Type a movie name first!")
    
    # How It Works Section
    st.markdown("""
        <div class="how-it-works-section" id="how-it-works">
            <h2 class="section-title">How does this work?</h2>
            <div class="section-subtitle">
                Don't worry, no complicated stuff here. We just look at what makes movies similar 
                and help you discover ones that match your taste.
            </div>
            <div class="features-grid">
                <div class="feature-card">
                    <div class="feature-icon">🎭</div>
                    <div class="feature-title">Genre matching</div>
                    <div class="feature-description">
                        We check if movies share similar genres. If you like sci-fi thrillers, 
                        we'll find more sci-fi thrillers for you. Pretty straightforward!
                    </div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🎬</div>
                    <div class="feature-title">Same cast & directors</div>
                    <div class="feature-description">
                        Love a particular actor or director? We'll find other movies they've worked on. 
                        Simple as that.
                    </div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🔍</div>
                    <div class="feature-title">Theme detection</div>
                    <div class="feature-description">
                        We look at keywords and themes to understand what the movie's actually about. 
                        Then we find similar stories you might enjoy.
                    </div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🤖</div>
                    <div class="feature-title">Smart matching</div>
                    <div class="feature-description">
                        Behind the scenes, we use some math (TF-IDF and cosine similarity if you're curious) 
                        to calculate how similar movies are. But you don't need to worry about that part!
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

# ============================================
# SETUP INSTRUCTIONS FOR VS CODE:
# ============================================
# 
# 1. PROJECT STRUCTURE:
#    your_project_folder/
#    ├── app.py (this file)
#    ├── movies.csv (your dataset)
#    └── banner.png (your banner image)
#
# 2. BANNER IMAGE SETUP:
#    - Find a movie-themed image
#    - Save it as "banner.png" in the same folder as app.py
#
# 3. INSTALL REQUIRED PACKAGES:
#    Open terminal in VS Code and run:
#    pip install streamlit pandas scikit-learn requests
#
# 4. RUN THE APP:
#    streamlit run app.py
#
# 5. REQUIREMENTS:
#    - Python 3.7+
#    - movies.csv with columns: title, genres, keywords, tagline, cast, director
#    - banner.png image file
#    - TMDB API key (already included in code)
#
# ============================================