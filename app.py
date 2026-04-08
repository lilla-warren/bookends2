import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from datetime import datetime

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Bookends | Your Storybook Adventure",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== STORYBOOK CSS ====================
st.markdown("""
<style>
    /* Import storybook fonts */
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;500;600;700&family=Quicksand:wght@300;400;500;600;700&display=swap');
    
    /* Main container - warm parchment background */
    .stApp {
        background: linear-gradient(135deg, #fdf6e3 0%, #f5e6d3 100%);
    }
    
    /* Header - magical storybook style */
    .main-header {
        text-align: center;
        padding: 2.5rem 2rem;
        background: linear-gradient(135deg, #2c1810 0%, #4a2c1a 100%);
        border-radius: 30px;
        margin-bottom: 2rem;
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
        border: 3px solid #d4a574;
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: "✨";
        position: absolute;
        top: 10px;
        left: 20px;
        font-size: 2rem;
        opacity: 0.3;
    }
    
    .main-header::after {
        content: "✨";
        position: absolute;
        bottom: 10px;
        right: 20px;
        font-size: 2rem;
        opacity: 0.3;
    }
    
    .main-header h1 {
        color: #f5e6d3;
        margin: 0;
        font-family: 'Cinzel', serif;
        font-size: 2.8rem;
        font-weight: 700;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
        letter-spacing: 2px;
    }
    
    .main-header p {
        color: #e8d5b7;
        margin-top: 0.8rem;
        font-family: 'Quicksand', sans-serif;
        font-size: 1.1rem;
        font-style: italic;
    }
    
    /* Storybook card */
    .story-card {
        background: #fffef7;
        padding: 1.2rem 1.5rem;
        border-radius: 20px;
        margin: 0.8rem 0;
        border: 2px solid #e8d5b7;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        position: relative;
    }
    
    .story-card:hover {
        transform: translateY(-5px) rotate(0.5deg);
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        border-color: #c4a47a;
    }
    
    .story-card h4 {
        color: #2c1810;
        font-family: 'Cinzel', serif;
        margin: 0 0 8px 0;
        font-size: 1.2rem;
    }
    
    .story-card p {
        color: #4a3728;
        font-family: 'Quicksand', sans-serif;
        margin: 5px 0;
    }
    
    /* Button styling - magical */
    .stButton > button {
        background: linear-gradient(135deg, #8b5e3c 0%, #6b3e1c 100%);
        color: #fdf6e3;
        border: 2px solid #d4a574;
        padding: 0.6rem 2rem;
        border-radius: 40px;
        font-weight: 600;
        font-family: 'Quicksand', sans-serif;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 3px 8px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(107, 62, 28, 0.4);
        background: linear-gradient(135deg, #9b6e4c 0%, #7b4e2c 100%);
        border-color: #e8c49a;
        color: white;
    }
    
    /* Sidebar - vintage book style */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c1810 0%, #1a0f08 100%);
        border-right: 3px solid #d4a574;
    }
    
    [data-testid="stSidebar"] * {
        color: #f5e6d3 !important;
        font-family: 'Quicksand', sans-serif;
    }
    
    [data-testid="stSidebar"] .stRadio label {
        font-size: 1rem;
        padding: 0.5rem;
    }
    
    /* Input fields - readable */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #d4a574;
        background: #fffef7;
        color: #2c1810;
        padding: 0.6rem 1.2rem;
        font-family: 'Quicksand', sans-serif;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #8b5e3c;
        box-shadow: 0 0 0 3px rgba(139, 94, 60, 0.1);
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #c4a47a;
    }
    
    /* Select box */
    .stSelectbox > div > div {
        border-radius: 25px;
        border: 2px solid #d4a574;
        background: #fffef7;
        color: #2c1810;
    }
    
    /* Tabs - storybook style */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: rgba(255, 254, 247, 0.8);
        padding: 0.5rem;
        border-radius: 50px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 30px;
        padding: 0.5rem 1.5rem;
        font-family: 'Cinzel', serif;
        font-weight: 600;
        color: #4a3728;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #8b5e3c 0%, #6b3e1c 100%);
        color: #fdf6e3 !important;
    }
    
    /* Metrics - vintage cards */
    .metric-card {
        background: #fffef7;
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        border: 2px solid #e8d5b7;
        box-shadow: 0 3px 10px rgba(0,0,0,0.05);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: #c4a47a;
    }
    
    .metric-card h3 {
        color: #8b5e3c;
        font-family: 'Cinzel', serif;
        margin: 0;
        font-size: 1.8rem;
    }
    
    .metric-card p {
        color: #4a3728;
        margin: 5px 0 0 0;
        font-family: 'Quicksand', sans-serif;
    }
    
    /* Chat messages */
    .chat-message-user {
        background: linear-gradient(135deg, #8b5e3c 0%, #6b3e1c 100%);
        color: #fdf6e3;
        padding: 0.8rem 1.2rem;
        border-radius: 20px;
        border-bottom-right-radius: 5px;
        max-width: 70%;
        margin-left: auto;
        margin-bottom: 1rem;
        font-family: 'Quicksand', sans-serif;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .chat-message-bot {
        background: #fffef7;
        color: #2c1810;
        padding: 0.8rem 1.2rem;
        border-radius: 20px;
        border-bottom-left-radius: 5px;
        max-width: 70%;
        margin-right: auto;
        margin-bottom: 1rem;
        font-family: 'Quicksand', sans-serif;
        border: 2px solid #e8d5b7;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Welcome banner */
    .welcome-banner {
        background: linear-gradient(135deg, #fffef7 0%, #fdf0e0 100%);
        padding: 1.5rem;
        border-radius: 25px;
        margin-bottom: 1.5rem;
        border: 2px solid #e8d5b7;
        text-align: center;
        box-shadow: 0 5px 20px rgba(0,0,0,0.05);
    }
    
    .welcome-banner h2 {
        color: #8b5e3c;
        font-family: 'Cinzel', serif;
        margin: 0;
    }
    
    .welcome-banner p {
        color: #4a3728;
        font-family: 'Quicksand', sans-serif;
        margin: 8px 0 0 0;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #fffef7;
        border-radius: 20px;
        border: 2px solid #e8d5b7;
        color: #2c1810;
        font-family: 'Quicksand', sans-serif;
        font-weight: 600;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 15px;
        font-family: 'Quicksand', sans-serif;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #8b5e3c;
        font-family: 'Quicksand', sans-serif;
        font-size: 0.9rem;
        margin-top: 2rem;
        border-top: 2px solid #e8d5b7;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #2c1810;
        font-family: 'Cinzel', serif;
    }
    
    h1 {
        font-size: 2.2rem;
    }
    
    h2 {
        font-size: 1.8rem;
        border-left: 4px solid #8b5e3c;
        padding-left: 1rem;
    }
    
    h3 {
        font-size: 1.4rem;
    }
    
    /* Regular text */
    p, li, label {
        color: #4a3728;
        font-family: 'Quicksand', sans-serif;
    }
    
    /* Animations */
    @keyframes pageFlip {
        0% {
            opacity: 0;
            transform: rotateY(-10deg);
        }
        100% {
            opacity: 1;
            transform: rotateY(0deg);
        }
    }
    
    .fade-in {
        animation: pageFlip 0.6s ease-out;
    }
    
    /* Decorative elements */
    .decorative-leaf {
        position: fixed;
        bottom: 20px;
        left: 20px;
        opacity: 0.3;
        font-size: 3rem;
        pointer-events: none;
    }
</style>
""", unsafe_allow_html=True)

# ==================== DATA LOADING FUNCTIONS ====================
@st.cache_data
def load_books_data():
    """Load books data from Excel or use sample data"""
    try:
        possible_paths = [
            "DetailedBooksExcel Cleaned (RemovedBlank).xlsx",
            "data/DetailedBooksExcel Cleaned (RemovedBlank).xlsx",
            "../DetailedBooksExcel Cleaned (RemovedBlank).xlsx"
        ]
        
        books = None
        for path in possible_paths:
            if os.path.exists(path):
                books = pd.read_excel(path)
                break
        
        if books is None:
            return create_storybook_data()
        
        books = books.dropna(subset=['Book Title', 'Author', 'Genre'])
        books['Book Title'] = books['Book Title'].astype(str)
        books['Author'] = books['Author'].astype(str)
        books['Genre'] = books['Genre'].astype(str).str.lower()
        books['combined'] = books['Book Title'] + " " + books['Author'] + " " + books['Genre']
        
        return books
    except Exception as e:
        return create_storybook_data()

def create_storybook_data():
    """Create enchanting storybook data"""
    sample_data = {
        'Book Title': [
            'Atomic Habits', 'The Psychology of Money', 'Deep Work', 
            'The 7 Habits of Highly Effective People', 'Think and Grow Rich',
            'Rich Dad Poor Dad', 'The Intelligent Investor', 'Zero to One',
            'Sapiens', 'Becoming', 'Educated', 'The Alchemist',
            'Harry Potter and the Sorcerer\'s Stone', 'The Hobbit', '1984',
            'Dune', 'Pride and Prejudice', 'The Great Gatsby',
            'To Kill a Mockingbird', 'The Catcher in the Rye',
            'The Little Prince', 'Alice in Wonderland'
        ],
        'Author': [
            'James Clear', 'Morgan Housel', 'Cal Newport', 
            'Stephen R. Covey', 'Napoleon Hill', 'Robert Kiyosaki',
            'Benjamin Graham', 'Peter Thiel', 'Yuval Noah Harari',
            'Michelle Obama', 'Tara Westover', 'Paulo Coelho',
            'J.K. Rowling', 'J.R.R. Tolkien', 'George Orwell',
            'Frank Herbert', 'Jane Austen', 'F. Scott Fitzgerald',
            'Harper Lee', 'J.D. Salinger',
            'Antoine de Saint-Exupéry', 'Lewis Carroll'
        ],
        'Genre': [
            'self-help', 'finance', 'productivity', 
            'self-help', 'self-help', 'finance', 
            'finance', 'business', 'history',
            'memoir', 'memoir', 'fiction',
            'fantasy', 'fantasy', 'fiction',
            'sci-fi', 'romance', 'classic',
            'classic', 'classic',
            'children', 'fantasy'
        ]
    }
    books = pd.DataFrame(sample_data)
    books['combined'] = books['Book Title'] + " " + books['Author'] + " " + books['Genre']
    return books

# ==================== RECOMMENDATION FUNCTIONS ====================
@st.cache_resource
def initialize_recommender(books):
    """Initialize TF-IDF and similarity matrix"""
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(books['combined'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return tfidf, tfidf_matrix, cosine_sim

def recommend_by_title(title, books, cosine_sim, top_n=5):
    """Recommend books based on title"""
    if title not in books['Book Title'].values:
        return []
    
    try:
        idx = books[books['Book Title'] == title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        
        recommendations = []
        for i, score in sim_scores:
            recommendations.append({
                'title': books['Book Title'].iloc[i],
                'author': books['Author'].iloc[i],
                'genre': books['Genre'].iloc[i],
                'similarity': score
            })
        
        return recommendations
    except Exception as e:
        return []

def recommend_by_genre(genre, books, top_n=6):
    """Recommend books based on genre"""
    genre = genre.lower()
    filtered = books[books['Genre'].str.contains(genre, na=False)]
    
    if filtered.empty:
        return []
    
    if len(filtered) > top_n:
        filtered = filtered.sample(min(top_n, len(filtered)))
    
    recommendations = []
    for _, row in filtered.iterrows():
        recommendations.append({
            'title': row['Book Title'],
            'author': row['Author'],
            'genre': row['Genre'],
            'similarity': 1.0
        })
    
    return recommendations

def vibe_recommend(text, books, top_n=6):
    """Recommend books based on vibe/text search"""
    text = text.lower()
    filtered = books[books['combined'].str.lower().str.contains(text, na=False)]
    
    if filtered.empty:
        keywords = text.split()
        mask = pd.Series([False] * len(books))
        for keyword in keywords:
            if len(keyword) > 2:
                mask = mask | books['combined'].str.lower().str.contains(keyword, na=False)
        filtered = books[mask]
    
    if filtered.empty:
        return []
    
    if len(filtered) > top_n:
        filtered = filtered.sample(min(top_n, len(filtered)))
    
    recommendations = []
    for _, row in filtered.iterrows():
        recommendations.append({
            'title': row['Book Title'],
            'author': row['Author'],
            'genre': row['Genre'],
            'similarity': 1.0
        })
    
    return recommendations

# ==================== FAQ FUNCTIONS ====================
FAQ_DATA = [
    {"q": "where is your location", 
     "a": "📖 **Our Enchanted Location**\n\nDubai Digital Park, Silicon Oasis Building A3, Lower Ground\n\n🕐 Open daily 10am–10pm\n\nCome visit our magical book realm!"},
    
    {"q": "can i sell books", 
     "a": "✨ **Yes, dear reader!**\n\nWe welcome your beloved books into our collection. You'll receive store credit or cash once your books find new homes."},
    
    {"q": "free delivery", 
     "a": "📚 **Free Delivery Magic**\n\nFree delivery for all orders above AED 180! Your books will arrive in 2-3 business days."},
    
    {"q": "delivery cost", 
     "a": "🚚 **Delivery Enchantment**\n\n• Dubai/Sharjah/Ajman: AED 19\n• Other Emirates: AED 24\n• Free for orders above AED 180"},
    
    {"q": "pick up books", 
     "a": "📦 **Pickup Adventure**\n\n• AED 25 up to 5kg\n• AED 2 per extra kg\n• Available during our magical hours"},
    
    {"q": "operating hours", 
     "a": "🕐 **When Our Doors Are Open**\n\nMonday - Sunday: 10am - 10pm\nOpen 7 days a week for your literary adventures!"}
]

@st.cache_resource
def initialize_faq_bot():
    """Initialize FAQ system"""
    questions = [item["q"] for item in FAQ_DATA]
    vectorizer = TfidfVectorizer()
    faq_matrix = vectorizer.fit_transform(questions)
    return vectorizer, faq_matrix, questions

def get_faq_answer(user_input, vectorizer, faq_matrix, questions):
    """Get answer for FAQ question"""
    try:
        user_vec = vectorizer.transform([user_input.lower()])
        similarity = cosine_similarity(user_vec, faq_matrix)
        index = similarity.argmax()
        
        if similarity[0][index] < 0.2:
            return "📖 *Turns the page thoughtfully...*\n\nI couldn't find an exact answer. Try asking about:\n• 📍 Our location\n• 🚚 Delivery\n• 💫 Selling books\n• 🕐 Store hours"
        
        for item in FAQ_DATA:
            if item["q"] == questions[index]:
                return item["a"]
        
        return "Please visit our enchanted store or contact our guardians for assistance."
    except Exception as e:
        return f"Error: {str(e)}"

# ==================== UI COMPONENTS ====================
def display_recommendations(recommendations, title="✨ Tales We Found For You"):
    """Display recommendations in beautiful storybook cards"""
    if not recommendations:
        st.info("📖 *No tales found... Try a different chapter!*")
        return
    
    st.markdown(f"### {title}")
    
    cols = st.columns(2)
    
    for idx, book in enumerate(recommendations):
        with cols[idx % 2]:
            similarity_percent = int(book['similarity'] * 100) if book['similarity'] < 1 else 85
            st.markdown(f"""
            <div class="story-card">
                <h4>📚 {book['title']}</h4>
                <p>✍️ <em>by {book['author']}</em></p>
                <p>🏷️ <em>{book['genre'].title()} • Match: {similarity_percent}%</em></p>
            </div>
            """, unsafe_allow_html=True)

def display_storybook_stats(books):
    """Display statistics in storybook style"""
    st.markdown("### 📖 Our Library's Tale")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(books)}</h3>
            <p>📚 Tales Within</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{books['Author'].nunique()}</h3>
            <p>✍️ Storytellers</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{books['Genre'].nunique()}</h3>
            <p>🏷️ Genres</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        most_common = books['Genre'].mode()[0].title()
        st.markdown(f"""
        <div class="metric-card">
            <h3>{most_common}</h3>
            <p>🌟 Most Popular</p>
        </div>
        """, unsafe_allow_html=True)

# ==================== MAIN APP ====================
def main():
    # Load data
    with st.spinner("📖 Opening the magical book..."):
        books = load_books_data()
    
    if books is not None and not books.empty:
        # Initialize systems
        _, _, cosine_sim = initialize_recommender(books)
        faq_vectorizer, faq_matrix, faq_questions = initialize_faq_bot()
        
        # Storybook Header
        st.markdown("""
        <div class="main-header">
            <h1>📖 The Enchanted Bookshelf</h1>
            <p>Where every book holds a magical story waiting to be discovered...</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Welcome message
        current_hour = datetime.now().hour
        if current_hour < 12:
            greeting = "Good morning, dear reader"
        elif current_hour < 17:
            greeting = "Good afternoon, story seeker"
        else:
            greeting = "Good evening, book wanderer"
        
        st.markdown(f"""
        <div class="welcome-banner">
            <h2>✨ {greeting} ✨</h2>
            <p>Welcome to your magical book realm. Let me help you find your next adventure...</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display stats
        display_storybook_stats(books)
        
        # Sidebar - Storybook navigation
        with st.sidebar:
            st.markdown("### 🏰 Navigation")
            st.markdown("---")
            
            menu = st.radio(
                "Choose your path:",
                ["🏠 Home", "🔍 Find Tales", "💬 Ask the Sage", "📊 Library Map", "📜 About This Realm"],
                index=0
            )
            
            st.markdown("---")
            st.markdown("*✨ Let the stories guide you...*")
        
        # Home Page
        if menu == "🏠 Home":
            st.markdown("## 🌟 Featured Tales")
            st.markdown("*Popular stories our readers love...*")
            
            # Get some popular recommendations
            popular_titles = books['Book Title'].sample(min(6, len(books))).tolist()
            recommendations = []
            for title in popular_titles:
                idx = books[books['Book Title'] == title].index[0]
                recommendations.append({
                    'title': title,
                    'author': books['Author'].iloc[idx],
                    'genre': books['Genre'].iloc[idx],
                    'similarity': 0.85
                })
            
            display_recommendations(recommendations, "✨ Treasures From Our Shelves")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                ### 🎭 How Our Magic Works
                
                1. **Choose your path** - By genre, title, or mood
                2. **Our magical scrolls** search through tales
                3. **Discover** your next great adventure
                
                *Each recommendation is crafted with care...*
                """)
            
            with col2:
                st.markdown("""
                ### ✨ Magical Features
                
                - 🔮 **Wise Sage** - Ask any question
                - 📚 **Genre Portals** - Explore by category  
                - 🎯 **Similar Spells** - Find books like your favorites
                - 💭 **Mood Magic** - Search by feeling
                """)
        
        # Find Tales (Book Recommender)
        elif menu == "🔍 Find Tales":
            st.markdown("## 🔍 Seek Your Next Adventure")
            st.markdown("*Let me help you discover a tale that calls to your heart...*")
            
            tab1, tab2, tab3 = st.tabs(["📚 By Genre Portal", "🎯 By Beloved Tale", "💭 By Magical Mood"])
            
            with tab1:
                st.markdown("### Enter a genre portal")
                
                genres = sorted(books['Genre'].unique())
                genre_options = [g.title() for g in genres]
                selected_genre = st.selectbox("Choose your portal:", genre_options)
                
                if st.button("🔮 Enter the Portal", key="genre_btn"):
                    with st.spinner("Opening the portal..."):
                        results = recommend_by_genre(selected_genre.lower(), books)
                        display_recommendations(results, f"✨ Tales from the {selected_genre} Realm")
            
            with tab2:
                st.markdown("### Find tales similar to a beloved story")
                
                titles = books['Book Title'].tolist()
                selected_title = st.selectbox("Which tale do you love?", titles)
                
                if st.button("🔮 Find Similar Tales", key="title_btn"):
                    with st.spinner("Searching through ancient scrolls..."):
                        results = recommend_by_title(selected_title, books, cosine_sim)
                        display_recommendations(results, f"📚 Tales similar to '{selected_title}'")
            
            with tab3:
                st.markdown("### Describe the mood you seek")
                st.caption("*What feelings do you want your next book to bring?*")
                
                mood_examples = ["inspiring", "mysterious", "heartwarming", "adventurous", "magical"]
                selected_mood = st.selectbox("Try a magical mood:", ["Choose a feeling..."] + mood_examples)
                
                mood_text = st.text_area("Or describe in your own words:", 
                                        value=selected_mood if selected_mood != "Choose a feeling..." else "",
                                        placeholder="e.g., 'tales that make my heart soar' or 'mysterious journeys through enchanted lands'")
                
                if st.button("🔮 Seek by Mood", key="vibe_btn"):
                    if mood_text:
                        with st.spinner("Listening to your heart's desire..."):
                            results = vibe_recommend(mood_text, books)
                            display_recommendations(results, "✨ Tales That Match Your Mood")
                    else:
                        st.warning("Please share the mood you're seeking...")
        
        # Ask the Sage (FAQ Chatbot)
        elif menu == "💬 Ask the Sage":
            st.markdown("## 💬 The Wise Sage")
            st.markdown("*Our ancient guardian knows all about our realm... Ask anything!*")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### Common Questions")
                st.markdown("""
                Travelers often ask:
                - 📍 Where is your location?
                - 📚 Can I sell books?
                - 🚚 Do you offer free delivery?
                - 🕐 What are your hours?
                """)
            
            with col2:
                st.markdown("### Speak with the Sage")
                
                if "messages" not in st.session_state:
                    st.session_state.messages = [{
                        "role": "assistant",
                        "content": "✨ Greetings, traveler! I am the guardian of this enchanted bookshelf. What wisdom do you seek today? ✨"
                    }]
                
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        st.markdown(f'<div class="chat-message-user">🧙‍♂️ {msg["content"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="chat-message-bot">📖 {msg["content"]}</div>', unsafe_allow_html=True)
                
                user_question = st.text_input("Your question:", key="faq_input", placeholder="e.g., Where is your enchanted location?")
                
                if st.button("Ask the Sage", key="send_btn"):
                    if user_question:
                        st.session_state.messages.append({"role": "user", "content": user_question})
                        with st.spinner("The sage ponders..."):
                            answer = get_faq_answer(user_question, faq_vectorizer, faq_matrix, faq_questions)
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                        st.rerun()
                
                if st.button("🗑️ Clear the Scroll"):
                    st.session_state.messages = [{
                        "role": "assistant",
                        "content": "✨ Greetings, traveler! I am the guardian of this enchanted bookshelf. What wisdom do you seek today? ✨"
                    }]
                    st.rerun()
        
        # Library Map (Dashboard)
        elif menu == "📊 Library Map":
            st.markdown("## 📊 Map of Our Library")
            st.markdown("*Visualizing the treasures within our collection...*")
            
            tab1, tab2 = st.tabs(["🏷️ Genre Landscape", "✍️ Storyteller's Circle"])
            
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    genre_counts = books['Genre'].value_counts()
                    fig, ax = plt.subplots(figsize=(8, 6))
                    colors = ['#8b5e3c', '#a0704c', '#b8886c', '#c49a7a', '#d4aa8a', '#e4ba9a']
                    genre_counts.head(8).plot(kind='barh', ax=ax, color=colors)
                    ax.set_xlabel("Number of Books", fontweight='bold', color='#2c1810')
                    ax.set_title("Our Enchanted Genres", fontweight='bold', size=14, color='#2c1810')
                    ax.set_facecolor('#fffef7')
                    fig.patch.set_facecolor('#fffef7')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.tick_params(colors='#4a3728')
                    st.pyplot(fig)
                
                with col2:
                    fig2, ax2 = plt.subplots(figsize=(8, 8))
                    genre_counts.head(8).plot(kind='pie', ax=ax2, autopct='%1.1f%%', colors=colors)
                    ax2.set_ylabel("")
                    ax2.set_title("Genre Distribution", fontweight='bold', size=14, color='#2c1810')
                    ax2.set_facecolor('#fffef7')
                    fig2.patch.set_facecolor('#fffef7')
                    for text in ax2.texts:
                        text.set_color('#2c1810')
                    st.pyplot(fig2)
            
            with tab2:
                top_authors = books['Author'].value_counts().head(10)
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                top_authors.plot(kind='bar', ax=ax3, color='#8b5e3c')
                ax3.set_xlabel("Storyteller", fontweight='bold', color='#2c1810')
                ax3.set_ylabel("Number of Tales", fontweight='bold', color='#2c1810')
                ax3.set_title("Our Most Prolific Storytellers", fontweight='bold', size=14, color='#2c1810')
                ax3.tick_params(axis='x', rotation=45, colors='#4a3728')
                ax3.tick_params(axis='y', colors='#4a3728')
                ax3.spines['top'].set_visible(False)
                ax3.spines['right'].set_visible(False)
                ax3.set_facecolor('#fffef7')
                fig3.patch.set_facecolor('#fffef7')
                st.pyplot(fig3)
        
        # About
        elif menu == "📜 About This Realm":
            st.markdown("## 📜 The Story of Our Realm")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("""
                ### ✨ Welcome, Dear Reader
                
                Once upon a time, in the heart of Dubai's Silicon Oasis, a magical bookstore was born...
                
                **Bookends** is more than just a bookstore - it's a sanctuary for story lovers, a gathering place for dreamers, and a portal to countless adventures.
                
                #### 🔮 Our Magical System
                
                This enchanted AI uses ancient wisdom (and modern magic) to help you discover books:
                
                1. **Content-Based Sorcery**: We analyze the essence of each tale
                2. **TF-IDV Enchantment**: Ancient runes that understand text
                3. **Cosine Similarity Magic**: Finds kindred spirits among books
                4. **Wisdom Ranking**: Presents the most relevant adventures first
                
                #### 🏰 Visit Our Physical Realm
                
                **Location:** Dubai Digital Park, Silicon Oasis Building A3, Lower Ground
                
                **Hours:** Daily, 10:00 AM - 10:00 PM
                
                **Contact:** support@bookends.ae
                
                #### ✨ Our Promise
                
                Every recommendation is crafted with care, every question answered with wisdom, and every reader treated like the hero of their own story.
                """)
            
            with col2:
                st.markdown("""
                ### 🛡️ Our Magical Tools
                
                - 🐍 **Python** - Our wand
                - 🎨 **Streamlit** - Our canvas
                - 🤖 **Scikit-learn** - Our spellbook
                - 📊 **Pandas** - Our library catalog
                
                ---
                
                ### 📜 Version Lore
                
                **Bookends AI v2.0**
                
                *The Enchanted Edition*
                
                Released: December 2024
                
                ---
                
                ### 💫 A Final Word
                
                *"A reader lives a thousand lives before they die..."*
                
                — George R.R. Martin
                """)
        
        # Footer
        st.markdown("""
        <div class="footer">
            <p>📖 *Every book is a new adventure waiting to begin* 📖</p>
            <p style="font-size: 0.8rem;">© 2024 Bookends Enchanted Bookshelf | Where stories come alive</p>
            <p style="font-size: 0.7rem;">✨ May your next read be magical ✨</p>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.error("The magical library is sleeping... Please refresh the page to wake it!")

# ==================== RUN THE MAGIC ====================
if __name__ == "__main__":
    main()
