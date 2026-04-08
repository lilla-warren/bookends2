import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from datetime import datetime
import random

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Bookends | AI Book Recommendation",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS FOR PROFESSIONAL LOOK ====================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        animation: fadeIn 0.8s ease-in;
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        margin-top: 0.5rem;
        font-size: 1.1rem;
    }
    
    /* Card styling */
    .recommendation-card {
        background: white;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        margin: 0.7rem 0;
        border: 1px solid #e0e7ff;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .recommendation-card:hover {
        transform: translateX(8px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-left: 4px solid #3b82f6;
        background: linear-gradient(90deg, #ffffff 0%, #f8fafc 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 12px;
        font-weight: 500;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59,130,246,0.3);
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.1);
    }
    
    /* Select box styling */
    .stSelectbox > div > div {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Chat message styling */
    .chat-message-user {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 18px;
        border-bottom-right-radius: 4px;
        max-width: 70%;
        margin-left: auto;
        margin-bottom: 1rem;
    }
    
    .chat-message-bot {
        background: #f1f5f9;
        color: #1e293b;
        padding: 0.8rem 1.2rem;
        border-radius: 18px;
        border-bottom-left-radius: 4px;
        max-width: 70%;
        margin-right: auto;
        margin-bottom: 1rem;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: #f8fafc;
        border-radius: 12px;
        font-weight: 500;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 2rem;
        border-top: 1px solid #e2e8f0;
    }
    
    /* Welcome banner */
    .welcome-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
    }
    
    /* Stats container */
    .stats-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
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
            return create_sample_books_data()
        
        books = books.dropna(subset=['Book Title', 'Author', 'Genre'])
        books['Book Title'] = books['Book Title'].astype(str)
        books['Author'] = books['Author'].astype(str)
        books['Genre'] = books['Genre'].astype(str).str.lower()
        books['combined'] = books['Book Title'] + " " + books['Author'] + " " + books['Genre']
        
        return books
    except Exception as e:
        return create_sample_books_data()

def create_sample_books_data():
    """Create enhanced sample book data for demo"""
    sample_data = {
        'Book Title': [
            'Atomic Habits', 'The Psychology of Money', 'Deep Work', 
            'The 7 Habits of Highly Effective People', 'Think and Grow Rich',
            'Rich Dad Poor Dad', 'The Intelligent Investor', 'Zero to One',
            'Sapiens', 'Becoming', 'Educated', 'The Alchemist',
            'Harry Potter and the Sorcerer\'s Stone', 'The Hobbit', '1984',
            'Dune', 'Pride and Prejudice', 'The Great Gatsby',
            'To Kill a Mockingbird', 'The Catcher in the Rye'
        ],
        'Author': [
            'James Clear', 'Morgan Housel', 'Cal Newport', 
            'Stephen R. Covey', 'Napoleon Hill', 'Robert Kiyosaki',
            'Benjamin Graham', 'Peter Thiel', 'Yuval Noah Harari',
            'Michelle Obama', 'Tara Westover', 'Paulo Coelho',
            'J.K. Rowling', 'J.R.R. Tolkien', 'George Orwell',
            'Frank Herbert', 'Jane Austen', 'F. Scott Fitzgerald',
            'Harper Lee', 'J.D. Salinger'
        ],
        'Genre': [
            'self-help', 'finance', 'productivity', 
            'self-help', 'self-help', 'finance', 
            'finance', 'business', 'history',
            'memoir', 'memoir', 'fiction',
            'fantasy', 'fantasy', 'fiction',
            'sci-fi', 'romance', 'classic',
            'classic', 'classic'
        ],
        'Rating': [4.8, 4.6, 4.7, 4.5, 4.4, 4.5, 4.3, 4.6, 4.7, 4.8, 4.6, 4.5, 4.9, 4.8, 4.7, 4.6, 4.5, 4.4, 4.8, 4.3]
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
            book_title = books['Book Title'].iloc[i]
            author = books['Author'].iloc[i]
            genre = books['Genre'].iloc[i]
            rating = books.get('Rating', [4.5]*len(books)).iloc[i]
            recommendations.append({
                'title': book_title,
                'author': author,
                'genre': genre,
                'rating': rating,
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
            'rating': row.get('Rating', 4.5),
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
            'rating': row.get('Rating', 4.5),
            'similarity': 1.0
        })
    
    return recommendations

def get_popular_books(books, top_n=6):
    """Get popular books based on rating"""
    if 'Rating' in books.columns:
        popular = books.nlargest(top_n, 'Rating')
    else:
        popular = books.head(top_n)
    
    recommendations = []
    for _, row in popular.iterrows():
        recommendations.append({
            'title': row['Book Title'],
            'author': row['Author'],
            'genre': row['Genre'],
            'rating': row.get('Rating', 4.5),
            'similarity': 1.0
        })
    
    return recommendations

# ==================== FAQ FUNCTIONS ====================
FAQ_DATA = [
    {"q": "where is your location", 
     "a": "📍 **Dubai Digital Park**, Silicon Oasis Building A3, Lower Ground\n\n🕐 Open daily 10am–10pm"},
    {"q": "can i sell books", 
     "a": "✅ **Yes, absolutely!**\n\nWe accept books in good condition. You'll receive store credit or cash once your books are sold."},
    {"q": "free delivery", 
     "a": "🚚 **Free delivery** for orders above **AED 180**\n\nStandard delivery takes 2-3 business days."},
    {"q": "delivery cost", 
     "a": "💰 **Delivery fees:**\n• Dubai/Sharjah/Ajman: AED 19\n• Other Emirates: AED 24\n• Free for orders above AED 180"},
    {"q": "pick up books", 
     "a": "📦 **Pickup service:**\n• AED 25 up to 5kg\n• AED 2 per extra kg\n• Available during store hours"},
    {"q": "operating hours", 
     "a": "🕐 **Store Hours:**\n• Monday - Sunday: 10am - 10pm\n• Open 7 days a week!"},
    {"q": "how to use credit", 
     "a": "🎯 **Using your credit:**\nSimply mention your account during checkout (online or in-store), and we'll deduct it manually."},
    {"q": "can i cancel order", 
     "a": "✅ **Yes**, you can cancel your order within 24 hours of purchase. Please contact us with your order number."},
    {"q": "return policy", 
     "a": "🔄 **14-day return policy**\n\nBooks can be returned within 14 days with original receipt. Books must be in original condition."}
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
            return "🤔 I couldn't find an exact match. Try asking about:\n• Location & hours\n• Delivery & pickup\n• Selling books\n• Returns & cancellations"
        
        for item in FAQ_DATA:
            if item["q"] == questions[index]:
                return item["a"]
        
        return "Please visit our store or contact support for assistance."
    except Exception as e:
        return f"Error: {str(e)}"

# ==================== UI COMPONENTS ====================
def display_recommendations(recommendations, title="Recommended Books"):
    """Display recommendations in a beautiful grid"""
    if not recommendations:
        st.warning("No recommendations found. Try different criteria!")
        return
    
    st.markdown(f"### {title}")
    
    # Create columns for grid layout
    cols = st.columns(2)
    
    for idx, book in enumerate(recommendations):
        with cols[idx % 2]:
            # Rating stars
            rating_stars = "⭐" * int(book['rating']) + "☆" * (5 - int(book['rating']))
            
            st.markdown(f"""
            <div class="recommendation-card">
                <h4 style="margin: 0 0 5px 0; color: #1e293b;">📖 {book['title']}</h4>
                <p style="margin: 5px 0; color: #64748b;">✍️ by {book['author']}</p>
                <p style="margin: 5px 0;">
                    <span style="background: #e0e7ff; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem;">🏷️ {book['genre'].title()}</span>
                    <span style="margin-left: 8px;">{rating_stars} {book['rating']}</span>
                </p>
            </div>
            """, unsafe_allow_html=True)

def display_welcome_section():
    """Display welcome banner with stats"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="welcome-banner">
            <h2>✨ Welcome to Bookends AI</h2>
            <p>Your intelligent companion for discovering amazing books</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        current_hour = datetime.now().hour
        if current_hour < 12:
            greeting = "🌅 Good Morning!"
        elif current_hour < 17:
            greeting = "☀️ Good Afternoon!"
        else:
            greeting = "🌙 Good Evening!"
        
        st.markdown(f"""
        <div class="stats-container" style="text-align: center;">
            <h3>{greeting}</h3>
            <p style="font-size: 0.9rem;">Ready to find your next great read?</p>
        </div>
        """, unsafe_allow_html=True)

def display_quick_stats(books):
    """Display quick statistics in a row"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📚 {len(books)}</h3>
            <p>Total Books</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>✍️ {books['Author'].nunique()}</h3>
            <p>Authors</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🏷️ {books['Genre'].nunique()}</h3>
            <p>Genres</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_rating = books.get('Rating', pd.Series([4.5]*len(books))).mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>⭐ {avg_rating:.1f}</h3>
            <p>Avg Rating</p>
        </div>
        """, unsafe_allow_html=True)

# ==================== MAIN APP ====================
def main():
    # Load data
    with st.spinner("📚 Loading your library..."):
        books = load_books_data()
    
    if books is not None and not books.empty:
        # Initialize systems
        _, _, cosine_sim = initialize_recommender(books)
        faq_vectorizer, faq_matrix, faq_questions = initialize_faq_bot()
        
        # Welcome section
        display_welcome_section()
        
        # Quick stats
        display_quick_stats(books)
        
        # Sidebar navigation
        with st.sidebar:
            st.image("https://cdn-icons-png.flaticon.com/512/1674/1674816.png", width=80)
            st.markdown("### 📚 Bookends AI")
            st.markdown("---")
            
            menu = st.radio(
                "Navigation",
                ["🏠 Home", "🔍 Book Recommender", "💬 FAQ Assistant", "📊 Insights", "ℹ️ About"],
                index=0
            )
            
            st.markdown("---")
            st.caption("Made with ❤️ for book lovers")
        
        # Home Page
        if menu == "🏠 Home":
            st.markdown("### 🔥 Popular Picks This Week")
            popular = get_popular_books(books)
            display_recommendations(popular, "")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                ### 🎯 How It Works
                
                1. **Choose a recommendation method**
                   - By Genre
                   - By Book Title
                   - By Vibe/Description
                
                2. **Get AI-powered suggestions**
                   - Smart similarity matching
                   - Personalized recommendations
                
                3. **Discover your next favorite book!**
                """)
            
            with col2:
                st.markdown("""
                ### ✨ Features
                
                - 🤖 **AI-Powered Recommendations**
                - 💬 **Smart FAQ Chatbot**
                - 📊 **Interactive Dashboard**
                - ⭐ **Ratings & Reviews**
                - 🎨 **Beautiful Interface**
                """)
        
        # Book Recommender
        elif menu == "🔍 Book Recommender":
            st.markdown("## 🔍 Find Your Perfect Book")
            
            # Create tabs for different recommendation types
            tab1, tab2, tab3 = st.tabs(["📖 By Genre", "🎯 By Title", "💭 By Vibe"])
            
            with tab1:
                st.markdown("### Explore by Genre")
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    genres = books['Genre'].unique()
                    genre_options = sorted([g.title() for g in genres])
                    selected_genre = st.selectbox("Select a genre:", genre_options)
                
                with col2:
                    if st.button("🔍 Find Books", key="genre_btn", use_container_width=True):
                        with st.spinner("Finding books for you..."):
                            results = recommend_by_genre(selected_genre.lower(), books)
                            display_recommendations(results, f"Top {selected_genre} Books")
            
            with tab2:
                st.markdown("### Find Similar Books")
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    titles = books['Book Title'].tolist()
                    selected_title = st.selectbox("Select a book you love:", titles)
                
                with col2:
                    if st.button("🔍 Find Similar", key="title_btn", use_container_width=True):
                        with st.spinner("Finding similar books..."):
                            results = recommend_by_title(selected_title, books, cosine_sim)
                            display_recommendations(results, f"Books similar to '{selected_title}'")
            
            with tab3:
                st.markdown("### Describe Your Vibe")
                st.caption("Tell me what you're in the mood for...")
                
                vibe_examples = ["inspiring stories", "page-turners", "mind-blowing ideas", "emotional journeys"]
                selected_vibe = st.selectbox("Try an example:", ["Choose an example..."] + vibe_examples)
                
                vibe_text = st.text_area("Or describe in your own words:", 
                                        value=selected_vibe if selected_vibe != "Choose an example..." else "",
                                        placeholder="e.g., 'books that make me think about success' or 'adventurous stories with magic'")
                
                if st.button("🔍 Find Matching Books", key="vibe_btn", use_container_width=True):
                    if vibe_text:
                        with st.spinner("Finding books that match your vibe..."):
                            results = vibe_recommend(vibe_text, books)
                            display_recommendations(results, "Books Matching Your Vibe")
                    else:
                        st.warning("Please describe what you're looking for!")
        
        # FAQ Assistant
        elif menu == "💬 FAQ Assistant":
            st.markdown("## 💬 Ask Me Anything")
            st.markdown("Get instant answers about our store, policies, and services")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### Popular Questions")
                faq_questions = [item["q"] for item in FAQ_DATA]
                quick_q = st.selectbox("Quick questions:", faq_questions)
                if st.button("Ask This Question"):
                    answer = get_faq_answer(quick_q, faq_vectorizer, faq_matrix, faq_questions)
                    st.info(answer)
            
            with col2:
                st.markdown("### Chat with Us")
                
                if "messages" not in st.session_state:
                    st.session_state.messages = [{
                        "role": "assistant",
                        "content": "👋 Hi! I'm Bookends AI Assistant. Ask me anything about our store, delivery, selling books, or store policies!"
                    }]
                
                # Display chat history
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        st.markdown(f'<div class="chat-message-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="chat-message-bot">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
                
                # Chat input
                user_question = st.text_input("Type your question:", key="faq_input", placeholder="e.g., Where is your location?")
                
                if st.button("Send", key="send_btn"):
                    if user_question:
                        st.session_state.messages.append({"role": "user", "content": user_question})
                        with st.spinner("Thinking..."):
                            answer = get_faq_answer(user_question, faq_vectorizer, faq_matrix, faq_questions)
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                        st.rerun()
                
                if st.button("🗑️ Clear Chat History"):
                    st.session_state.messages = [{
                        "role": "assistant",
                        "content": "👋 Hi! I'm Bookends AI Assistant. How can I help you today?"
                    }]
                    st.rerun()
        
        # Insights Dashboard
        elif menu == "📊 Insights":
            st.markdown("## 📊 Library Insights")
            st.markdown("Discover fascinating statistics about our collection")
            
            tab1, tab2 = st.tabs(["📈 Genre Distribution", "👨‍💼 Top Authors"])
            
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    genre_counts = books['Genre'].value_counts()
                    fig, ax = plt.subplots(figsize=(8, 6))
                    colors = ['#3b82f6', '#60a5fa', '#93c5fd', '#bfdbfe', '#dbeafe', '#eff6ff']
                    genre_counts.head(8).plot(kind='barh', ax=ax, color=colors)
                    ax.set_xlabel("Number of Books", fontweight='bold')
                    ax.set_title("Top Genres", fontweight='bold', size=14)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    st.pyplot(fig)
                
                with col2:
                    fig2, ax2 = plt.subplots(figsize=(8, 8))
                    genre_counts.head(8).plot(kind='pie', ax=ax2, autopct='%1.1f%%', colors=colors)
                    ax2.set_ylabel("")
                    ax2.set_title("Genre Distribution", fontweight='bold', size=14)
                    st.pyplot(fig2)
            
            with tab2:
                top_authors = books['Author'].value_counts().head(10)
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                top_authors.plot(kind='bar', ax=ax3, color='#3b82f6')
                ax3.set_xlabel("Author", fontweight='bold')
                ax3.set_ylabel("Number of Books", fontweight='bold')
                ax3.set_title("Most Prolific Authors", fontweight='bold', size=14)
                ax3.tick_params(axis='x', rotation=45)
                ax3.spines['top'].set_visible(False)
                ax3.spines['right'].set_visible(False)
                st.pyplot(fig3)
            
            # Additional stats
            st.markdown("---")
            st.markdown("### 📊 Quick Facts")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                most_common_genre = books['Genre'].mode()[0]
                st.metric("Most Popular Genre", most_common_genre.title())
            with col2:
                total_books = len(books)
                st.metric("Total Collection", total_books)
            with col3:
                if 'Rating' in books.columns:
                    highest_rated = books.loc[books['Rating'].idxmax(), 'Book Title']
                    st.metric("Highest Rated", highest_rated[:20] + "...")
        
        # About
        elif menu == "ℹ️ About":
            st.markdown("## ℹ️ About Bookends AI")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("""
                ### 📚 Your AI-Powered Book Companion
                
                Bookends AI uses cutting-edge machine learning to help you discover books you'll love.
                
                #### 🤖 How It Works
                
                1. **Content-Based Filtering**: We analyze book features (title, author, genre)
                2. **TF-IDF Vectorization**: Converts text into numerical features for comparison
                3. **Cosine Similarity**: Finds books most similar to your preferences
                4. **Smart Ranking**: Presents recommendations in order of relevance
                
                #### ✨ Features
                
                - **Smart Recommendations**: Find books by genre, title, or vibe
                - **FAQ Chatbot**: Instant answers to common questions
                - **Interactive Dashboard**: Visual insights about our collection
                - **Beautiful Interface**: Modern, responsive design
                
                #### 🏢 Visit Us
                
                **Location:** Dubai Digital Park, Silicon Oasis Building A3, Lower Ground
                
                **Hours:** Daily, 10:00 AM - 10:00 PM
                
                **Contact:** support@bookends.ae
                """)
            
            with col2:
                st.markdown("""
                ### 🛠️ Tech Stack
                
                - 🐍 **Python** 3.9+
                - 🎨 **Streamlit** for UI
                - 🤖 **Scikit-learn** for ML
                - 📊 **Pandas/NumPy** for data
                - 📈 **Matplotlib** for viz
                
                ---
                
                ### 📊 Version
                
                **Current Version:** 2.0
                
                **Last Updated:** December 2024
                
                ---
                
                ### 💡 Pro Tips
                
                - Try different recommendation methods
                - Use the FAQ bot for quick answers
                - Check Insights for popular genres
                """)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div class="footer">
            <p>© 2024 Bookends AI Book Recommendation System | Made with ❤️ for book lovers</p>
            <p style="font-size: 0.8rem;">Powered by AI | Your next favorite book is just a click away</p>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.error("Failed to load data. Please refresh the page.")

# ==================== RUN APP ====================
if __name__ == "__main__":
    main()
