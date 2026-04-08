import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# ==================== DATA LOADING FUNCTIONS ====================
@st.cache_data
def load_books_data():
    """Load books data from Excel or use sample data"""
    try:
        # Try to find the Excel file in different locations
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
        st.warning(f"Using sample data: {str(e)}")
        return create_sample_books_data()

def create_sample_books_data():
    """Create sample book data for demo"""
    sample_data = {
        'Book Title': [
            'Atomic Habits', 'The Psychology of Money', 'Deep Work', 
            'The 7 Habits of Highly Effective People', 'Think and Grow Rich',
            'Rich Dad Poor Dad', 'The Intelligent Investor', 'Zero to One',
            'Sapiens', 'Becoming', 'Educated', 'The Alchemist',
            'Harry Potter and the Sorcerer\'s Stone', 'The Hobbit', '1984'
        ],
        'Author': [
            'James Clear', 'Morgan Housel', 'Cal Newport', 
            'Stephen R. Covey', 'Napoleon Hill', 'Robert Kiyosaki',
            'Benjamin Graham', 'Peter Thiel', 'Yuval Noah Harari',
            'Michelle Obama', 'Tara Westover', 'Paulo Coelho',
            'J.K. Rowling', 'J.R.R. Tolkien', 'George Orwell'
        ],
        'Genre': [
            'self-help', 'finance', 'productivity', 
            'self-help', 'self-help', 'finance', 
            'finance', 'business', 'history',
            'memoir', 'memoir', 'fiction',
            'fantasy', 'fantasy', 'fiction'
        ]
    }
    books = pd.DataFrame(sample_data)
    books['combined'] = books['Book Title'] + " " + books['Author'] + " " + books['Genre']
    return books

# ==================== RECOMMENDATION FUNCTIONS ====================
@st.cache_resource
def initialize_recommender(books):
    """Initialize TF-IDF and similarity matrix"""
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(books['combined'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return tfidf, tfidf_matrix, cosine_sim

def recommend_by_title(title, books, cosine_sim, top_n=5):
    """Recommend books based on title"""
    if title not in books['Book Title'].values:
        return [f"❌ '{title}' not found. Please try another title."]
    
    try:
        idx = books[books['Book Title'] == title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        
        recommendations = []
        for i in sim_scores:
            book_title = books['Book Title'].iloc[i[0]]
            author = books['Author'].iloc[i[0]]
            recommendations.append(f"📖 **{book_title}** by *{author}*")
        
        return recommendations
    except Exception as e:
        return [f"Error: {str(e)}"]

def recommend_by_genre(genre, books, top_n=5):
    """Recommend books based on genre"""
    genre = genre.lower()
    filtered = books[books['Genre'].str.contains(genre, na=False)]
    
    if filtered.empty:
        return [f"❌ No books found in '{genre}'. Try: self-help, finance, fiction, fantasy, history, memoir, business, productivity"]
    
    if len(filtered) > top_n:
        filtered = filtered.sample(min(top_n, len(filtered)))
    else:
        filtered = filtered.head(top_n)
    
    recommendations = []
    for _, row in filtered.iterrows():
        recommendations.append(f"📖 **{row['Book Title']}** by *{row['Author']}*")
    
    return recommendations

def vibe_recommend(text, books, top_n=5):
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
        return [f"❌ No books matching '{text}'. Try different keywords."]
    
    if len(filtered) > top_n:
        filtered = filtered.sample(min(top_n, len(filtered)))
    
    recommendations = []
    for _, row in filtered.iterrows():
        recommendations.append(f"📖 **{row['Book Title']}** by *{row['Author']}*")
    
    return recommendations

# ==================== FAQ FUNCTIONS ====================
FAQ_DATA = [
    {"q": "where is your location", 
     "a": "📍 Dubai Digital Park, Silicon Oasis Building A3, Lower Ground. Open daily 10am–10pm."},
    {"q": "can i sell books", 
     "a": "✅ Yes, you can sell your books with us. You receive credit or cash once sold."},
    {"q": "free delivery", 
     "a": "🚚 Free delivery for orders above AED 180."},
    {"q": "delivery cost", 
     "a": "💰 AED 19 in Dubai/Sharjah/Ajman and AED 24 in other emirates."},
    {"q": "pick up books", 
     "a": "📦 Pickup available with AED 25 charge up to 5kg, AED 2 per extra kg."},
    {"q": "operating hours", 
     "a": "🕐 We are open daily from 10am to 10pm."},
    {"q": "how to use credit", 
     "a": "🎯 Use your credit online or in-store, we deduct it manually."},
    {"q": "can i cancel order", 
     "a": "✅ Yes, please tell us the reason for cancellation."},
    {"q": "return policy", 
     "a": "🔄 Books can be returned within 14 days with original receipt."}
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
            return "🤔 I couldn't find an exact answer. Try asking about: location, delivery, selling books, store hours, or returns."
        
        for item in FAQ_DATA:
            if item["q"] == questions[index]:
                return item["a"]
        
        return "Please contact support for assistance."
    except Exception as e:
        return f"Error: {str(e)}"

# ==================== DASHBOARD FUNCTION ====================
def show_dashboard(books):
    """Display dashboard with visualizations"""
    st.subheader("📊 Genre Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        genre_counts = books['Genre'].value_counts().head(8)
        if not genre_counts.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            genre_counts.plot(kind='barh', ax=ax, color='skyblue')
            ax.set_xlabel("Number of Books")
            ax.set_title("Top Genres")
            st.pyplot(fig)
    
    with col2:
        if not genre_counts.empty:
            fig2, ax2 = plt.subplots(figsize=(8, 8))
            genre_counts.plot(kind='pie', ax=ax2, autopct='%1.1f%%')
            ax2.set_ylabel("")
            st.pyplot(fig2)
    
    # Statistics
    st.subheader("📈 Statistics")
    col3, col4, col5 = st.columns(3)
    with col3:
        st.metric("Total Books", len(books))
    with col4:
        st.metric("Unique Genres", books['Genre'].nunique())
    with col5:
        st.metric("Unique Authors", books['Author'].nunique())
    
    # Top Authors
    st.subheader("Top Authors")
    top_authors = books['Author'].value_counts().head(10)
    if not top_authors.empty:
        st.bar_chart(top_authors)

# ==================== MAIN APP ====================
def main():
    # Page config
    st.set_page_config(
        page_title="Bookends - AI Book Recommendation",
        page_icon="📚",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            text-align: center;
            padding: 2rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .main-header h1 {
            color: white;
            margin: 0;
        }
        .main-header p {
            color: white;
            opacity: 0.9;
        }
        .recommendation-card {
            background-color: #f0f2f6;
            padding: 0.8rem;
            border-radius: 10px;
            margin: 0.5rem 0;
            border-left: 4px solid #667eea;
        }
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.5rem 2rem;
            border-radius: 25px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>📚 Bookends AI Book Recommendation System</h1>
            <p>Discover your next favorite book with AI-powered recommendations</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading books data..."):
        books = load_books_data()
    
    if books is not None and not books.empty:
        # Initialize systems
        _, _, cosine_sim = initialize_recommender(books)
        faq_vectorizer, faq_matrix, faq_questions = initialize_faq_bot()
        
        # Sidebar navigation
        st.sidebar.title("Navigation")
        menu = st.sidebar.radio(
            "Choose a feature:",
            ["📖 Book Recommender", "💬 FAQ Chatbot", "📊 Dashboard", "ℹ️ About"]
        )
        
        # Book Recommender
        if menu == "📖 Book Recommender":
            st.header("📖 Find Your Next Book")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                rec_type = st.selectbox(
                    "Recommend by:",
                    ["Genre", "Book Title", "Vibe/Description"]
                )
            
            with col2:
                if rec_type == "Genre":
                    genre = st.text_input("Enter genre:", placeholder="e.g., self-help, finance, fiction")
                    if st.button("🔍 Find Books", type="primary"):
                        if genre:
                            with st.spinner("Finding recommendations..."):
                                results = recommend_by_genre(genre, books)
                                for r in results:
                                    st.markdown(f'<div class="recommendation-card">{r}</div>', unsafe_allow_html=True)
                        else:
                            st.warning("Please enter a genre.")
                
                elif rec_type == "Book Title":
                    titles = books['Book Title'].tolist()
                    title = st.selectbox("Select a book you like:", titles)
                    if st.button("🔍 Find Similar", type="primary"):
                        with st.spinner("Finding similar books..."):
                            results = recommend_by_title(title, books, cosine_sim)
                            for r in results:
                                st.markdown(f'<div class="recommendation-card">{r}</div>', unsafe_allow_html=True)
                
                else:
                    vibe = st.text_area("Describe what you're looking for:", 
                                       placeholder="e.g., inspiring stories about success")
                    if st.button("🔍 Find Books", type="primary"):
                        if vibe:
                            with st.spinner("Finding matching books..."):
                                results = vibe_recommend(vibe, books)
                                for r in results:
                                    st.markdown(f'<div class="recommendation-card">{r}</div>', unsafe_allow_html=True)
                        else:
                            st.warning("Please describe what you're looking for.")
        
        # FAQ Chatbot
        elif menu == "💬 FAQ Chatbot":
            st.header("💬 Bookends FAQ Assistant")
            st.markdown("Ask me anything about orders, delivery, selling books, or store policies!")
            
            with st.expander("📝 Example questions"):
                st.markdown("""
                - Where is your location?
                - Can I sell books?
                - Do you offer free delivery?
                - What is the delivery cost?
                - What are your operating hours?
                """)
            
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            
            if prompt := st.chat_input("Ask your question..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    response = get_faq_answer(prompt, faq_vectorizer, faq_matrix, faq_questions)
                    st.markdown(response)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
            
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.rerun()
        
        # Dashboard
        elif menu == "📊 Dashboard":
            show_dashboard(books)
        
        # About
        else:
            st.header("ℹ️ About Bookends AI")
            st.markdown("""
            ### Welcome to Bookends AI Book Recommendation System!
            
            **Features:**
            - 📖 **Smart Recommendations**: Based on genre, title, or vibe
            - 💬 **FAQ Chatbot**: Instant answers to common questions
            - 📊 **Interactive Dashboard**: Visual insights about our collection
            
            **How it works:**
            1. We analyze book features using TF-IDF (Text Analysis)
            2. Calculate similarity between books
            3. Recommend the most relevant books to you
            
            **Store Location:**
            Dubai Digital Park, Silicon Oasis Building A3, Lower Ground
            
            **Hours:** Daily 10am–10pm
            """)
        
        # Sidebar info
        st.sidebar.markdown("---")
        st.sidebar.info(f"📚 {len(books)} books available\n\nMade with ❤️ for book lovers")
    
    else:
        st.error("Failed to load data. Please check your installation.")

# ==================== RUN APP ====================
if __name__ == "__main__":
    main()
