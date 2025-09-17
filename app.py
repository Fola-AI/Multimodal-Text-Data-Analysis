# ===== MULTIMODAL TEXT ANALYTICS AI ASSISTANT =====
# This is a comprehensive text analytics system with multiple AI API integrations
# and smart column detection capabilities for customer feedback analysis

# ===== IMPORTS SECTION =====
# Core Python libraries for basic functionality
import os  # Operating system interface for environment variables and file operations
import warnings  # Python warnings control to suppress unnecessary warnings
warnings.filterwarnings('ignore')  # Suppress all warnings to keep output clean

# Environment and API management
from dotenv import load_dotenv  # Load environment variables from .env file for API keys
from anthropic import Anthropic  # Anthropic's Claude AI API client

# Additional AI APIs - using try/except to handle missing dependencies gracefully
try:
    from openai import OpenAI  # OpenAI's GPT API client
except ImportError:
    OpenAI = None  # Set to None if not installed, will be checked later
    
try:
    from groq import Groq  # Groq's fast inference API client
except ImportError:
    Groq = None  # Set to None if not installed
    
try:
    import google.generativeai as genai  # Google's Gemini API client
except ImportError:
    genai = None  # Set to None if not installed

# Data processing and manipulation libraries
import pandas as pd  # Primary data manipulation library for DataFrames
import numpy as np  # Numerical computing library for array operations
from datetime import datetime, timedelta  # Date and time handling utilities
import json  # JSON data format handling
import gc  # Garbage collection for memory management - important for large datasets

# Natural Language Processing libraries
import nltk  # Natural Language Toolkit - comprehensive NLP library
from nltk.corpus import stopwords  # Common words to filter out (the, and, or, etc.)
from nltk.tokenize import word_tokenize  # Split text into individual words/tokens
from nltk.stem import WordNetLemmatizer  # Reduce words to their root form (running -> run)
from textblob import TextBlob  # Simple API for diving into common NLP tasks
import re  # Regular expressions for text pattern matching and cleaning
from collections import Counter  # Efficient counting of hashable objects

# Machine Learning libraries for text analysis
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer  # Convert text to numerical features
from sklearn.decomposition import LatentDirichletAllocation  # Topic modeling algorithm
from sklearn.cluster import KMeans  # Clustering algorithm for grouping similar texts
from sklearn.preprocessing import StandardScaler  # Normalize numerical features
from sklearn.metrics.pairwise import cosine_similarity  # Measure similarity between text vectors

# Visualization libraries for creating charts and graphs
import plotly.express as px  # High-level plotting interface
import plotly.graph_objects as go  # Low-level plotting interface for custom charts
from plotly.subplots import make_subplots  # Create multiple charts in one figure
import matplotlib.pyplot as plt  # Traditional plotting library
import seaborn as sns  # Statistical data visualization built on matplotlib

# Web interface framework
import gradio as gr  # Create web interfaces for machine learning models

# Download required NLTK data packages - these contain language models and corpora
nltk.download('punkt', quiet=True)  # Sentence tokenizer models
nltk.download('punkt_tab', quiet=True)  # New tokenizer format for latest NLTK versions
nltk.download('stopwords', quiet=True)  # Lists of common words to filter out
nltk.download('wordnet', quiet=True)  # Lexical database for lemmatization
nltk.download('averaged_perceptron_tagger', quiet=True)  # Part-of-speech tagger
nltk.download('omw-1.4', quiet=True)  # Open Multilingual Wordnet for lemmatizer
nltk.download('brown', quiet=True)  # Brown corpus required for TextBlob

# Download TextBlob corpora for sentiment analysis
try:
    from textblob import download_corpora  # Import corpora downloader
    download_corpora.main()  # Download all required corpora
except:
    # Alternative method if the above doesn't work - use subprocess
    import subprocess  # Execute shell commands from Python
    import sys  # System-specific parameters and functions
    try:
        # Run TextBlob download command as subprocess with timeout
        subprocess.run([sys.executable, "-m", "textblob.download_corpora"], 
                      capture_output=True, text=True, timeout=30)
    except:
        # If download fails, print warning but continue execution
        print("Warning: Could not download TextBlob corpora. Sentiment analysis may not work properly.")
        print("Please run: python -m textblob.download_corpora")

# Load environment variables from .env file, override existing ones
load_dotenv(override=True)

# ===== SMART COLUMN DETECTOR CLASS =====
class SmartColumnDetector:
    """
    Intelligently detect and extract relevant columns from uploaded data
    This class automatically identifies what type of data each column contains
    """
    
    def __init__(self):
        """Initialize the detector with keyword lists for different column types"""
        # Keywords for detecting text/feedback columns - these usually contain the main content
        self.text_keywords = ['comment', 'feedback', 'review', 'description', 'text', 
                             'response', 'opinion', 'message', 'notes', 'remarks']
        
        # Keywords for detecting ID/identifier columns - these uniquely identify records
        self.id_keywords = ['id', 'identifier', 'key', 'number', 'code', 'ref', 
                           'reference', 'index', 'uuid']
        
        # Keywords for detecting product/category columns - these describe what's being reviewed
        self.product_keywords = ['product', 'item', 'model', 'variant', 'type', 
                                'category', 'brand', 'name', 'sku']
        
        # Keywords for detecting date/time columns - these show when feedback was given
        self.date_keywords = ['date', 'time', 'created', 'updated', 'timestamp']
        
    def detect_column_types(self, df):
        """
        Detect column types based on column names and content analysis
        Returns a dictionary categorizing each column by its likely purpose
        """
        # Initialize results dictionary with empty lists for each category
        detected = {
            'text_columns': [],      # Columns containing feedback/comments
            'id_columns': [],        # Columns containing unique identifiers
            'product_columns': [],   # Columns describing products/categories
            'date_columns': [],      # Columns containing dates/timestamps
            'other_columns': []      # Everything else
        }
        
        # Iterate through each column in the dataframe
        for col in df.columns:
            col_lower = col.lower()  # Convert to lowercase for case-insensitive matching
            
            # Check if column name contains text-related keywords
            if any(keyword in col_lower for keyword in self.text_keywords):
                detected['text_columns'].append(col)
            # Check if column name contains ID-related keywords
            elif any(keyword in col_lower for keyword in self.id_keywords):
                detected['id_columns'].append(col)
            # Check if column name contains product-related keywords
            elif any(keyword in col_lower for keyword in self.product_keywords):
                detected['product_columns'].append(col)
            # Check if column name contains date-related keywords
            elif any(keyword in col_lower for keyword in self.date_keywords):
                detected['date_columns'].append(col)
            else:
                # If no keywords match, analyze the actual content to determine type
                sample = df[col].dropna().head(100)  # Get first 100 non-null values
                if len(sample) > 0:  # If we have sample data
                    # Check if column contains text data (object dtype in pandas)
                    if df[col].dtype == 'object':
                        # Calculate average length of text in this column
                        avg_length = sample.astype(str).str.len().mean()
                        if avg_length > 50:  # Long text likely indicates feedback/comments
                            detected['text_columns'].append(col)
                        elif avg_length < 20 and df[col].nunique() / len(df) > 0.5:
                            # Short, mostly unique values likely indicate IDs
                            detected['id_columns'].append(col)
                        else:
                            # Short, non-unique text likely indicates categories/products
                            detected['product_columns'].append(col)
                    else:
                        # Non-text columns go to 'other' category
                        detected['other_columns'].append(col)
        
        return detected  # Return the categorized column dictionary
    
    def extract_relevant_data(self, df):
        """
        Extract only relevant columns and create optimized dataset for analysis
        This reduces memory usage and focuses on important data
        """
        # First, detect what type each column is
        detected = self.detect_column_types(df)
        
        # Create new dataframe with only relevant columns
        extracted_data = pd.DataFrame()
        
        # Add unique identifier column - use existing ID or create one
        if detected['id_columns'] and len(detected['id_columns']) > 0:
            # Use first detected ID column
            extracted_data['unique_id'] = df[detected['id_columns'][0]]
        else:
            # Create sequential ID numbers if no ID column exists
            extracted_data['unique_id'] = range(1, len(df) + 1)
        
        # Add product information columns (limit to first 2 to avoid too many columns)
        if detected['product_columns'] and len(detected['product_columns']) > 0:
            # Convert to list if needed and limit to 2 product columns
            product_cols = list(detected['product_columns'])[:2]
            for col in product_cols:
                # Add with 'product_' prefix to make purpose clear
                extracted_data[f'product_{col}'] = df[col]
        
        # Combine all text columns into a single 'combined_text' column
        if detected['text_columns'] and len(detected['text_columns']) > 0:
            text_cols = list(detected['text_columns'])  # Ensure it's a list
            text_data = []  # Initialize list to store combined text
            
            # For each row, combine all text columns
            for idx in df.index:
                combined_text = ' '.join([
                    str(df.loc[idx, col])  # Convert to string
                    for col in text_cols   # For each text column
                    if col in df.columns and pd.notna(df.loc[idx, col])  # If column exists and value is not null
                ])
                text_data.append(combined_text)  # Add to our list
            extracted_data['combined_text'] = text_data  # Add as new column
        else:
            # If no text columns detected, create empty combined_text column
            extracted_data['combined_text'] = [''] * len(df)
        
        # Add date column if available (use first detected date column)
        if detected['date_columns'] and len(detected['date_columns']) > 0:
            # Convert to datetime format, handle errors gracefully
            extracted_data['date'] = pd.to_datetime(df[detected['date_columns'][0]], errors='coerce')
        
        # Return both the extracted data and the detection results
        return extracted_data, detected

# ===== ENHANCED TEXT PROCESSOR CLASS =====
class EnhancedTextProcessor:
    """
    Enhanced text preprocessing with actionable insights extraction
    This class handles text cleaning and extracts meaningful patterns from customer feedback
    """

    def __init__(self):
        """Initialize the text processor with NLP tools and insight dictionaries"""
        self.lemmatizer = WordNetLemmatizer()  # Tool to reduce words to root form
        self.stop_words = set(stopwords.words('english'))  # Common words to ignore
        
        # Dictionary mapping actionable items to keywords that indicate them
        # This helps identify what customers want improved
        self.actionable_dictionary = {
            'improve speed': ['slow', 'faster', 'quick', 'speed up', 'takes too long', 'waiting'],
            'better quality': ['poor quality', 'cheap', 'breaks', 'defective', 'flimsy', 'weak'],
            'enhance ui': ['confusing', 'hard to use', 'complicated', 'not intuitive', 'difficult to navigate'],
            'fix bugs': ['bug', 'error', 'crash', 'freeze', 'not working', 'glitch', 'broken'],
            'add features': ['missing', 'need', 'want', 'should have', 'would be nice', 'lacks'],
            'improve support': ['no response', 'unhelpful', 'rude', 'poor service', 'bad support'],
            'better packaging': ['damaged', 'poor packaging', 'arrived broken', 'not protected'],
            'clearer docs': ['unclear', 'no instructions', 'confusing manual', 'hard to understand'],
            'reduce price': ['expensive', 'overpriced', 'too costly', 'not worth', 'cheaper'],
            'faster delivery': ['late', 'delayed', 'slow shipping', 'took forever', 'still waiting'],
            'better communication': ['no updates', 'not informed', 'lack of communication', 'no tracking'],
            'improve reliability': ['unreliable', 'stops working', 'inconsistent', 'sometimes works'],
            'enhance performance': ['slow performance', 'laggy', 'sluggish', 'not responsive'],
            'better design': ['ugly', 'poor design', 'looks cheap', 'not attractive', 'outdated look'],
            'more options': ['limited options', 'no variety', 'need more choices', 'only one option']
        }

    def clean_text(self, text):
        """
        Clean and normalize text for analysis
        Removes special characters and standardizes format
        """
        # Handle null or empty text
        if pd.isna(text) or text == '':
            return ""

        text = str(text).lower()  # Convert to lowercase string
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters, keep only letters, numbers, spaces
        text = ' '.join(text.split())  # Remove extra whitespace
        return text

    def extract_actionable_insights(self, text):
        """
        Extract actionable insights using dictionary matching
        Returns comma-separated list of suggested improvements
        """
        # Handle null or empty text
        if pd.isna(text) or text == '':
            return ""
        
        text_lower = text.lower()  # Convert to lowercase for matching
        found_insights = []  # List to store found actionable items
        
        # Check each actionable item against the text
        for action, keywords in self.actionable_dictionary.items():
            for keyword in keywords:
                if keyword in text_lower:  # If keyword found in text
                    found_insights.append(action)  # Add the actionable item
                    break  # Only add each action once per text
        
        # Return top 3 most relevant insights to avoid overwhelming output
        if found_insights:
            return ', '.join(found_insights[:3])
        return ""

    def extract_specific_topics(self, text):
        """
        Extract specific topics from text using keyword extraction and noun phrase detection
        Returns list of 3 topics (may include empty strings if not enough topics found)
        """
        # Handle null, empty, or very short text
        if pd.isna(text) or text == '' or len(text) < 10:
            return ['', '', '']  # Return 3 empty strings
        
        text_lower = text.lower()  # Convert to lowercase
        
        # Remove stopwords for better topic extraction
        words = word_tokenize(text_lower)  # Split into individual words
        # Filter out stopwords and very short words
        filtered_words = [w for w in words if w not in self.stop_words and len(w) > 3]
        
        # Extract noun phrases using TextBlob (these are usually good topics)
        blob = TextBlob(text)
        noun_phrases = blob.noun_phrases  # Get noun phrases from text
        
        topics = []  # Initialize topics list
        
        # Add noun phrases (these are usually good topics)
        for phrase in noun_phrases[:5]:  # Limit to top 5 noun phrases
            if len(phrase.split()) <= 3:  # Only include short phrases (3 words or less)
                topics.append(phrase)
        
        # Add frequent meaningful words if we don't have enough topics
        if len(topics) < 3:
            word_freq = Counter(filtered_words)  # Count word frequencies
            for word, _ in word_freq.most_common(5):  # Get top 5 most common words
                if word not in str(topics):  # Avoid duplicates
                    topics.append(word)
                if len(topics) >= 3:  # Stop when we have 3 topics
                    break
        
        # Ensure we always return exactly 3 items
        topics = topics[:3]  # Take only first 3
        while len(topics) < 3:  # Add empty strings if needed
            topics.append('')
        
        return topics

    def determine_topic(self, text):
        """
        Legacy method kept for compatibility - returns first specific topic
        This maintains backward compatibility with older versions
        """
        topics = self.extract_specific_topics(text)  # Get all topics
        return topics[0] if topics[0] else 'General'  # Return first topic or 'General'

# ===== SEARCH ENGINE CLASS =====
class TextSearchEngine:
    """
    Advanced search functionality for text data with semantic capabilities
    Uses TF-IDF vectorization and cosine similarity for intelligent text search
    """
    
    def __init__(self):
        """Initialize the search engine with TF-IDF vectorizer and synonym dictionary"""
        # TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer
        # Converts text to numerical vectors for similarity calculations
        self.vectorizer = TfidfVectorizer(
            max_features=1000,        # Limit to top 1000 most important terms
            ngram_range=(1, 3),       # Include unigrams, bigrams, and trigrams for better matching
            stop_words='english',     # Remove common English words
            use_idf=True,            # Use inverse document frequency weighting
            smooth_idf=True,         # Add smoothing to IDF
            sublinear_tf=True        # Apply sublinear tf scaling for better performance
        )
        self.tfidf_matrix = None     # Will store the TF-IDF matrix after building index
        self.data = None             # Will store the original data
        
        # Synonym dictionary for semantic search - helps find related terms
        self.synonyms = {
            'fast': ['quick', 'rapid', 'speedy', 'swift', 'prompt'],
            'slow': ['sluggish', 'delayed', 'laggy', 'lengthy', 'prolonged'],
            'good': ['excellent', 'great', 'wonderful', 'fantastic', 'amazing', 'positive'],
            'bad': ['poor', 'terrible', 'awful', 'negative', 'horrible', 'disappointing'],
            'problem': ['issue', 'bug', 'error', 'defect', 'fault', 'glitch'],
            'help': ['support', 'assistance', 'aid', 'service'],
            'price': ['cost', 'fee', 'charge', 'rate', 'payment', 'expensive', 'cheap'],
            'quality': ['standard', 'grade', 'condition', 'caliber'],
            'delivery': ['shipping', 'dispatch', 'arrival', 'transport'],
            'easy': ['simple', 'straightforward', 'effortless', 'user-friendly'],
            'hard': ['difficult', 'complex', 'complicated', 'challenging'],
            'broken': ['damaged', 'defective', 'faulty', 'malfunctioning'],
            'love': ['like', 'enjoy', 'appreciate', 'adore'],
            'hate': ['dislike', 'despise', 'detest'],
            'feature': ['function', 'capability', 'option', 'characteristic'],
            'customer': ['client', 'buyer', 'purchaser', 'consumer', 'user']
        }
        
    def expand_query_with_synonyms(self, query):
        """
        Expand search query with synonyms for better semantic matching
        This helps find relevant results even when different words are used
        """
        query_words = query.lower().split()  # Split query into individual words
        expanded_terms = []  # List to store original words and synonyms
        
        for word in query_words:
            expanded_terms.append(word)  # Add the original word
            
            # Add synonyms if available for this word
            if word in self.synonyms:
                expanded_terms.extend(self.synonyms[word])
            
            # Check if word is a synonym of something else and add related terms
            for key, syns in self.synonyms.items():
                if word in syns:  # If current word is a synonym
                    expanded_terms.append(key)  # Add the main term
                    expanded_terms.extend([s for s in syns if s != word])  # Add other synonyms
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in expanded_terms:
            if term not in seen:
                unique_terms.append(term)
                seen.add(term)
        
        return ' '.join(unique_terms)  # Return expanded query as single string
        
    def build_index(self, df, text_column):
        """
        Build search index from text data
        Creates TF-IDF vectors for all documents to enable fast similarity search
        """
        self.data = df.copy()  # Store copy of the data
        texts = df[text_column].fillna('').tolist()  # Get all text, fill nulls with empty string
        
        # Add other searchable columns to improve search accuracy
        if 'topic_1' in df.columns:
            # Combine main text with topic information for better searchability
            texts = [f"{text} {df.iloc[i]['topic_1']} {df.iloc[i]['topic_2']} {df.iloc[i]['topic_3']}" 
                    for i, text in enumerate(texts)]
        if 'actionable_insights' in df.columns:
            # Also include actionable insights in searchable text
            texts = [f"{texts[i]} {df.iloc[i]['actionable_insights']}" 
                    for i in range(len(texts))]
            
        # Create TF-IDF matrix from all texts
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        
    def search(self, query, top_k=10):
        """
        Enhanced search with semantic understanding
        Returns top matching documents with similarity scores
        """
        # Check if index has been built
        if self.tfidf_matrix is None:
            return pd.DataFrame()  # Return empty DataFrame if no index
        
        # Expand query with synonyms for better semantic matching
        expanded_query = self.expand_query_with_synonyms(query)
        
        # Vectorize both original and expanded queries
        query_vector = self.vectorizer.transform([query])  # Original query vector
        expanded_vector = self.vectorizer.transform([expanded_query])  # Expanded query vector
        
        # Calculate similarities for both queries against all documents
        similarities_orig = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        similarities_exp = cosine_similarity(expanded_vector, self.tfidf_matrix).flatten()
        
        # Combine scores (weighted average - original query gets more weight)
        combined_similarities = (0.7 * similarities_orig + 0.3 * similarities_exp)
        
        # Get top results
        top_indices = combined_similarities.argsort()[-top_k:][::-1]  # Get indices of top scores, reverse order
        top_scores = combined_similarities[top_indices]  # Get the actual scores
        
        # Filter results with score > 0.05 (lower threshold for better recall)
        valid_indices = [idx for idx, score in zip(top_indices, top_scores) if score > 0.05]
        
        if valid_indices:
            # Create results dataframe from valid matches
            results = self.data.iloc[valid_indices].copy()
            results['search_score'] = [combined_similarities[idx] for idx in valid_indices]
            
            # Boost results that have exact matches in the text
            query_lower = query.lower()
            for idx in results.index:
                if 'combined_text' in results.columns:
                    # If exact query appears in text, boost the score
                    if query_lower in str(results.at[idx, 'combined_text']).lower():
                        results.at[idx, 'search_score'] *= 1.5  # 50% boost for exact matches
                        
            return results.sort_values('search_score', ascending=False)  # Return sorted by relevance
        
        return pd.DataFrame()  # Return empty DataFrame if no valid results

# ===== AI MODEL MANAGER CLASS =====
class AIModelManager:
    """
    Manages multiple AI model APIs and provides unified interface
    Supports OpenAI, Anthropic, Deepseek, Groq, and Google Gemini
    """
    
    def __init__(self):
        """Initialize the model manager and set up all available AI APIs"""
        self.available_models = {}  # Dictionary to store available models
        self.clients = {}          # Dictionary to store API clients
        self.current_model = None  # Currently selected model
        self.initialize_apis()     # Set up all APIs
        
    def initialize_apis(self):
        """Initialize all available AI APIs based on environment variables"""
        
        # Anthropic Claude API setup
        ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")  # Get API key from environment
        if ANTHROPIC_API_KEY:  # If API key exists
            try:
                self.clients['anthropic'] = Anthropic(api_key=ANTHROPIC_API_KEY)  # Create client
                # Add Claude model to available models
                self.available_models['Claude 3 Haiku'] = {
                    'provider': 'anthropic',
                    'model': 'claude-3-haiku-20240307'
                }
                print(f"Anthropic API Key exists and begins {ANTHROPIC_API_KEY[:4]}")  # Confirm setup
            except Exception as e:
                print(f"Error initializing Anthropic: {e}")
        else:
            print("Anthropic API Key not set")
            
        # OpenAI API setup
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if OPENAI_API_KEY and OpenAI:  # Check both API key and library availability
            try:
                self.clients['openai'] = OpenAI(api_key=OPENAI_API_KEY)
                # Add multiple OpenAI models
                self.available_models['GPT-4o-mini'] = {
                    'provider': 'openai',
                    'model': 'gpt-4o-mini'
                }
                self.available_models['GPT-3.5 Turbo'] = {
                    'provider': 'openai',
                    'model': 'gpt-3.5-turbo'
                }
                print(f"OpenAI API Key exists and begins {OPENAI_API_KEY[:7]}")
            except Exception as e:
                print(f"Error initializing OpenAI: {e}")
        else:
            print("OpenAI API Key not set or library not installed")
            
        # Deepseek API setup (uses OpenAI-compatible API)
        DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
        if DEEPSEEK_API_KEY and OpenAI:
            try:
                # Deepseek uses OpenAI client with different base URL
                self.clients['deepseek'] = OpenAI(
                    api_key=DEEPSEEK_API_KEY,
                    base_url="https://api.deepseek.com"  # Deepseek's API endpoint
                )
                self.available_models['Deepseek Chat'] = {
                    'provider': 'deepseek',
                    'model': 'deepseek-chat'
                }
                print(f"Deepseek API Key exists and begins {DEEPSEEK_API_KEY[:7]}")
            except Exception as e:
                print(f"Error initializing Deepseek: {e}")
        else:
            print("Deepseek API Key not set or OpenAI library not installed")
            
        # Groq API setup
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        if GROQ_API_KEY and Groq:
            try:
                self.clients['groq'] = Groq(api_key=GROQ_API_KEY)
                # Add multiple Groq models
                self.available_models['Llama 3.3 70B'] = {
                    'provider': 'groq',
                    'model': 'llama-3.3-70b-versatile'
                }
                self.available_models['Mixtral 8x7B'] = {
                    'provider': 'groq',
                    'model': 'mixtral-8x7b-32768'
                }
                print(f"Groq API Key exists and begins {GROQ_API_KEY[:4]}")
            except Exception as e:
                print(f"Error initializing Groq: {e}")
        else:
            print("Groq API Key not set or library not installed")
            
        # Google Gemini API setup
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        if GOOGLE_API_KEY and genai:
            try:
                genai.configure(api_key=GOOGLE_API_KEY)  # Configure Google AI
                self.clients['google'] = genai  # Store the configured module
                # Add Google models
                self.available_models['Gemini 1.5 Flash'] = {
                    'provider': 'google',
                    'model': 'gemini-1.5-flash'
                }
                self.available_models['Gemini 1.5 Pro'] = {
                    'provider': 'google',
                    'model': 'gemini-1.5-pro'
                }
                print(f"Google API Key exists and begins {GOOGLE_API_KEY[:2]}")
            except Exception as e:
                print(f"Error initializing Google Gemini: {e}")
        else:
            print("Google API Key not set or library not installed")
            
        # Set default model to first available model
        if self.available_models:
            self.current_model = list(self.available_models.keys())[0]
            
    def get_available_models(self):
        """Return list of available model names"""
        return list(self.available_models.keys())
    
    def set_model(self, model_name):
        """Set the current model for text generation"""
        if model_name in self.available_models:
            self.current_model = model_name
            return True  # Success
        return False  # Model not available
    
    def generate_text(self, prompt, max_tokens=1000):
        """
        Generate text using the current model
        Handles different API formats for each provider
        """
        # Check if we have a valid current model
        if not self.current_model or self.current_model not in self.available_models:
            return None
            
        model_info = self.available_models[self.current_model]  # Get model configuration
        provider = model_info['provider']  # Which API provider to use
        model = model_info['model']        # Specific model name
        
        try:
            # Handle Anthropic API format
            if provider == 'anthropic':
                client = self.clients['anthropic']
                response = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text  # Extract text from response
                
            # Handle OpenAI and Deepseek API format (both use OpenAI-compatible format)
            elif provider in ['openai', 'deepseek']:
                client = self.clients[provider]
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content  # Extract text from response
                
            # Handle Groq API format (similar to OpenAI)
            elif provider == 'groq':
                client = self.clients['groq']
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
                
            # Handle Google Gemini API format
            elif provider == 'google':
                model_obj = genai.GenerativeModel(model)  # Create model object
                response = model_obj.generate_content(prompt)  # Generate response
                return response.text  # Extract text
                
        except Exception as e:
            print(f"Error generating text with {self.current_model}: {e}")
            return None

# Initialize the model manager globally so it can be used throughout the application
model_manager = AIModelManager()

# ===== ENHANCED TEXT ANALYZER CLASS =====
class EnhancedTextAnalyzer:
    """
    Main analysis engine with all enhanced features and multi-model support
    This is the core class that orchestrates all text analysis functionality
    """
    
    def __init__(self, model_manager=None):
        """Initialize the analyzer with all component classes"""
        self.model_manager = model_manager                              # AI model manager for generating insights
        self.column_detector = SmartColumnDetector()                   # Smart column detection
        self.text_processor = EnhancedTextProcessor()                  # Text processing and insights
        self.search_engine = TextSearchEngine()                       # Text search functionality
        self.original_df = None                                        # Store original data
        self.processed_df = None                                       # Store processed data
        self.results = {}                                              # Store analysis results
        self.visualizations = {}                                       # Store generated visualizations
        
    def load_file(self, file):
        """
        Load data from various file formats (CSV, Excel, JSON)
        Returns the loaded dataframe and a status message
        """
        try:
            # Determine file type based on extension and load accordingly
            if file.name.endswith('.csv'):
                df = pd.read_csv(file.name)  # Load CSV file
            elif file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file.name)  # Load Excel file
            elif file.name.endswith('.json'):
                df = pd.read_json(file.name)  # Load JSON file
            else:
                return None, "Unsupported file format"  # Return error for unsupported formats
            
            return df, f"File loaded: {len(df)} records"  # Return success message with record count
        except Exception as e:
            return None, f"Error loading file: {str(e)}"  # Return error message
    
    def process_data(self, df):
        """
        Process data with smart extraction and analysis
        This is the main processing pipeline that analyzes the uploaded data
        """
        # Step 1: Extract relevant columns using smart detection
        extracted_df, detected_columns = self.column_detector.extract_relevant_data(df)
        
        # Step 2: Store processed data for later use
        self.processed_df = extracted_df
        
        # Step 3: Clean up memory by deleting original large dataframe
        del df
        gc.collect()  # Force garbage collection to free memory
        
        # Step 4: Add analysis columns if we have text data to analyze
        if 'combined_text' in extracted_df.columns:
            # Initialize lists to store analysis results for each row
            sentiments = []      # Positive/Negative/Neutral sentiment classification
            polarities = []      # Numerical sentiment scores (-1 to 1)
            topics_1 = []        # Primary topic for each text
            topics_2 = []        # Secondary topic for each text
            topics_3 = []        # Tertiary topic for each text
            insights = []        # Actionable insights for each text
            
            # Process each text entry
            for text in extracted_df['combined_text']:
                # Sentiment analysis using TextBlob
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity  # Get numerical sentiment score
                
                # Convert numerical score to categorical sentiment
                if polarity > 0.1:      # Positive threshold
                    sentiment = 'Positive'
                elif polarity < -0.1:   # Negative threshold
                    sentiment = 'Negative'
                else:                   # Neutral range
                    sentiment = 'Neutral'
                
                sentiments.append(sentiment)    # Add categorical sentiment
                polarities.append(polarity)     # Add numerical score
                
                # Extract specific topics (3 separate topics per text)
                specific_topics = self.text_processor.extract_specific_topics(text)
                topics_1.append(specific_topics[0])  # Primary topic
                topics_2.append(specific_topics[1])  # Secondary topic
                topics_3.append(specific_topics[2])  # Tertiary topic
                
                # Extract actionable insights using dictionary matching
                insight = self.text_processor.extract_actionable_insights(text)
                insights.append(insight)
            
            # Add all analysis results as new columns to the dataframe
            extracted_df['sentiment'] = sentiments           # Categorical sentiment
            extracted_df['sentiment_score'] = polarities     # Numerical sentiment score
            extracted_df['topic_1'] = topics_1              # Primary topic
            extracted_df['topic_2'] = topics_2              # Secondary topic
            extracted_df['topic_3'] = topics_3              # Tertiary topic
            extracted_df['actionable_insights'] = insights   # Actionable insights
            
            # Build search index with enhanced search capabilities
            self.search_engine.build_index(extracted_df, 'combined_text')
        
        # Step 5: Save processed data to Excel file for download
        output_file = 'processed_data.xlsx'
        extracted_df.to_excel(output_file, index=False)
        
        # Return processed data, detected column info, and output file path
        return extracted_df, detected_columns, output_file
    
    def generate_ai_insights(self, df, num_samples=5):
        """
        Generate AI-powered insights using selected model
        Takes sample texts and generates high-level insights using AI
        """
        # Check if AI model is available
        if not self.model_manager or not self.model_manager.current_model:
            return "No AI model available for generating insights"
        
        # Check if we have text data to analyze
        if 'combined_text' not in df.columns or df.empty:
            return "No text data available for AI analysis"
        
        # Sample some texts for analysis (to avoid sending too much data to AI)
        sample_texts = df['combined_text'].dropna().head(num_samples).tolist()
        if not sample_texts:
            return "No valid text samples found"
        
        # Create prompt for AI analysis
        # This prompt asks the AI to analyze the customer feedback samples
        prompt = f"""Analyze the following customer feedback samples and provide key insights:

Samples:
{chr(10).join([f"{i+1}. {text[:200]}..." if len(text) > 200 else f"{i+1}. {text}" for i, text in enumerate(sample_texts)])}

Please provide:
1. Main themes and patterns
2. Key sentiment indicators
3. Actionable recommendations
4. Areas of concern

Keep the response concise and focused on actionable insights."""

        # Generate insights using selected model
        try:
            response = self.model_manager.generate_text(prompt, max_tokens=500)
            if response:
                return f"**AI Insights (using {self.model_manager.current_model}):**\n\n{response}"
            else:
                return "Failed to generate AI insights. Please check your API configuration."
        except Exception as e:
            return f"Error generating AI insights: {str(e)}"
    
    def generate_visualizations(self, df):
        """
        Generate various visualizations from the analyzed data
        Creates interactive charts using Plotly for better user experience
        """
        visualizations = {}  # Dictionary to store all visualizations
        
        # Generate sentiment distribution pie chart
        if 'sentiment' in df.columns:
            sentiment_counts = df['sentiment'].value_counts()  # Count each sentiment category
            fig_sentiment = px.pie(
                values=sentiment_counts.values,     # Values for pie slices
                names=sentiment_counts.index,       # Labels for pie slices
                title="Sentiment Distribution",     # Chart title
                color_discrete_map={                # Custom colors for each sentiment
                    'Positive': '#27AE60',          # Green for positive
                    'Negative': '#E74C3C',          # Red for negative
                    'Neutral': '#95A5A6'            # Gray for neutral
                }
            )
            visualizations['Sentiment Distribution'] = fig_sentiment
            
        # Generate topic distribution bar chart
        if 'topic_1' in df.columns:
            # Combine all topics from all three topic columns
            all_topics = []
            for col in ['topic_1', 'topic_2', 'topic_3']:
                if col in df.columns:
                    topics = df[col].dropna().tolist()         # Get non-null topics
                    all_topics.extend([t for t in topics if t != ''])  # Add non-empty topics
            
            if all_topics:
                topic_counts = Counter(all_topics)                    # Count topic frequencies
                top_topics = dict(topic_counts.most_common(15))       # Get top 15 topics
                
                fig_topics = px.bar(
                    x=list(top_topics.values()),                     # Frequency values
                    y=list(top_topics.keys()),                       # Topic names
                    orientation='h',                                  # Horizontal bar chart
                    title="Top 15 Specific Topics",                  # Chart title
                    labels={'x': 'Count', 'y': 'Topic'}             # Axis labels
                )
                visualizations['Topic Distribution'] = fig_topics
            
        # Generate sentiment by topic heatmap
        if 'sentiment' in df.columns and 'topic_1' in df.columns:
            df_temp = df[df['topic_1'] != ''].copy()              # Filter out empty topics
            if not df_temp.empty:
                # Get top 10 topics for cleaner visualization
                top_topics = df_temp['topic_1'].value_counts().head(10).index
                df_filtered = df_temp[df_temp['topic_1'].isin(top_topics)]
                
                # Create cross-tabulation of topics vs sentiments
                pivot_table = pd.crosstab(df_filtered['topic_1'], df_filtered['sentiment'])
                fig_heatmap = px.imshow(
                    pivot_table,                                      # Data for heatmap
                    labels=dict(x="Sentiment", y="Primary Topic", color="Count"),  # Labels
                    title="Sentiment by Primary Topic Heatmap",      # Title
                    color_continuous_scale="RdYlGn"                  # Color scale (red to green)
                )
                visualizations['Sentiment by Topic'] = fig_heatmap
            
        # Generate sentiment timeline if date data is available
        if 'date' in df.columns and 'sentiment' in df.columns:
            df_time = df.copy()
            df_time['date'] = pd.to_datetime(df_time['date'])         # Ensure date format
            # Group by month and sentiment to show trends over time
            time_data = df_time.groupby([pd.Grouper(key='date', freq='M'), 'sentiment']).size().reset_index(name='count')
            
            fig_timeline = px.line(
                time_data,
                x='date',                                             # X-axis: time
                y='count',                                            # Y-axis: count
                color='sentiment',                                    # Different lines for each sentiment
                title="Sentiment Trends Over Time",                  # Chart title
                color_discrete_map={                                 # Custom colors
                    'Positive': '#27AE60',
                    'Negative': '#E74C3C',
                    'Neutral': '#95A5A6'
                }
            )
            visualizations['Sentiment Timeline'] = fig_timeline
        
        # Generate actionable insights bar chart
        if 'actionable_insights' in df.columns:
            all_insights = []  # List to store all individual insights
            for insight in df['actionable_insights']:
                if insight and insight != "":
                    # Split by comma as we're now using comma-separated insights
                    all_insights.extend([i.strip() for i in insight.split(',')])
            
            if all_insights:
                insight_counts = Counter(all_insights)                 # Count insight frequencies
                top_insights = dict(insight_counts.most_common(10))    # Get top 10 insights
                
                fig_insights = px.bar(
                    x=list(top_insights.values()),                    # Frequency values
                    y=list(top_insights.keys()),                      # Insight names
                    orientation='h',                                   # Horizontal bar chart
                    title="Top 10 Actionable Insights",               # Chart title
                    labels={'x': 'Frequency', 'y': 'Insight'}        # Axis labels
                )
                visualizations['Top Insights'] = fig_insights
        
        return visualizations  # Return dictionary of all generated visualizations

# ===== GRADIO INTERFACE FUNCTIONS =====
# Global variables to maintain state across function calls
analyzer = None                # Main analyzer instance
current_data = None           # Currently processed data
current_visualizations = None # Currently generated visualizations

def update_model(model_name):
    """Update the selected AI model"""
    global model_manager
    
    if model_manager.set_model(model_name):  # Try to set the new model
        return f"✅ Model switched to: {model_name}"
    else:
        return f"❌ Failed to switch to: {model_name}"

def process_file(file, model_name):
    """
    Process uploaded file with selected model
    This is the main function called when user uploads a file
    """
    global analyzer, current_data, current_visualizations, model_manager
    
    # Check if file was uploaded
    if file is None:
        return "Please upload a file", None, None, None, None, None, gr.update(choices=[])
    
    try:
        # Update model if changed
        if model_name and model_manager:
            model_manager.set_model(model_name)
        
        # Create new analyzer instance
        analyzer = EnhancedTextAnalyzer(model_manager)
        
        # Load the uploaded file
        df, message = analyzer.load_file(file)
        if df is None:  # If file loading failed
            return message, None, None, None, None, None, gr.update(choices=[])
        
        # Process the loaded data
        processed_df, detected_cols, output_file = analyzer.process_data(df)
        current_data = processed_df  # Store for later use
        
        # Generate visualizations from processed data
        visualizations = analyzer.generate_visualizations(processed_df)
        current_visualizations = visualizations  # Store for later use
        
        # Generate AI insights using the selected model
        ai_insights = analyzer.generate_ai_insights(processed_df)
        
        # Create summary of processing results
        # Safely handle detected columns (convert to lists and limit length)
        text_cols = list(detected_cols.get('text_columns', []))[:3] if detected_cols.get('text_columns') else []
        id_cols = list(detected_cols.get('id_columns', []))[:3] if detected_cols.get('id_columns') else []
        product_cols = list(detected_cols.get('product_columns', []))[:3] if detected_cols.get('product_columns') else []
        
        summary = f"""
        ### ✅ File Processing Complete!
        
        **Detected Columns:**
        - Text Columns: {', '.join(text_cols) if text_cols else 'None'}
        - ID Columns: {', '.join(id_cols) if id_cols else 'Auto-generated'}
        - Product Columns: {', '.join(product_cols) if product_cols else 'None'}
        
        **Analysis Results:**
        - Total Records: {len(processed_df)}
        - Processed File Saved: {output_file}
        - AI Model Used: {model_manager.current_model if model_manager else 'None'}
        """
        
        # Create data preview (first 10 rows for display)
        preview = processed_df.head(10)
        
        # Get first visualization for immediate display
        first_viz = list(visualizations.values())[0] if visualizations else None
        
        # Return all results for the Gradio interface
        return (
            summary,                                          # Processing status
            preview,                                          # Data preview
            output_file,                                      # Downloadable processed file
            ai_insights,                                      # AI-generated insights
            first_viz,                                        # First visualization
            "Ready for search",                               # Search status
            gr.update(choices=list(visualizations.keys()))    # Update visualization dropdown
        )
        
    except Exception as e:
        # Return error message if anything goes wrong
        return f"Error: {str(e)}", None, None, None, None, None, gr.update(choices=[])

def search_data(query):
    """
    Search through the data with enhanced semantic search
    Uses the built search engine to find relevant text entries
    """
    global analyzer, current_data
    
    # Check if data has been processed
    if analyzer is None or current_data is None:
        return "Please process a file first", None, None
    
    # Check if search query was provided
    if not query:
        return "Please enter a search query", None, None
    
    try:
        # Perform the search using the search engine
        results = analyzer.search_engine.search(query, top_k=10)
        
        # Check if any results were found
        if results.empty:
            return "No results found", None, None
        
        # Select relevant columns for display (updated to include new topic columns)
        display_cols = ['unique_id', 'combined_text', 'sentiment', 'topic_1', 'topic_2', 'topic_3', 'actionable_insights', 'search_score']
        display_cols = [col for col in display_cols if col in results.columns]  # Only include existing columns
        
        results_display = results[display_cols]  # Create display dataframe
        
        # Save search results to file for download
        search_output = f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        results_display.to_excel(search_output, index=False)
        
        # Return search results and status
        return f"Found {len(results)} results", results_display.head(10), search_output
        
    except Exception as e:
        return f"Search error: {str(e)}", None, None

def update_visualization(viz_type):
    """
    Update displayed visualization based on user selection
    Called when user selects a different visualization from dropdown
    """
    global current_visualizations
    
    # Check if visualization exists and return it
    if current_visualizations and viz_type in current_visualizations:
        return current_visualizations[viz_type]
    return None  # Return None if visualization not found

def export_results(format_type):
    """
    Export processed data in different formats (Excel or CSV)
    Allows users to download their analyzed data
    """
    global current_data
    
    # Check if there's data to export
    if current_data is None:
        return "No data to export", None
    
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # Create timestamp for unique filename
        
        # Export based on selected format
        if format_type == "Excel":
            output_file = f"analysis_results_{timestamp}.xlsx"
            current_data.to_excel(output_file, index=False)  # Save as Excel
        else:  # CSV
            output_file = f"analysis_results_{timestamp}.csv"
            current_data.to_csv(output_file, index=False)    # Save as CSV
        
        return f"Data exported to {output_file}", output_file
    
    except Exception as e:
        return f"Export error: {str(e)}", None

# ===== GRADIO INTERFACE CREATION =====
def create_interface():
    """
    Create the Gradio interface with model selection
    This function builds the entire web interface using Gradio
    """
    
    # Create the main Gradio application with soft theme
    with gr.Blocks(theme=gr.themes.Soft()) as app:
        # Main title and description
        gr.Markdown(
            """
            # 📊 Enhanced Text Analytics AI Agent
            ### Smart Column Detection & Comprehensive Text Analysis with Multiple AI Models
            
            **Features:**
            - 🤖 Multiple AI Model Support (OpenAI, Anthropic, Deepseek, Groq, Google)
            - 🔍 Automatic detection of text, ID, and product columns
            - 💾 Memory-efficient processing with automatic file cleanup
            - 😊 Sentiment analysis with scoring
            - 🎯 Topic/theme extraction
            - 💡 Actionable insights generation
            - 🔎 Advanced text search with similarity scoring
            - 📈 Multiple visualization options
            - 📥 Export results in Excel or CSV format
            """
        )
        
        # Tab 1: Upload & Process
        with gr.Tab("📤 Upload & Process"):
            with gr.Row():
                with gr.Column(scale=1):  # Left column for controls
                    # Model selection dropdown
                    model_dropdown = gr.Dropdown(
                        label="🤖 Select AI Model",
                        choices=model_manager.get_available_models(),  # Get available models
                        value=model_manager.current_model if model_manager.current_model else None,
                        interactive=True
                    )
                    
                    # File upload component
                    file_upload = gr.File(
                        label="Upload Data File",
                        file_types=[".csv", ".xlsx", ".xls", ".json"]  # Supported file types
                    )
                    
                    # Process button
                    process_btn = gr.Button("🚀 Process File", variant="primary")
                
                with gr.Column(scale=2):  # Right column for results
                    status_output = gr.Markdown(label="Processing Status")      # Processing status display
                    ai_insights = gr.Markdown(label="AI-Generated Insights")   # AI insights display
            
            # Data preview section
            with gr.Row():
                data_preview = gr.Dataframe(
                    label="Data Preview (First 10 rows)",
                    interactive=False  # Read-only display
                )
            
            # Processed file download
            processed_file = gr.File(
                label="📁 Processed Data File",
                interactive=False  # Read-only, for download only
            )
        
        # Tab 2: Search
        with gr.Tab("🔍 Search"):
            gr.Markdown("### Search through your text data")
            
            with gr.Row():
                # Search input box
                search_input = gr.Textbox(
                    label="Enter search query",
                    placeholder="Type keywords to search..."
                )
                # Search button
                search_btn = gr.Button("🔎 Search", variant="primary")
            
            # Search results display
            search_status = gr.Markdown(label="Search Status")       # Search status
            search_results = gr.Dataframe(                          # Search results table
                label="Search Results",
                interactive=False
            )
            search_file = gr.File(                                   # Download search results
                label="📥 Download Search Results",
                interactive=False
            )
        
        # Tab 3: Visualizations
        with gr.Tab("📈 Visualizations"):
            with gr.Row():
                # Visualization selector dropdown
                viz_selector = gr.Dropdown(
                    label="Select Visualization",
                    choices=[],          # Will be populated after processing
                    interactive=True
                )
            
            # Visualization display area
            viz_plot = gr.Plot(label="Visualization")
        
        # Tab 4: Export
        with gr.Tab("📥 Export"):
            gr.Markdown("### Export your analyzed data")
            
            with gr.Row():
                # Export format selection
                export_format = gr.Radio(
                    choices=["Excel", "CSV"],
                    value="Excel",
                    label="Export Format"
                )
                # Export button
                export_btn = gr.Button("📥 Export Data", variant="primary")
            
            # Export results display
            export_status = gr.Markdown(label="Export Status")      # Export status
            export_file = gr.File(                                  # Download exported file
                label="📁 Download Exported File",
                interactive=False
            )
        
        # ===== EVENT HANDLERS =====
        # These connect user interactions to the backend functions
        
        # Model selection change handler
        model_dropdown.change(
            fn=update_model,                # Function to call
            inputs=[model_dropdown],        # Input components
            outputs=[status_output]         # Output components
        )
        
        # File processing button click handler
        process_btn.click(
            fn=process_file,               # Function to call
            inputs=[file_upload, model_dropdown],  # Input components
            outputs=[                      # Output components
                status_output,
                data_preview,
                processed_file,
                ai_insights,
                viz_plot,
                search_status,
                viz_selector
            ]
        )
        
        # Search button click handler
        search_btn.click(
            fn=search_data,                # Function to call
            inputs=[search_input],         # Input components
            outputs=[search_status, search_results, search_file]  # Output components
        )
        
        # Visualization selector change handler
        viz_selector.change(
            fn=update_visualization,       # Function to call
            inputs=[viz_selector],         # Input components
            outputs=[viz_plot]             # Output components
        )
        
        # Export button click handler
        export_btn.click(
            fn=export_results,             # Function to call
            inputs=[export_format],        # Input components
            outputs=[export_status, export_file]  # Output components
        )
    
    return app  # Return the complete Gradio application

# ===== APPLICATION LAUNCH =====
# Launch the application when script is run directly
if __name__ == "__main__":
    app = create_interface()                    # Create the Gradio interface
    app.launch(share=True, debug=True)          # Launch with public sharing and debug mode