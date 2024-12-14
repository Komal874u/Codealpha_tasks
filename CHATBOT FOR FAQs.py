import nltk
import random
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Lemmatizer
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()

# FAQ Data for multiple topics
faq_data = {
    "General": [
        {'question': 'What is your return policy?', 'answer': 'Our return policy lasts 30 days.'},
        {'question': 'How can I track my order?', 'answer': 'You can track your order through our website.'}
    ],
    "Shipping": [
        {'question': 'Do you ship internationally?', 'answer': 'Yes, we ship to over 50 countries worldwide.'},
        {'question': 'What shipping methods do you offer?', 'answer': 'We offer standard and express shipping.'}
    ],
    "Payment": [
        {'question': 'What payment methods do you accept?', 'answer': 'We accept credit/debit cards and PayPal.'},
        {'question': 'Can I pay with cryptocurrency?', 'answer': 'Currently, we do not accept cryptocurrency payments.'}
    ],
    "Python":[
        {"question":"What is the python?", "answer":"Python is the high level programming language."},
        {"question":"Whta are python key features?","answer":"python is easy to learn,interpreted, and dynamically typed."}
        
    ],
     "Flask": [
        {"question": "What is Flask?", "answer": "Flask is a micro web framework written in Python."},
        {"question": "How to install Flask?", "answer": "You can install Flask using pip: pip install flask."}
    ],
    "Machine Learning": [
        {"question": "What is Machine Learning?", "answer": "Machine Learning is a branch of artificial intelligence that focuses on building systems that learn from data."},
        {"question": "What are types of machine learning?", "answer": "The main types are supervised learning, unsupervised learning, and reinforcement learning."}
    ]
}


# Preprocessing Function (Tokenization & Lemmatization)
def preprocess_text(text):
    words = nltk.word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(word) for word in words if word.isalnum()]
    return words

# TF-IDF Vectorization of the FAQ Data for a given topic
def vectorize_faq_data(topic):
    corpus = [faq['question'] for faq in faq_data[topic]]
    vectorizer = TfidfVectorizer(stop_words='english')
    return vectorizer.fit_transform(corpus)

# Function to Get Most Similar Question based on the selected topic
def get_most_similar_question(user_input, topic):
    vectorizer = TfidfVectorizer(stop_words='english')
    question_vectors = vectorizer.fit_transform([faq['question'] for faq in faq_data[topic]])
    user_input_vec = vectorizer.transform([user_input])
    
    similarity_scores = cosine_similarity(user_input_vec, question_vectors)
    most_similar_idx = similarity_scores.argmax()

    return faq_data[topic][most_similar_idx]['answer']

# Chatbot Response Function
def get_response(user_input, topic):
    response = get_most_similar_question(user_input, topic)
    return response

# Chatbot Loop
def chatbot():
    print("Hello! I'm here to answer your FAQ. (Type 'quit' to exit)")

    # Ask the user to select a topic
    print("Available topics: General, Shipping, Payment")
    topic = input("Choose a topic: ").capitalize()
    
    if topic not in faq_data:
        print("Sorry, that topic is not available.")
        return

    # Start the chatbot loop for the selected topic
    while True:
        user_input = input(f"You ({topic}): ")
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        response = get_response(user_input, topic)
        print(f"Bot: {response}")

# Run the chatbot
chatbot()