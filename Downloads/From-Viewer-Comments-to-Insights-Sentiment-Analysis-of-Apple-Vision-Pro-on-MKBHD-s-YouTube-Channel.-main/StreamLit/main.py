import streamlit as st
import pandas as pd
from nltk.corpus import stopwords 
import nltk
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re  # Import the regular expressions library

# Ensure nltk resources are downloaded
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Write a title and some text to the app:
st.title('Sentiment Analysis Tool Dashboard')

st.write('From Viewer Comments to Insights: Sentiment Analysis of Apple Vision Pro on MKBHD YouTube Channel')

# Define a set of crucial negation words
negations = {'not', 'no', 'never', 'none'}

# Text cleaning and sentiment analysis
with st.expander('Text Cleaning and Analysis'):
    text_input = st.text_area('Enter text to clean and analyze:')
    if text_input:
        # Cleaning the text
        text_cleaned = re.sub(r'#\w+', '', text_input)  # Remove hashtags
        text_cleaned = re.sub(r'\d+', '', text_cleaned)  # Remove numbers
        text_cleaned = ' '.join(text_cleaned.split())  # Remove extra spaces
        tokens = word_tokenize(text_cleaned.lower())  # Tokenize and convert to lower case
        clean_tokens = [word for word in tokens if word not in stopwords.words('english') or word in negations]  # Keep crucial negations
        cleaned_text = ' '.join(clean_tokens)
        st.write('Cleaned Text:', cleaned_text)

        # Sentiment Analysis
        sentiment_result = analyzer.polarity_scores(cleaned_text)
        st.write('Polarity Scores:', sentiment_result)
        sentiment_category = 'Positive' if sentiment_result['compound'] > 0.05 else 'Negative' if sentiment_result['compound'] < -0.05 else 'Neutral'
        st.write('Overall Sentiment:', sentiment_category)

        # CSV analysis with sentiment classification and word clouds
with st.expander('Upload and Analyze CSV using VADER and Generate Word Clouds'):
    uploaded_file = st.file_uploader("Choose a CSV file", key="csv2")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Here's the first few rows of your file:")
        st.write(df.head())

        # Apply VADER sentiment analysis on a specific column
        if 'comments' in df.columns:  # Assuming 'comments' is the column to analyze
            df['sentiment_scores'] = df['comments'].apply(lambda x: analyzer.polarity_scores(x))
            df['sentiment_category'] = df['sentiment_scores'].apply(
                lambda x: 'Positive' if x['compound'] > 0.05 else 'Negative' if x['compound'] < -0.05 else 'Neutral')
            st.write(df[['comments', 'sentiment_category']])

            # Select a sentiment category to generate word cloud
            selected_category = st.selectbox('Select a sentiment category to generate word cloud:', ['Positive', 'Negative', 'Neutral'])
            filtered_comments = df[df['sentiment_category'] == selected_category]['comments'].str.cat(sep=' ')
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(filtered_comments)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt)
        else:
            st.error("The uploaded file does not contain the required column: 'comments'.")