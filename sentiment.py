import nltk
from happytransformer import HappyTextClassification
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from typing import List

import config
from rss import RSSArticle

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('vader_lexicon')

happy_tc = HappyTextClassification(model_type="DISTILBERT",  model_name="distilbert-base-uncased-finetuned-sst-2-english")

class RSSArticleSentiment:
    """
    Class representing the sentiment of an RSS article.

    Attributes:
        title_label (str): The sentiment label of the article's title.
        description_label (str): The sentiment label of the article's description.
        both_label (str): The sentiment label of both the title and description.
    """

    def __init__(self, title_label, description_label, both_label):
        """
        Initialize an RSSArticleSentiment instance.

        Args:
            title_label (str): The sentiment label of the article's title.
            description_label (str): The sentiment label of the article's description.
            both_label (str): The sentiment label of both the title and description.
        """
        self.title_label = title_label
        self.description_label = description_label
        self.both_label = both_label

    def __repr__(self):
        """
        Return a string representation of the RSSArticleSentiment instance.
        """
        return f"Title Label: {self.title_label} -- Description Label: {self.description_label} -- Both Label: {self.both_label}"


def preprocess_text(text):
    """
    Preprocesses the given text by tokenizing, removing stopwords, lemmatizing, and joining the tokens back into a string.

    Args:
        text (str): The text to preprocess.

    Returns:
        str: The preprocessed text.
    """
    # Tokenize the text
    tokens = word_tokenize(text.lower())

    # Remove stop words
    filtered_tokens = [
        token for token in tokens if token not in stopwords.words("english")
    ]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Join the tokens back into a string
    processed_text = " ".join(lemmatized_tokens)

    return processed_text


analyzer = SentimentIntensityAnalyzer()


def get_sentiment_vader(text):
    """
    Performs sentiment analysis using the VADER (Valence Aware Dictionary and sEntiment Reasoner) model.

    Args:
        text (str): The text to analyze.

    Returns:
        tuple: A tuple containing the sentiment label and compound score.
    """
    scores = analyzer.polarity_scores(text)
    compound_score = scores["compound"]
    if compound_score >= config.NEUTRAL_THRESHOLD:
        label = "Positive"
    elif compound_score <= -config.NEUTRAL_THRESHOLD:
        label = "Negative"
    else:
        label = "Neutral"
    return label, compound_score


def get_sentiment_textblob(text):
    """
    Performs sentiment analysis using the TextBlob model.

    Args:
        text (str): The text to analyze.

    Returns:
        tuple: A tuple containing the sentiment label and polarity score.
    """
    blob = TextBlob(text)

    polarity_score = blob.sentiment.polarity

    if polarity_score > config.NEUTRAL_THRESHOLD:
        label = "Positive"
    elif polarity_score < -config.NEUTRAL_THRESHOLD:
        label = "Negative"
    else:
        label = "Neutral"

    return label, polarity_score


def get_sentiment_happytransformer(text):
    """
    Performs sentiment analysis using the HappyTransformers model.

    Args:
        text (str): The text to analyze.

    Returns:
        tuple: A tuple containing the sentiment label and score.
    """
    result = happy_tc.classify_text(text)

    label = result.label
    if label == "POSITIVE":
        label = "Positive"
    elif label == "NEGATIVE":
        label = "Negative"
    else:
        label = "Unknown"
    return label, result.score


def get_sentiment(model, text):
    """
    Performs sentiment analysis using the specified model.

    Args:
        model (str): The name of the sentiment analysis model to use.
        text (str): The text to analyze.

    Returns:
        tuple: A tuple containing the sentiment label and score based on the specified model.
    """
    if model == "TextBlob":
        return get_sentiment_textblob(text)
    elif model == "VADER":
        return get_sentiment_vader(text)
    elif model == "HappyTransformer":
        return get_sentiment_happytransformer(text)
    else:
        raise ValueError("Invalid Model: The provided model is unknown.")


def get_sentiments_from_rss_feed(model, articles: List[RSSArticle]):
    """
    Performs sentiment analysis on the titles, descriptions, and combined text of RSS articles.

    Args:
        model (str): The name of the sentiment analysis model to use.
        articles (List[RSSArticle]): A list of RSSArticle objects representing the articles.

    Returns:
        list: A list of RSSArticleSentiment objects representing the sentiments of the articles.
    """
    print(model)
    sentiments = []
    for article in articles:
        title_label, _ = get_sentiment(model, article.title)
        description_label, _ = get_sentiment(model, article.description)
        both_label, _ = get_sentiment(model, article.title + " " + article.description)

        print(title_label, description_label, both_label)

        sentiments.append(
            RSSArticleSentiment(title_label, description_label, both_label)
        )

    return sentiments
