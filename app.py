from collections import defaultdict
from flask import Flask, jsonify, render_template, request

from rss import get_rss_articles
from sentiment import get_sentiment, get_sentiments_from_rss_feed

app = Flask(__name__)

model_list = ["VADER", "TextBlob", "HappyTransformer"]


def generate_chart_data(articles):
    """
    Generates chart data based on the sentiments of the articles.

    Args:
        articles (list): A list of RSSArticleSentiment objects representing the sentiments of the articles.

    Returns:
        dict: A dictionary containing chart data for titles, descriptions, and both.
    """
    title_counts = defaultdict(int)
    description_counts = defaultdict(int)
    both_counts = defaultdict(int)

    for article in articles:
        title_counts[article.title_label] += 1
        description_counts[article.description_label] += 1
        both_counts[article.both_label] += 1

    sorted_labels = ["Positive", "Negative", "Neutral"]

    chart_data = {
        "titles": {
            "labels": sorted_labels,
            "data": [title_counts[label] for label in sorted_labels],
        },
        "descriptions": {
            "labels": sorted_labels,
            "data": [description_counts[label] for label in sorted_labels],
        },
        "both": {
            "labels": sorted_labels,
            "data": [both_counts[label] for label in sorted_labels],
        },
    }

    return chart_data


@app.route("/")
def home():
    """
    Renders the home page with the list of available models.
    """
    return render_template("index.html", models=model_list)


@app.route("/predict-sentiment", methods=["POST"])
def predict_sentiment():
    """
    Predicts the sentiment of a given text using the specified model.

    Returns:
        JSON: A JSON response containing the predicted sentiment label and compound score.
    """
    data = request.json

    text = data["text"]
    model = data["model"]

    label, compound_score = get_sentiment(model, text)

    response = {"prediction": compound_score, "label": label}

    return jsonify(response)


@app.route("/predict-sentiments-from-rss", methods=["POST"])
def predict_sentiments_from_rss():
    """
    Predicts the sentiments of articles from an RSS feed using the specified model.

    Returns:
        JSON: A JSON response containing chart data for titles, descriptions, and both.
    """
    data = request.json

    feed_url = data["feed_url"]
    model = data["model"]

    articles = get_rss_articles(feed_url)

    articles_sentiments = get_sentiments_from_rss_feed(model, articles)

    chart_data = generate_chart_data(articles_sentiments)

    return jsonify(chart_data)
