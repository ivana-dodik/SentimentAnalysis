import feedparser


class RSSArticle:
    """
    Class representing an RSS article.

    Attributes:
        title (str): The title of the article.
        description (str): The description or content of the article.
    """

    def __init__(self, title, description):
        """
        Initialize an RSSArticle instance.

        Args:
            title (str): The title of the article.
            description (str): The description or content of the article.
        """
        self.title = title
        self.description = description


def get_rss_articles(url, num_articles=50):
    """
    Fetches RSS articles from the given URL.

    Args:
        url (str): The URL of the RSS feed.
        num_articles (int, optional): The number of articles to retrieve. Defaults to 50.

    Returns:
        list: A list of RSSArticle objects representing the fetched articles.
    """
    feed = feedparser.parse(url)
    entries = feed.entries[:num_articles]

    articles = []
    for entry in entries:
        title = entry.get("title", "")
        description = entry.get("description", "")

        article = RSSArticle(title, description)
        articles.append(article)

    return articles
