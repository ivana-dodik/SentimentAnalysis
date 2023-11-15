/**
 * Predicts the sentiment of a given text using the selected model.
 */
function predictSentiment() {
    let text = document.getElementById("text").value;
    let selectedModel = document.getElementById("model").value;

    let data = {
        "text": text,
        "model": selectedModel
    };

    fetch("/predict-sentiment", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(data)
    })
        .then(response => response.json())
        .then(result => {
            let score = result.prediction;
            let label = result.label;

            document.getElementById("score").textContent = score.toFixed(2);
            document.getElementById("label").textContent = label;

            // Update label class based on sentiment
            let labelElement = document.getElementById("label");
            labelElement.classList.remove("positive");
            labelElement.classList.remove("negative");
            labelElement.classList.remove("neutral");
            labelElement.classList.add(label.toLowerCase());
        })
        .catch(error => {
            console.error("Error:", error);
        });
}

// Global variable to store chart instances
const chartInstances = {};

/**
 * Predicts the sentiments of articles from an RSS feed using the selected model.
 */
function predictSentimentFromRSS() {
    let rssFeedUrl = document.getElementById("rss-url").value;
    let selectedModel = document.getElementById("model").value;

    fetch("/predict-sentiments-from-rss", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            feed_url: rssFeedUrl,
            model: selectedModel,
        }),
    })
        .then((response) => response.json())
        .then((data) => {
            generateChart("titles", data.titles.labels, data.titles.data);
            generateChart("descriptions", data.descriptions.labels, data.descriptions.data);
            generateChart("both", data.both.labels, data.both.data);
        })
        .catch((error) => {
            console.error("Error:", error);
        });
}

/**
 * Generates a chart using Chart.js library.
 * @param {string} chartId - The ID of the chart canvas element.
 * @param {Array} labels - The labels for the chart.
 * @param {Array} data - The data points for the chart.
 */
function generateChart(chartId, labels, data) {
    const ctx = document.getElementById(chartId).getContext("2d");

    if (chartInstances[chartId]) {
        // Update existing chart instance
        chartInstances[chartId].data.labels = labels;
        chartInstances[chartId].data.datasets[0].data = data;
        chartInstances[chartId].update();
    } else {
        // Create new chart instance
        chartInstances[chartId] = new Chart(ctx, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [
                    {
                        label: "Sentiment Count",
                        data: data,
                        backgroundColor: [
                            "rgba(54, 162, 235, 0.9)",
                            "rgba(255, 99, 132, 0.9)",
                            "rgba(255, 206, 86, 0.9)",
                        ],
                        borderColor: [
                            "rgb(54, 162, 235)",
                            "rgb(255, 99, 132)",
                            "rgb(255, 206, 86)",
                        ],
                        borderWidth: 1,
                    },
                ],
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: true,
                        precision: 0,
                    },
                },
            },
        });
    }
}
