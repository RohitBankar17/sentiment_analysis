from flask import Flask, request, jsonify
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


app = Flask(__name__)

output = {}


def sentiment_analysis(sentence):
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(sentence)['compound']

    if (score > 0):
        return "Positive"
    else:
        return "Negative"


@app.route('/', methods=['GET', 'POST'])
def sentiment_request():
    if request.method == 'POST':
        sentence = request.form["q"]
    else:
        sentence = request.args.get("q")

    sent = sentiment_analysis(sentence)
    output['sentiment'] = sent
    return jsonify(output)


if __name__ == '__main__':
    app.run(debug=True)
