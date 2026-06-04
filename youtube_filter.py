import os
from flask import Flask, render_template, request
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from dotenv import load_dotenv

load_dotenv("key.env")

app = Flask(__name__)

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)


def train_model(user_keywords):
    texts = []
    labels = []

    for kw in user_keywords:
        texts.append(f"{kw} tutorial")
        texts.append(f"{kw} lecture")
        texts.append(f"{kw} guide")
        labels.extend([1, 1, 1])

    negatives = [
        "셰프", "쉐프", "게임 방송", "먹방", "브이로그",
        "funny video", "prank", "reaction"
    ]

    for n in negatives:
        texts.append(n)
        labels.append(0)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    model = LogisticRegression()
    model.fit(X, labels)

    return vectorizer, model


def is_relevant(text, vectorizer, model):
    X_test = vectorizer.transform([text])
    prob = model.predict_proba(X_test)[0][1]
    return prob > 0.5


def search_videos(query):
    request_api = youtube.search().list(
        part="snippet",
        q=query,
        maxResults=10,
        type="video"
    )
    response = request_api.execute()

    videos = []
    for item in response["items"]:
        videos.append({
            "title": item["snippet"]["title"],
            "description": item["snippet"]["description"],
            "videoId": item["id"]["videoId"]
        })
    return videos


def filter_videos(videos, include_kw, exclude_kw, use_ml):
    filtered = []

    include_list = [k.strip().lower() for k in include_kw.split(",") if k.strip()]
    exclude_list = [k.strip().lower() for k in exclude_kw.split(",") if k.strip()]

    auto_block = ["셰프", "쉐프"]

    if use_ml and include_list:
        vectorizer, model = train_model(include_list)
    else:
        vectorizer, model = None, None

    for v in videos:
        text = (v["title"] + " " + v["description"]).lower()

        # 자동 차단
        if use_ml:
            if any(k.lower() in text for k in auto_block):
                continue

        # 포함 키워드 필터
        if include_list:
            score = sum(1 for k in include_list if k in text)
            if score == 0:
                continue

        # 제외 키워드 필터
        if exclude_list:
            if any(k in text for k in exclude_list):
                continue

        # ML 필터
        if use_ml and vectorizer:
            if not is_relevant(text, vectorizer, model):
                continue

        filtered.append(v)

    return filtered


@app.route("/", methods=["GET", "POST"])
def index():
    videos = []

    if request.method == "POST":
        query = request.form.get("query", "")
        include_kw = request.form.get("include", "")
        exclude_kw = request.form.get("exclude", "")
        use_ml = request.form.get("ml") == "on"

        results = search_videos(query)
        videos = filter_videos(results, include_kw, exclude_kw, use_ml)

    return render_template("index.html", videos=videos)


if __name__ == "__main__":
    app.run(debug=True)