from flask import Flask, render_template, request, jsonify  # type: ignore
import pandas as pd  # type: ignore
import os
import json

app = Flask(__name__)

# ======================================
# Dataset Configuration
# ======================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "dataset", "tmdb_5000_movies.csv")

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError("Dataset not found inside dataset folder.")

df = pd.read_csv(DATA_FILE)

# Keep Required Columns
df = df[['budget', 'revenue', 'popularity', 'vote_average', 'genres', 'release_date']]
df = df.dropna()

# Extract Year
df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
df = df.dropna(subset=['release_year'])

# Extract Main Genre
def extract_genre(g):
    try:
        genre_list = json.loads(g.replace("'", '"'))
        return genre_list[0]['name'] if genre_list else None
    except:
        return None

df['main_genre'] = df['genres'].apply(extract_genre)
df = df.dropna(subset=['main_genre'])

# Profit & ROI
df['profit'] = df['revenue'] - df['budget']
df['roi'] = (df['profit'] / df['budget']) * 100

# ======================================
# HOME
# ======================================

@app.route('/')
def home():
    return render_template("home.html")

# ======================================
# PREDICTION (REAL DATA BASED)
# ======================================

@app.route('/predict', methods=['GET', 'POST'])
def predict():

    genres = sorted(df['main_genre'].unique())

    if request.method == 'POST':
        try:
            budget = float(request.form['budget'])
            genre = request.form['genre']

            # Filter dataset by selected genre
            genre_df = df[df['main_genre'] == genre]

            if genre_df.empty:
                return render_template(
                    'predict.html',
                    genres=genres,
                    error="No data available for selected genre."
                )

            # Real values from dataset
            avg_revenue = genre_df['revenue'].mean()
            avg_budget = genre_df['budget'].mean()
            avg_roi = genre_df['roi'].mean()
            success_rate = (len(genre_df[genre_df['profit'] > 0]) / len(genre_df)) * 100

            # Prediction logic:
            # Adjust revenue based on user budget vs average genre budget
            predicted_revenue = avg_revenue * (budget / avg_budget)

            profit = predicted_revenue - budget
            roi = (profit / budget) * 100
            status = "Hit" if profit > 0 else "Flop"

            return render_template(
                'predict.html',
                genres=genres,
                prediction=float(f"{predicted_revenue:.2f}"),
                profit=float(f"{profit:.2f}"),
                roi=float(f"{roi:.2f}"),
                status=status,
                success_rate=float(f"{success_rate:.2f}"),
                avg_genre_revenue=float(f"{avg_revenue:.2f}")
            )

        except Exception:
            return render_template(
                'predict.html',
                genres=genres,
                error="Invalid input values."
            )

    return render_template('predict.html', genres=genres)

# ======================================
# DASHBOARD
# ======================================

@app.route('/dashboard')
def dashboard():
    genres = sorted(df['main_genre'].unique())
    years = sorted(df['release_year'].unique())
    return render_template("dashboard.html", genres=genres, years=years)

# ======================================
# DASHBOARD DATA API
# ======================================

@app.route('/get_dashboard_data')
def get_dashboard_data():

    genre = request.args.get('genre')
    year = request.args.get('year')
    budget = request.args.get('budget')

    filtered = df.copy()

    if genre and genre != "All":
        filtered = filtered[filtered['main_genre'] == genre]

    if year and year != "All":
        filtered = filtered[filtered['release_year'] == int(year)]

    if budget and budget != "All":
        if budget == "Low":
            filtered = filtered[filtered['budget'] < 20000000]
        elif budget == "Medium":
            filtered = filtered[(filtered['budget'] >= 20000000) & (filtered['budget'] < 100000000)]
        else:
            filtered = filtered[filtered['budget'] >= 100000000]

    revenue_by_genre = filtered.groupby('main_genre')['revenue'].mean().to_dict()
    success = len(filtered[filtered['profit'] > 0])
    failure = len(filtered[filtered['profit'] <= 0])
    year_trend = filtered.groupby('release_year')['revenue'].mean().to_dict()
    avg_budget = filtered.groupby('main_genre')['budget'].mean().to_dict()

    roi_dist = {
        "High ROI (>100%)": len(filtered[filtered['roi'] > 100]),
        "Moderate ROI (0-100%)": len(filtered[(filtered['roi'] > 0) & (filtered['roi'] <= 100)]),
        "Loss": len(filtered[filtered['roi'] <= 0])
    }

    popularity_data = list(zip(filtered['popularity'], filtered['revenue']))
    
    kpis = {
        "total_movies": len(filtered),
        "avg_revenue": filtered['revenue'].mean() if not filtered.empty else 0,
        "avg_budget": filtered['budget'].mean() if not filtered.empty else 0,
        "success_rate": (success / len(filtered) * 100) if len(filtered) > 0 else 0
    }

    # Handle NaN in JSON properly
    import math
    if math.isnan(kpis["avg_revenue"]): kpis["avg_revenue"] = 0
    if math.isnan(kpis["avg_budget"]): kpis["avg_budget"] = 0

    return jsonify({
        "revenue_by_genre": revenue_by_genre,
        "success": success,
        "failure": failure,
        "year_trend": year_trend,
        "avg_budget": avg_budget,
        "roi_dist": roi_dist,
        "popularity": popularity_data,
        "kpis": kpis
    })

# ======================================
# About Page
# ======================================    
    
@app.route('/about')
def about():

    if not os.path.exists(DATA_FILE):
        return "CSV file not found. Please check dataset folder."

    df = pd.read_csv(DATA_FILE)

    # Basic Insights
    total_movies = len(df)
    avg_budget = round(df['budget'].mean(), 2)
    avg_revenue = round(df['revenue'].mean(), 2)
    max_revenue = int(df['revenue'].max())

    # Success vs Failure
    df['profit'] = df['revenue'] - df['budget']
    success = len(df[df['profit'] > 0])
    failure = len(df[df['profit'] <= 0])

    # Genre Distribution
    genre_count = {}

    for item in df['genres']:
        try:
            genre_list = json.loads(item.replace("'", '"'))
            for g in genre_list:
                name = g['name']
                genre_count[name] = genre_count.get(name, 0) + 1
        except:
            continue

    sorted_genres = sorted(
        genre_count.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    top_genres = [sorted_genres[i] for i in range(min(5, len(sorted_genres)))]

    genres = [g[0] for g in top_genres]
    genre_values = [g[1] for g in top_genres]

    return render_template(
        "about.html",
        total_movies=total_movies,
        avg_budget=avg_budget,
        avg_revenue=avg_revenue,
        max_revenue=max_revenue,
        success=success,
        failure=failure,
        genres=genres,
        genre_values=genre_values
    )

# ======================================
# CLASSIFICATION (ML INFERENCE INTERFACE)
# ======================================

@app.route('/classification', methods=['GET', 'POST'])
def classification():
    # Provide simple genres array for custom form inputs
    genres = sorted(df['main_genre'].unique())
    import joblib  # type: ignore
    model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    
    if request.method == 'POST':
        try:
            budget = float(request.form['budget'])
            runtime = float(request.form['runtime'])
            vote_average = float(request.form['vote_average'])
            vote_count = float(request.form['vote_count'])
            popularity = float(request.form['popularity'])
            
            # Predict using existing Regression Model
            import numpy as np  # type: ignore
            features = np.array([budget, runtime, vote_average, vote_count, popularity]).reshape(1, -1)
            scaled_features = scaler.transform(features)
            predicted_revenue = model.predict(scaled_features)[0]
            
            profit = predicted_revenue - budget
            roi = (profit / budget) * 100
            status = "Hit" if profit > 0 else "Flop"
            
            return render_template(
                'classification.html',
                genres=genres,
                prediction=float(f"{predicted_revenue:.2f}"),
                profit=float(f"{profit:.2f}"),
                roi=float(f"{roi:.2f}"),
                status=status
            )
            
        except Exception as e:
            return render_template('classification.html', genres=genres, error=str(e))

    return render_template('classification.html', genres=genres)


# ======================================
# RUN APP
# ======================================

if __name__ == "__main__":
    app.run(debug=True)