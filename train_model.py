import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

df = pd.read_csv("dataset/tmdb_5000_movies.csv")

# Clean Data: Drop rows with zero or negligible budget/revenue that distort training
df = df[(df['budget'] > 10000) & (df['revenue'] > 10000)]

features = ["budget", "runtime", "vote_average", "vote_count", "popularity"]
df = df.dropna(subset=features + ["revenue"])

X = df[features]
y = df["revenue"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print(f"Model Trained Successfully! R2 Score: {r2:.2f}, MAE: ${mae:,.2f}")
