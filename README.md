# 🎧 Spotify Tracks Analysis with PySpark

This project performs comprehensive data analysis on the [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) using Apache Spark. It explores various aspects of musical trends, popularity metrics, and artist behaviors using PySpark DataFrames and built-in functions.

## 📁 Dataset

- **Source:** [Kaggle - Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)  
- **File:** `dataset.csv`  
- **Columns:** Includes details like `track_name`, `artists`, `popularity`, `genre`, `energy`, `danceability`, `valence`, `instrumentalness`, etc.

## 🔍 Features & Analysis Performed

### 1. 📊 Dataset Distribution
Statistical distribution for numerical columns (mean, stddev, min, max, percentiles) and frequency counts for categorical columns.  
✅ Outputs: `distribution_analysis.json`

### 2. 👯 Collaboration Popularity
Identifies popular collaborations (multiple artists per track) and highlights those with the least popular contributors.

### 3. 🚀 Breakthrough Songs
Detects high-popularity songs (>80) from otherwise low-popularity albums. Also calculates deviations in energy, danceability, and valence.

### 4. 🎼 Genre Sweet Spot
Analyzes tempo ranges (slow, medium, fast) across top genres and their relationship with popularity.

### 5. 🔞 Explicit Content by Genre
Compares popularity between explicit and non-explicit songs across genres.

### 6. ⏱️ Popularity by Length & Danceability
Examines long, danceable tracks and how their popularity compares to their genre’s average.

### 7. 🤐 Explicit Valence Patterns
Studies how explicit content affects emotional tone (`valence`) across different popularity ranges.

### 8. 🎨 Artist Consistency
Evaluates how consistent an artist is in terms of popularity and genre spread, using expected value and variance.

### 9. 🎻 Instrumental Impact
Analyzes the effect of acousticness and instrumentalness on track popularity, especially within instrumental-heavy genres.

