# House Price Prediction

A Machine Learning Project Report



---

# 1. Problem Statement

Housing prices depend on many factors such as size, location, age, and quality of construction. Manual valuation is slow and inconsistent.

The goal of this project is to:

- Predict house prices automatically from property features
- Compare multiple regression algorithms and pick the best one
- Deploy the model as a web application where anyone can get an instant estimate

---

# 2. Dataset

- Source: King County (USA) housing data, 50k property listings
- File: `dataset/house_price_50k.csv`
- Size: 21,613 rows, 21 columns
- Target variable: Price

| Statistic | Value |
|---|---|
| Minimum price | $75,000 |
| Average price | $540,182 |
| Maximum price | $7,700,000 |
| Living area range | 290 - 13,540 sq.ft |
| Year built range | 1900 - 2015 |

---

# 3. Features Used

12 input features were kept for modeling:

| Feature | Description |
|---|---|
| Bedrooms | Number of bedrooms |
| Bathrooms | Number of bathrooms |
| Sqft_living | Living area in sq.ft |
| Sqft_lot | Lot size in sq.ft |
| Floors | Number of floors |
| Waterfront | Waterfront view (0/1) |
| View | View quality rating (0-4) |
| Grade | Build quality grade (1-13) |
| Sqft_living15 | Avg. living area of 15 nearest neighbors |
| Sqft_lot15 | Avg. lot size of 15 nearest neighbors |
| Is_renovated | Whether the house was renovated (0/1) |
| Age | 2026 minus year built |

---

# 4. Data Preprocessing

Steps performed in the Jupyter notebook:

- Checked for missing values and duplicates
- Dropped unused columns: ID, Date, Condition, Zipcode, Latitude, Longitude, Sqft_above, Sqft_basement
- Converted Yr_renovated into a binary flag `Is_renovated`
- Replaced Yr_built with `Age` (current year minus year built)
- Applied StandardScaler on selected features for Linear Regression
- 80/20 train-test split with random state 42

Note: Zipcode, Latitude and Longitude were dropped because location was already indirectly captured by Sqft_living15 and Sqft_lot15, and removing them reduced noise from the model.

---

# 5. Exploratory Data Analysis

Analysis was done using Matplotlib and Seaborn:

- Distribution of house prices - right skewed, most houses between $200k and $800k
- Price vs Living area - strong positive correlation
- Price vs Grade - higher grade houses cost significantly more
- Count plots for bedrooms, bathrooms and floors
- Correlation heatmap of all remaining numeric features

Strongest correlations with Price:

- Grade
- Sqft_living
- Sqft_living15

---

# 6. Correlation Heatmap

A heatmap was generated on the final 13 numeric columns (12 features + Price).

Key observations:

- Sqft_living and Sqft_above were highly related (hence Sqft_above was dropped)
- Sqft_living15 correlates strongly with Sqft_living - neighborhood effect
- Grade has the highest correlation with Price among all features
- No two retained features are redundant with each other

---

# 7. Models Trained

Three regression algorithms were compared on the same test split:

1. Linear Regression (with scaled features) - baseline
2. Decision Tree Regressor (max depth 10)
3. Random Forest Regressor (200 trees)

Evaluation metrics: R² score, Mean Squared Error (MSE), Mean Absolute Error (MAE)

---

# 8. Model Comparison

| Model | R² Score | MSE | MAE |
|---|---|---|---|
| Linear Regression | 0.647 | 42.2 Billion | $137,376 |
| Decision Tree | 0.628 | 44.5 Billion | $129,046 |
| **Random Forest** | **0.739** | **31.2 Billion** | **$114,421** |

Random Forest gave the highest R² and the lowest prediction error.

---

# 9. Result

- Selected model: Random Forest Regressor (200 estimators)
- R² score: 0.74 on unseen test data
- Average error: about $114k on house prices averaging $540k, roughly 21% error
- The model captures non-linear relationships (e.g. grade and waterfront effects) that Linear Regression misses

Sample prediction: a 30 year old, 2000 sq.ft house with grade 8 is predicted around $380,000.

---

# 10. Web Application

Built with Streamlit, the app takes all 12 features as input:

- Numeric inputs: bedrooms, bathrooms, living area, lot size, age, neighbor averages
- Choice inputs: floors, waterfront, view rating, grade, renovated status
- One click returns the predicted price
- Submitted details can be reviewed in an expandable table

The model is loaded once using `st.cache_resource` for fast predictions.

---

# 11. Deployment

- Hosted on Streamlit Community Cloud
- Live link: https://housepriceprediction-vyv.streamlit.app/
- Model file (`Random_Forest_House_Price.pkl`, ~58 MB) and feature list (`columns.pkl`) are stored in the `models/` folder of the repository
- The app reads these files at startup and serves real-time predictions

---

# 12. Project Structure

```
HousePricePrediction/
|
+-- dataset/
|   +-- house_price_50k.csv
+-- models/
|   +-- Random_Forest_House_Price.pkl
|   +-- columns.pkl
+-- Notebook/
|   +-- HousePricePrediction.ipynb
+-- Streamlit_App/
|   +-- app.py
+-- requirements.txt
+-- README.md
```

---

# 13. Tech Stack

- Python
- Pandas, NumPy - data handling
- Matplotlib, Seaborn - visualization
- scikit-learn - models and metrics
- Joblib - model serialization
- Streamlit - web app and deployment
- GitHub - version control and hosting model files

---

# 14. Limitations

- Average error of ~21% is high for precise valuation use
- No real-time market data - prices reflect the training period only
- Property tax, school district and crime data are not included
- Random Forest cannot extrapolate to markets very different from the training data

---

# 15. Future Scope

- Hyperparameter tuning with GridSearchCV
- Feature importance plot to show which factors drive price
- Add SHAP values for per-prediction explanations
- Include external data: school ratings, crime rate, market trends
- Try Gradient Boosting / XGBoost for better accuracy

---

# 16. Thank You

Questions?

Repository: https://github.com/YashKull2005/HousePricePrediction
