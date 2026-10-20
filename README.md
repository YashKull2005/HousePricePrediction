# 🏠 House Price Prediction App

A machine learning web application that predicts house prices based on property features such as area, number of bedrooms, bathrooms, stories, and amenities. The model is trained on a housing dataset and deployed as an interactive web app using **Streamlit**.

🔗 **Live Demo:** [https://housepriceprediction-vyv.streamlit.app/](https://housepriceprediction-vyv.streamlit.app/)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Project Workflow](#-project-workflow)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Known Limitations](#-known-limitations)
- [Future Improvements](#-future-improvements)
- [License](#-license)

---

## 🔍 Overview

This project predicts residential house prices using a **Random Forest Regressor** trained on the `Housing.csv` dataset. The workflow covers data exploration, feature encoding, model comparison across three algorithms, and deployment of the best-performing model through a Streamlit interface where users can input property details and get an instant price estimate.

---

## 📊 Dataset

The dataset (`Housing.csv`) contains **545 records** and **13 columns** describing residential properties:

| Column | Description |
|---|---|
| `price` | Sale price of the house (target variable) |
| `area` | Total area of the plot (sq. ft.) |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `stories` | Number of stories/floors |
| `mainroad` | Whether the house faces a main road (yes/no) |
| `guestroom` | Whether a guest room is present (yes/no) |
| `basement` | Whether a basement is present (yes/no) |
| `hotwaterheating` | Whether hot water heating is available (yes/no) |
| `airconditioning` | Whether air conditioning is available (yes/no) |
| `parking` | Number of parking spots |
| `prefarea` | Whether located in a preferred area (yes/no) |
| `furnishingstatus` | Furnishing level (furnished / semi-furnished / unfurnished) |

---

## 🛠 Tech Stack

- **Language:** Python 3
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** scikit-learn (RandomForestRegressor, DecisionTreeRegressor, LinearRegression)
- **Model Serialization:** Joblib
- **Web App / Deployment:** Streamlit

---

## ⚙️ Project Workflow

1. **Exploratory Data Analysis (EDA)** — Inspected data types, checked for missing values and duplicates, and visualized price distribution along with its relationship to features like `mainroad`, `guestroom`, `basement`, `area`, and `furnishingstatus`.
2. **Feature Encoding** — Categorical columns (`mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`, `furnishingstatus`) were one-hot encoded using `pd.get_dummies()`.
3. **Feature Scaling** — Numerical features were standardized using `StandardScaler` for the linear regression baseline.
4. **Model Training & Comparison** — Three regression models were trained and evaluated on a held-out test set (80/20 split):
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor
5. **Model Selection** — The Random Forest Regressor achieved the best performance and was serialized with `joblib` for deployment.
6. **Deployment** — The saved model (`Random_Forest_House_Price.pkl`) and the training feature column order (`columns.pkl`) are loaded in the Streamlit app to serve real-time predictions.

---

## 📈 Model Performance

| Model | R² Score | MAE |
|---|---|---|
| Linear Regression | 0.884 | 0.27 (scaled) |
| Decision Tree Regressor | 0.919 | 379,717 |
| **Random Forest Regressor (selected)** | **0.936** | **280,895** |

*Random Forest was selected as the final model based on the highest R² score and lowest error on the test set.*

---

## 📁 Project Structure

```
HousePricePrediction/
│
├── dataset/
│   └── Housing.csv                     # Dataset
│
├── Documentation/
│   └── documentation.txt               # Project documentation notes
│
├── models/
│   ├── columns.pkl                     # Feature column order used during training
│   └── Random_Forest_House_Price.pkl   # Trained Random Forest model
│
├── Notebook/
│   ├── columns.pkl
│   ├── HousePricePrediction.ipynb      # Notebook: EDA, preprocessing & model training
│   └── Random_Forest_House_Price.pkl
│
├── Streamlit_App/
│   └── app.py                          # Streamlit application
│
├── requirements.txt                     # Python dependencies
└── README.md                            # Project documentation
```

---

## 💻 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/house-price-prediction.git
   cd house-price-prediction
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

**`requirements.txt`**
```
streamlit
pandas
numpy
scikit-learn
joblib
matplotlib
seaborn
```

---

## 🚀 Usage

Run the Streamlit app locally:

```bash
streamlit run Streamlit_App/app.py
```

Then open the local URL shown in your terminal (typically `http://localhost:8501`), enter the property details in the input form, and click **Predict** to get the estimated house price.

---

## ⚠️ Known Limitations

- The dataset is relatively small (545 rows), which limits the model's ability to generalize to markets or property types outside its distribution.
- Predictions are based on historical data and do not account for real-time market trends, location-specific pricing, or inflation.

---

## 🔮 Future Improvements

- Add model explainability (e.g., SHAP values) to show which features drive each prediction.
- Expand the dataset with more diverse, real-world listings.
- Add hyperparameter tuning (GridSearchCV/RandomizedSearchCV) for further performance gains.
- Add input validation and confidence intervals to predictions in the Streamlit UI.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙋 Author

**Yash Kull**
📧 kullsharmayash@gmail.com

