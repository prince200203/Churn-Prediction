# 📊 Customer Churn Prediction App

An AI-powered web application that predicts the likelihood of customer churn. This application has been enhanced to use the full feature set and analysis logic from the project's Jupyter Notebook (`main.ipynb`).

## 🚀 Features

- **Full Notebook Logic Integration:** Now uses all 19 features identified as predictors in the original analysis.
- **Interactive Sidebar:** Customizable customer parameters including demographics, services (Internet, Security), and account details (Contract, Payment Method).
- **Real-time Prediction:** Get instant results using a Random Forest model trained on the full dataset.
- **Probability Scores:** View the specific probability of churn for each prediction.
- **Training Data Preview:** View the cleaned and encoded training dataset directly in the app.

## 🛠️ Installation

To run this project locally, ensure you have Python installed.

1.  **Clone the repository** (if applicable) or navigate to the project folder.
2.  **Install dependencies:**
    ```bash
    py -m pip install streamlit pandas numpy scikit-learn
    ```

## 💻 Running the App

Execute the following command in your terminal:

```bash
py -m streamlit run app.py
```

The app will be available at `http://localhost:8501`.

## 📂 Project Structure

- `app.py`: The enhanced Streamlit application with notebook-driven logic.
- `main.ipynb`: The original research and model exploration notebook.
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`: The dataset used for training.
- `README.md`: This file.

## 📊 Dataset Information

The dataset used is the **Telco Customer Churn** dataset, which includes comprehensive customer information used to predict churn behavior.
