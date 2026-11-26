# Sentiment Analysis on IMDB Movie Reviews

A machine learning project that classifies movie reviews as **positive** or **negative**, with a Streamlit web app for live predictions.

## Features
- Exploratory analysis of the IMDB review dataset (class balance, word clouds)
- Text vectorization with TF-IDF (top 2,500 features)
- Logistic Regression classifier evaluated with a confusion matrix and accuracy score
- Streamlit app: type a review and get an instant prediction

## Model Performance
Accuracy on the test set: **XX%**

## Project Structure
| File | Description |
|------|-------------|
| reviewanalysis.ipynb | Data exploration, training and evaluation |
| app.py | Streamlit web app |
| model.pkl | Trained Logistic Regression model |
| scalar.pkl | Fitted TF-IDF vectorizer |
| IMDB Dataset.csv | Review dataset (review, sentiment) |
| requirements.txt | Python dependencies |

## Tech Stack
Python, pandas, scikit-learn, NLTK, matplotlib, WordCloud, Streamlit

## Installation
    git clone https://github.com/Renumadhuri/sentiment-analysis.git
    cd sentiment-analysis
    pip install -r requirements.txt

## Usage
    python -m streamlit run app.py

Open http://localhost:8501, type a review, and click **Predict**.

## Example
| Review | Prediction |
|--------|------------|
| "This movie was fantastic, I loved it!" | Positive |
| "Terrible plot and awful acting." | Negative |

## Future Improvements
- Train on the full 50,000-review IMDB dataset
- Try other models (SVM, Naive Bayes, LSTM, BERT)
- Show prediction confidence scores
