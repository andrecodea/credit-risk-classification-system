# Credit Risk Classification System

A machine learning application for predicting credit risk using the German Credit Risk dataset. The system includes model training, evaluation, and an interactive web interface for predictions with LLM-powered strategic recommendations.

## 🎯 Features

- **Machine Learning Pipeline**: Random Forest classifier with SMOTE for handling imbalanced data
- **Model Evaluation**: ROC curves, confusion matrices, and comprehensive metrics
- **Web Interface**: Interactive Gradio app for predictions
- **LLM Integration**: Strategic recommendations via OpenAI-compatible API (rendered in Markdown)
- **Feature Engineering**: Credit amount per month derived feature

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| F1 Score (Bad) | 0.84 |
| ROC AUC | 0.758 |
| Recall (Bad) | 85.7% |
| Precision (Bad) | 82.8% |

## 🗂️ Project Structure

```
credit-risk-classification/
├── app.py                    # Gradio web application
├── service.py                # Prediction service and model loading
├── prompt.py                 # LLM prompt templates
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (API keys)
├── german_credit_data.csv    # Original dataset
├── README.md                 # This file
└── artifacts/                # Model artifacts
    ├── rf_smote_pipeline.pkl    # Trained model pipeline
    ├── metrics.pkl              # Model evaluation metrics
    ├── feature_names.pkl        # Feature names
    ├── X_test.pkl              # Test features
    ├── y_test.pkl              # Test labels
    ├── target_mapping.pkl       # Target variable mapping
    ├── target_encoder.pkl       # Target encoder
    ├── sex_encoder.pkl          # Sex encoder
    ├── housing_encoder.pkl      # Housing encoder
    ├── saving_accounts_encoder.pkl  # Savings encoder
    └── checking_account_encoder.pkl # Checking encoder
```

## 🚀 Quick Start

> **Note**: Model artifacts (`artifacts/*.pkl`) are not included in the repo. Run `credit_risk_analysis.ipynb` to train the model and generate them.

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Key

Edit the `.env` file:

```
INCEPTION_API_KEY=your-api-key-here
```

### 3. Run the App

```bash
python app.py
```

The app will open in your browser at `http://localhost:7860`.

## 📖 How to Use

### Model Showcase Tab
- View model performance metrics
- Interactive ROC curve
- Confusion matrix visualization

### Make Prediction Tab
1. Select applicant features:
   - Age (slider: 18-80)
   - Sex (male/female)
   - Job type (0-3)
   - Housing (own/rent/free)
   - Savings account
   - Checking account
   - Credit amount
   - Duration (months)

2. Click "Analyze & Get Recommendation"

3. View:
   - Risk prediction (HIGH RISK / LOW RISK)
   - Confidence percentage
   - Strategic LLM recommendation (Markdown formatted)

## 🔧 Technical Details

### Model Pipeline

```python
Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight='balanced',
        random_state=42
    ))
])
```

### Feature Mappings

| Feature | Values |
|---------|--------|
| Sex | female → 0, male → 1 |
| Housing | free → 0, own → 1, rent → 2 |
| Savings | little → 0, moderate → 1, quite rich → 2, rich → 3, none → 4 |
| Checking | little → 0, moderate → 1, rich → 2, none → 3 |
| Job | 0 (unskilled), 1 (skilled), 2-3 (highly skilled) |

### Derived Features

- `credit_per_month`: Credit amount / Duration

## 🛠️ Dependencies

- gradio
- openai
- python-dotenv
- joblib
- pandas
- numpy
- scikit-learn
- imbalanced-learn (SMOTE)
- matplotlib

## 📝 License

This project is for educational purposes.

## 👤 Author

- André Costa

## 🙏 Acknowledgments

- German Credit Risk Dataset
- scikit-learn for ML tools
- Gradio for web interface
- OpenAI-compatible LLM for recommendations
