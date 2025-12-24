# 🔮 Telco Churn Prediction API

A production-ready machine learning pipeline for predicting customer churn in telecommunications, featuring automated training, model versioning with MLflow, CI/CD integration, and a deployed REST API.

[![CI](https://github.com/businessmansince2005/mlflow/actions/workflows/train.yml/badge.svg)](https://github.com/businessmansince2005/mlflow/actions/workflows/train.yml)
[![Live API](https://img.shields.io/badge/API-Live-brightgreen)](https://mlflow-2.onrender.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.127-blue)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1-orange)](https://xgboost.ai)

## 📋 Overview

This project demonstrates a complete MLOps pipeline for churn prediction:

- **Data Processing**: Preprocessing and feature engineering for telco customer data
- **Model Training**: XGBoost classifier with automated hyperparameter tuning
- **Model Versioning**: MLflow for experiment tracking and model registry
- **CI/CD**: Automated training on every push via GitHub Actions
- **API Deployment**: FastAPI REST API deployed on Render.com
- **Production Ready**: Docker containerization, health checks, and monitoring

**Live API**: https://mlflow-2.onrender.com

## 🏗️ Architecture

```
┌─────────────┐
│   Dataset   │
│ telco_churn │
└──────┬──────┘
       │
       ▼
┌─────────────┐      ┌──────────────┐
│   Training  │─────▶│   MLflow     │
│  (XGBoost)  │      │  (Tracking)  │
└──────┬──────┘      └──────┬───────┘
       │                    │
       │                    ▼
       │            ┌──────────────┐
       │            │ Model Store  │
       │            │  (mlruns/)   │
       │            └──────┬───────┘
       │                   │
       ▼                   ▼
┌─────────────┐      ┌──────────────┐
│ GitHub      │─────▶│   FastAPI    │
│ Actions     │      │   (Render)   │
│  (CI/CD)    │      │              │
└─────────────┘      └──────┬───────┘
                            │
                            ▼
                    ┌──────────────┐
                    │   Predict    │
                    │   Endpoint   │
                    └──────────────┘
```

## 🛠️ Tech Stack

- **Machine Learning**: XGBoost, scikit-learn, pandas
- **MLOps**: MLflow (experiment tracking, model registry)
- **API Framework**: FastAPI, Uvicorn
- **CI/CD**: GitHub Actions
- **Deployment**: Docker, Render.com
- **Language**: Python 3.9+

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Git

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/businessmansince2005/mlflow.git
   cd mlflow
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model**
   ```bash
   python scripts/train.py
   ```
   This will:
   - Load and preprocess the telco churn dataset
   - Train an XGBoost classifier
   - Log metrics and model to MLflow
   - Output: Accuracy and AUC scores

5. **Run the API locally**
   ```bash
   uvicorn app.app:app --reload
   ```
   API will be available at: http://localhost:8000

6. **View API documentation**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## 📡 API Usage

### Health Check

```bash
curl https://mlflow-2.onrender.com/health
```

### Predict Churn

```bash
curl -X POST "https://mlflow-2.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.0,
    "TotalCharges": "350"
  }'
```

### Response

```json
{
  "churn_probability": 0.4523,
  "churn_prediction": 0
}
```

- `churn_probability`: Probability of churning (0.0 to 1.0)
- `churn_prediction`: Binary prediction (0 = No churn, 1 = Churn)

### Interactive API Documentation

Visit the live API documentation:
- **Swagger UI**: https://mlflow-2.onrender.com/docs
- **ReDoc**: https://mlflow-2.onrender.com/redoc

## 🔄 CI/CD Pipeline

The project includes automated CI/CD via GitHub Actions:

- **Workflow**: `.github/workflows/train.yml`
- **Trigger**: Automatic on every push to `master` branch
- **Actions**:
  1. Checkout code
  2. Set up Python 3.9
  3. Install dependencies
  4. Run training script
  5. Log results to MLflow

View workflow runs: [GitHub Actions](https://github.com/businessmansince2005/mlflow/actions)

## 📊 Model Performance

Current model metrics (from latest training run):

- **Accuracy**: 78.64%
- **AUC-ROC**: 82.74%

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t telco-churn-api .
```

### Run Container

```bash
docker run -p 8000:8000 telco-churn-api
```

### Test

```bash
curl http://localhost:8000/health
```

## 📁 Project Structure

```
.
├── app/
│   └── app.py              # FastAPI application
├── scripts/
│   ├── train.py            # Model training script
│   └── data/
│       └── telco_churn.csv.csv
├── mlruns/                 # MLflow experiment tracking
│   └── 0/
│       └── models/         # Trained models
├── .github/
│   └── workflows/
│       └── train.yml       # CI/CD pipeline
├── Dockerfile              # Container configuration
├── render.yaml             # Render.com deployment config
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🔍 Key Features

### 1. Automated Model Training
- XGBoost classifier with optimized hyperparameters
- Automatic feature engineering (one-hot encoding)
- Train/test split with reproducible random state

### 2. MLflow Integration
- Experiment tracking and versioning
- Model registry and artifact storage
- Automatic model loading in production

### 3. Production API
- FastAPI with automatic OpenAPI documentation
- Input validation with Pydantic models
- Health check endpoint for monitoring
- Error handling and logging

### 4. CI/CD Pipeline
- Automated training on code changes
- Version control for models
- Reproducible builds

### 5. Cloud Deployment
- Docker containerization
- Deployed on Render.com
- Auto-scaling and health monitoring

## 📸 Screenshots

### MLflow UI
View experiment runs, compare models, and track metrics in the MLflow UI.

### GitHub Actions
Automated training pipeline runs on every push, ensuring models stay up-to-date.

### API Documentation
Interactive Swagger UI for testing and exploring the API endpoints.

### Live API Response
Real-time churn predictions with probability scores.

## 🧪 Testing

### Test Health Endpoint

```bash
curl https://mlflow-2.onrender.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Test Prediction

See [API Usage](#-api-usage) section for complete examples.

## 📝 API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root endpoint with API info |
| GET | `/health` | Health check endpoint |
| POST | `/predict` | Predict churn probability |

### Request Schema

See [API_EXAMPLES.md](API_EXAMPLES.md) for complete field descriptions and examples.

## 🚢 Deployment

The API is deployed on Render.com with:
- Automatic deployments from GitHub
- Health check monitoring
- Environment variable configuration
- Docker-based builds

Deployment configuration: `render.yaml`

## 📚 Documentation

- [API Examples](API_EXAMPLES.md) - Complete API usage examples
- [Deployment Guide](README_DEPLOYMENT.md) - Detailed deployment instructions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Your Name**
- GitHub: [@businessmansince2005](https://github.com/businessmansince2005)

## 🙏 Acknowledgments

- MLflow team for excellent MLOps tooling
- FastAPI for the modern Python web framework
- XGBoost for powerful gradient boosting
- Render.com for seamless deployment platform

---

⭐ If you find this project helpful, please give it a star!
