from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow.pyfunc
import mlflow.xgboost
from pathlib import Path
import os
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
import xgboost as xgb

# Global variable to store the loaded model
model = None
feature_columns = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    load_model_on_startup()
    yield
    # Shutdown (if needed)
    pass


app = FastAPI(title="Churn Prediction API", version="1.0.0", lifespan=lifespan)


def find_latest_model():
    """Find the latest MLflow model from mlruns directory."""
    mlruns_path = Path("mlruns/0")
    if not mlruns_path.exists():
        raise FileNotFoundError("mlruns directory not found")
    
    # First, try to find models directly in the models directory
    models_path = mlruns_path / "models"
    if models_path.exists():
        model_dirs = [d for d in models_path.iterdir() if d.is_dir() and d.name.startswith("m-")]
        if model_dirs:
            # Get the most recently modified model directory
            latest_model_dir = max(model_dirs, key=lambda p: p.stat().st_mtime)
            artifacts_path = latest_model_dir / "artifacts"
            # Check if artifacts directory exists and has MLmodel file
            if artifacts_path.exists() and (artifacts_path / "MLmodel").exists():
                return str(artifacts_path.absolute())
    
    # Fallback: try to find run directories (32-char run IDs)
    run_dirs = [d for d in mlruns_path.iterdir() if d.is_dir() and len(d.name) == 32]
    if run_dirs:
        # Get the most recently modified run directory
        latest_run = max(run_dirs, key=lambda p: p.stat().st_mtime)
        artifacts_path = latest_run / "artifacts" / "model"
        # Check if model subdirectory exists, otherwise check artifacts directly
        if artifacts_path.exists():
            return str(artifacts_path.absolute())
        elif (latest_run / "artifacts").exists() and (latest_run / "artifacts" / "MLmodel").exists():
            return str((latest_run / "artifacts").absolute())
    
    # Last resort: try MLflow tracking API (may fail if meta.yaml missing)
    try:
        import mlflow
        mlflow.set_tracking_uri("file:./mlruns")
        experiments = mlflow.search_experiments()
        if experiments:
            exp_id = experiments[0].experiment_id
            runs = mlflow.search_runs(experiment_ids=[exp_id], order_by=["start_time DESC"], max_results=1)
            if not runs.empty:
                run_id = runs.iloc[0]["run_id"]
                model_uri = f"runs:/{run_id}/model"
                return model_uri
    except Exception as e:
        print(f"Could not use MLflow tracking: {e}")
    
    raise FileNotFoundError("No MLflow model found in mlruns directory")


def load_model_on_startup():
    """Load the latest MLflow model on application startup."""
    global model, feature_columns
    
    # Monkey-patch XGBoost to handle missing _estimator_type
    original_get_type = xgb.sklearn.XGBModel._get_type
    
    def patched_get_type(self):
        """Patched _get_type that defaults to classifier if _estimator_type is missing."""
        try:
            return original_get_type(self)
        except TypeError as e:
            if "_estimator_type" in str(e):
                # If _estimator_type is missing, assume it's a classifier
                if not hasattr(self, '_estimator_type'):
                    self._estimator_type = "classifier"
                return self._estimator_type
            raise
    
    # Apply the patch
    xgb.sklearn.XGBModel._get_type = patched_get_type
    
    try:
        model_uri = find_latest_model()
        print(f"Loading model from: {model_uri}")
        # Try to load as XGBoost model first (for predict_proba support)
        try:
            model = mlflow.xgboost.load_model(model_uri)
            # Ensure _estimator_type is set
            if not hasattr(model, '_estimator_type') or model._estimator_type is None:
                model._estimator_type = "classifier"
            print("Model loaded as XGBoost model (with predict_proba support)")
        except Exception as e:
            print(f"Could not load as XGBoost model: {e}")
            # Fallback to pyfunc
            try:
                model = mlflow.pyfunc.load_model(model_uri)
                print("Model loaded as PyFunc model")
                # Try to extract raw model and set _estimator_type if possible
                if hasattr(model, '_model_impl'):
                    if hasattr(model._model_impl, 'get_raw_model'):
                        raw_model = model._model_impl.get_raw_model()
                        if hasattr(raw_model, '_estimator_type'):
                            raw_model._estimator_type = "classifier"
                    elif hasattr(model._model_impl, '_estimator_type'):
                        model._model_impl._estimator_type = "classifier"
            except Exception as e2:
                print(f"Could not load as PyFunc model either: {e2}")
                raise
        print("Model loaded successfully!")
        
        # Get feature columns by running preprocessing on a sample
        # This helps us understand what columns the model expects
        sample_data_path = Path("scripts/data/telco_churn.csv.csv")
        if sample_data_path.exists():
            df_sample = pd.read_csv(sample_data_path, nrows=1)
            df_sample = df_sample.drop("customerID", axis=1)
            df_sample["Churn"] = df_sample["Churn"].map({"Yes": 1, "No": 0})
            df_sample = pd.get_dummies(df_sample)
            feature_columns = [col for col in df_sample.columns if col != "Churn"]
            print(f"Model expects {len(feature_columns)} feature columns")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise




# Input schema - accepts original features (without customerID and Churn)
class PredictionRequest(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: Optional[str] = None  # Can be string or numeric


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int


def preprocess_input(data: Dict[str, Any]) -> pd.DataFrame:
    """Preprocess input data to match training format."""
    # Convert to DataFrame
    df = pd.DataFrame([data])
    
    # Apply same preprocessing as training
    # Note: customerID is not in the input, so we skip dropping it
    # Churn is not in the input either
    
    # Convert categorical columns to dummy variables
    df = pd.get_dummies(df)
    
    return df


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Churn Prediction API",
        "status": "running",
        "model_loaded": model is not None
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict churn probability for a customer."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert request to dict
        input_data = request.dict()
        
        # Preprocess the input
        df_processed = preprocess_input(input_data)
        
        # Ensure all expected feature columns are present
        if feature_columns:
            # Add missing columns with 0 values
            for col in feature_columns:
                if col not in df_processed.columns:
                    df_processed[col] = 0
        
            # Select only the expected columns in the right order
            df_processed = df_processed.reindex(columns=feature_columns, fill_value=0)
        
        # Make prediction
        # Try to get probabilities from the model
        churn_prob = 0.5  # Default
        churn_pred = 0
        
        # If it's an XGBoost model, use predict_proba
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(df_processed)
                churn_prob = float(probabilities[0][1]) if len(probabilities[0]) > 1 else float(probabilities[0][0])
                churn_pred = int(churn_prob > 0.5)
            except Exception:
                # Fallback to predict
                prediction_result = model.predict(df_processed)
                churn_pred = int(prediction_result[0]) if not isinstance(prediction_result, pd.DataFrame) else int(prediction_result.iloc[0, 0])
                churn_prob = float(churn_pred)
        else:
            # For pyfunc models, try to access underlying model
            try:
                # Try to get raw model from pyfunc wrapper
                if hasattr(model, '_model_impl'):
                    raw_model = model._model_impl
                    if hasattr(raw_model, 'get_raw_model'):
                        raw_model = raw_model.get_raw_model()
                    if hasattr(raw_model, 'predict_proba'):
                        probabilities = raw_model.predict_proba(df_processed)
                        churn_prob = float(probabilities[0][1]) if len(probabilities[0]) > 1 else float(probabilities[0][0])
                        churn_pred = int(churn_prob > 0.5)
                    else:
                        # Use predict
                        prediction_result = model.predict(df_processed)
                        if isinstance(prediction_result, pd.DataFrame):
                            churn_pred = int(prediction_result.iloc[0, 0])
                        else:
                            churn_pred = int(prediction_result[0])
                        churn_prob = float(churn_pred)
                else:
                    # Standard pyfunc predict
                    prediction_result = model.predict(df_processed)
                    if isinstance(prediction_result, pd.DataFrame):
                        churn_pred = int(prediction_result.iloc[0, 0])
                    else:
                        churn_pred = int(prediction_result[0])
                    churn_prob = float(churn_pred)
            except Exception as e:
                # Final fallback
                prediction_result = model.predict(df_processed)
                if isinstance(prediction_result, pd.DataFrame):
                    churn_pred = int(prediction_result.iloc[0, 0])
                else:
                    churn_pred = int(prediction_result[0])
                churn_prob = float(churn_pred)
        
        return PredictionResponse(
            churn_probability=churn_prob,
            churn_prediction=churn_pred
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

