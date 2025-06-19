import os
import pickle
import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import joblib
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EarthquakeModelTrainer:
    """
    Class to train and evaluate machine learning models for earthquake magnitude prediction.
    """
    def __init__(self, models_dir='models'):
        """
        Initialize the model trainer.
        
        Args:
            models_dir (str): Directory to save trained models.
        """
        self.models_dir = models_dir
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_model_confidence = None
        self.feature_names = None
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
    def train_models(self, X_train, y_train, feature_names=None):
        """
        Train multiple regression models.
        
        Args:
            X_train (array-like): Training features.
            y_train (array-like): Training targets.
            feature_names (list, optional): Names of features for importance analysis.
            
        Returns:
            dict: Trained models.
        """
        try:
            self.feature_names = feature_names
            
            # Initialize models
            models = {
                'random_forest': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=20,
                    min_samples_split=10,
                    random_state=42,
                    n_jobs=-1
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                ),
                'neural_network': MLPRegressor(
                    hidden_layer_sizes=(100, 50),
                    activation='relu',
                    solver='adam',
                    alpha=0.001,
                    batch_size='auto',
                    learning_rate='adaptive',
                    max_iter=1000,
                    random_state=42
                )
            }
            
            # Train each model
            for name, model in models.items():
                logger.info(f"Training {name} model...")
                try:
                    model.fit(X_train, y_train)
                    self.models[name] = model
                    logger.info(f"{name} model trained successfully")
                except Exception as e:
                    logger.error(f"Error training {name} model: {e}")
            
            return self.models
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate trained models and select the best one.
        
        Args:
            X_test (array-like): Testing features.
            y_test (array-like): Testing targets.
            
        Returns:
            dict: Evaluation metrics for each model.
        """
        try:
            if not self.models:
                raise ValueError("No trained models available for evaluation")
                
            results = {}
            best_rmse = float('inf')
            
            for name, model in self.models.items():
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mape = mean_absolute_percentage_error(y_test, y_pred)
                explained_var = explained_variance_score(y_test, y_pred)

                results[name] = {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'mape': mape,
                    'explained_variance': explained_var
                }

                logger.info(f"{name} evaluation: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}, MAPE={mape:.4f}, Explained Variance={explained_var:.4f}")

                # Track best model
                if rmse < best_rmse:
                    best_rmse = rmse
                    self.best_model = model
                    self.best_model_name = name
            
            logger.info(f"Best model: {self.best_model_name} with RMSE={best_rmse:.4f}")
            
            # Save visualization of predictions vs actual
            self._save_prediction_plot(X_test, y_test)
            
            return results
            
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            raise
    
    def _save_prediction_plot(self, X_test, y_test):
        """
        Save a plot comparing predicted vs actual values for the best model.
        
        Args:
            X_test (array-like): Testing features.
            y_test (array-like): Testing targets.
        """
        try:
            if self.best_model is None:
                logger.warning("No best model selected for visualization")
                return
                
            # Generate predictions
            y_pred = self.best_model.predict(X_test)
            
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            plt.xlabel('Actual Magnitude')
            plt.ylabel('Predicted Magnitude')
            plt.title(f'Actual vs Predicted Earthquake Magnitudes ({self.best_model_name})')
            
            # Add metrics to plot
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            plt.text(
                0.05, 0.95, 
                f'RMSE: {rmse:.3f}\nR²: {r2:.3f}',
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', alpha=0.1)
            )
            
            # Save plot
            os.makedirs('static/img', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f'static/img/model_performance_{timestamp}.png')
            plt.close()
            
            # If it's a tree-based model, save feature importance plot
            if hasattr(self.best_model, 'feature_importances_') and self.feature_names:
                self._save_feature_importance_plot()
                
        except Exception as e:
            logger.error(f"Error saving prediction plot: {e}")
    
    def _save_feature_importance_plot(self):
        """Save feature importance plot for tree-based models."""
        try:
            if not hasattr(self.best_model, 'feature_importances_'):
                return
                
            # Get feature importances
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Plot
            plt.figure(figsize=(12, 8))
            plt.title('Feature Importances')
            plt.bar(range(len(importances)), 
                    importances[indices], 
                    align='center')
            
            # Add feature names if available
            if self.feature_names and len(self.feature_names) == len(importances):
                plt.xticks(
                    range(len(importances)), 
                    [self.feature_names[i] for i in indices], 
                    rotation=90
                )
            
            plt.tight_layout()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig('static/img/feature_importance_{timestamp}.png')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error saving feature importance plot: {e}")
    
    def save_models(self):
        """
        Save all trained models and the best model with unique versioning.
        
        Returns:
            str: Path to the best model.
        """
        try:
            if not self.models:
                raise ValueError("No trained models available to save")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save all models with version tracking
            for name, model in self.models.items():
                # Check for existing versions
                existing_versions = [f for f in os.listdir(self.models_dir) 
                                   if f.startswith(f"{name}_v")]
                version = len(existing_versions) + 1
                
                model_path = os.path.join(self.models_dir, f"{name}_v{version}_{timestamp}.pkl")
                joblib.dump({
                    'model': model,
                    'name': name,
                    'version': version,
                    'timestamp': timestamp
                }, model_path)
                logger.info(f"Saved {name} model (v{version}) to {model_path}")
            
            # Save best model with version tracking
            if self.best_model:
                existing_best_versions = [f for f in os.listdir(self.models_dir) 
                                        if f.startswith("best_model_v")]
                best_version = len(existing_best_versions) + 1
                
                best_model_path = os.path.join(self.models_dir, f"best_model_v{best_version}.pkl")
                joblib.dump({
                    'model': self.best_model,
                    'name': self.best_model_name,
                    'version': best_version,
                    'timestamp': timestamp
                }, best_model_path)
                logger.info(f"Saved best model ({self.best_model_name}, v{best_version}) to {best_model_path}")
                return best_model_path
            else:
                logger.warning("No best model selected for saving")
                return None
                
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            raise
    
    def load_best_model(self):
        """
        Load the best trained model from file.
        
        Returns:
            object: The loaded model.
        """
        try:
            best_model_path = os.path.join(self.models_dir, "best_model.pkl")
            
            if not os.path.exists(best_model_path):
                logger.warning(f"Best model file not found at {best_model_path}")
                return None
                
            model_data = joblib.load(best_model_path)
            self.best_model = model_data['model']
            self.best_model_name = model_data['name']
            
            logger.info(f"Loaded best model ({self.best_model_name}) from {best_model_path}")
            
            return self.best_model
            
        except Exception as e:
            logger.error(f"Error loading best model: {e}")
            raise
    
    def predict(self, X):
        """
        Make prediction using the best model.
        
        Args:
            X (array-like): Input features.
            
        Returns:
            array: Predicted values.
        """
        try:
            if self.best_model is None:
                # Try to load the best model if not already loaded
                self.load_best_model()
                
                if self.best_model is None:
                    raise ValueError("No model available for prediction")
            
            # Make prediction
            prediction = self.best_model.predict(X)
            return prediction
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise