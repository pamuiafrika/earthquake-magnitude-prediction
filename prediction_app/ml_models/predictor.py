import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import pickle
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from .data_processor import EarthquakeDataProcessor
from .model_trainer import EarthquakeModelTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EarthquakePredictionService:
    """
    Service for earthquake magnitude prediction and analytics.
    """
    def __init__(self, data_path=None, models_dir='models'):
        """
        Initialize the prediction service.
        
        Args:
            data_path (str, optional): Path to earthquake data CSV.
            models_dir (str): Directory containing trained models.
        """
        self.data_path = data_path
        self.models_dir = models_dir
        self.data_processor = None
        self.model_trainer = None
        self.scaler_fitted = False
        self.feature_names = None
        
        # Initialize components if data path is provided
        if data_path:
            self.initialize_components()
    
    def initialize_components(self, data_path=None):
        """
        Initialize data processor and model trainer.
        
        Args:
            data_path (str, optional): Path to earthquake data CSV.
        """
        try:
            if data_path:
                self.data_path = data_path
                
            if not self.data_path:
                raise ValueError("No data path provided")
                
            # Initialize data processor
            self.data_processor = EarthquakeDataProcessor(self.data_path)
            self.data_processor.clean_data()
            
            # Load saved scaler if exists
            scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
            if os.path.exists(scaler_path):
                try:
                    self.data_processor.scaler = joblib.load(scaler_path)
                    # Verify scaler is fitted by checking if it has mean_ attribute
                    if hasattr(self.data_processor.scaler, 'mean_'):
                        self.scaler_fitted = True
                        logger.info("Loaded fitted scaler from disk")
                    else:
                        logger.warning("Loaded scaler does not appear to be fitted")
                except Exception as e:
                    logger.error(f"Error loading scaler: {e}")
            
            # Try to load feature names
            feature_names_path = os.path.join(self.models_dir, 'feature_names.pkl')
            if os.path.exists(feature_names_path):
                try:
                    self.feature_names = joblib.load(feature_names_path)
                    # Pass feature names to data processor for compatibility
                    self.data_processor.load_feature_names(self.feature_names)
                    logger.info(f"Loaded {len(self.feature_names)} feature names from disk")
                except Exception as e:
                    logger.error(f"Error loading feature names: {e}")
            
            # Initialize model trainer
            self.model_trainer = EarthquakeModelTrainer(self.models_dir)
            
            # Try to load pre-trained model
            self.model_trainer.load_best_model()
            
            logger.info("Prediction service components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing prediction service: {e}")
            raise
    
    def train_new_model(self, test_size=0.2, random_state=42):
        """
        Train a new ML model using the provided data.
        
        Args:
            test_size (float): Proportion of data to use for testing.
            random_state (int): Random seed for reproducibility.
            
        Returns:
            dict: Evaluation metrics for trained models.
        """
        try:
            if self.data_processor is None:
                raise ValueError("Data processor not initialized")
                
            # Prepare data
            X_train, X_test, y_train, y_test, feature_names = \
                self.data_processor.prepare_train_test_data(test_size, random_state)
                
            # Store feature names
            self.feature_names = feature_names
            
            # Save feature names for future use
            feature_names_path = os.path.join(self.models_dir, 'feature_names.pkl')
            os.makedirs(self.models_dir, exist_ok=True)
            joblib.dump(self.feature_names, feature_names_path)
            logger.info(f"Saved {len(self.feature_names)} feature names")
            
            # At this point, scaler has been fitted during prepare_train_test_data
            self.scaler_fitted = True
            
            # Train models
            self.model_trainer.train_models(X_train, y_train, feature_names)
            
            # Save scaler state from data processor
            scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
            joblib.dump(self.data_processor.scaler, scaler_path)
            
            # Evaluate models
            results = self.model_trainer.evaluate_models(X_test, y_test)
            
            # Save models
            self.model_trainer.save_models()
            
            return results
            
        except Exception as e:
            logger.error(f"Error training new model: {e}")
            raise
    
    def predict_magnitude(self, lat, lon, depth, date=None):
        """
        Predict earthquake magnitude for given parameters.
        
        Args:
            lat (float): Latitude.
            lon (float): Longitude.
            depth (float): Depth in km.
            date (datetime, optional): Date for prediction. Defaults to current date.
            
        Returns:
            float: Predicted earthquake magnitude.
        """
        try:
            if self.data_processor is None or self.model_trainer is None:
                raise ValueError("Prediction service not properly initialized")
                
            # Load model if not already loaded
            if self.model_trainer.best_model is None:
                self.model_trainer.load_best_model()
                if self.model_trainer.best_model is None:
                    raise ValueError("No trained model available for prediction")
            
            # Check if scaler is fitted
            if not self.scaler_fitted:
                logger.warning("Scaler not fitted. Training temporary scaler on available data.")
                # Fit the scaler on available data
                self.ensure_scaler_is_fitted()
            
            # Make sure the data processor has feature names loaded
            if self.feature_names and not self.data_processor.feature_names:
                self.data_processor.load_feature_names(self.feature_names)
            
            # Preprocess input data
            X = self.data_processor.preprocess_for_prediction(lat, lon, depth, date)
            
            # Make prediction
            magnitude = self.model_trainer.predict(X)[0]
            
            logger.info(f"Predicted magnitude {magnitude:.2f} for location ({lat}, {lon}) at depth {depth} km")
            
            return magnitude
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    def ensure_scaler_is_fitted(self):
        """
        Ensure that the scaler is fitted by creating features and fitting if needed.
        """
        try:
            # Create features from the available data
            df_features = self.data_processor.create_features(for_prediction=False)
            
            # Define features (excluding target)
            feature_cols = [col for col in df_features.columns if col != 'Mag']
            X = df_features[feature_cols]
            
            # If we have feature names from a saved model, align columns
            if self.feature_names:
                # Add missing columns
                missing_cols = set(self.feature_names) - set(X.columns)
                for col in missing_cols:
                    X[col] = 0  # Add missing columns with default values
                
                # Remove extra columns
                extra_cols = set(X.columns) - set(self.feature_names)
                if extra_cols:
                    X = X.drop(columns=list(extra_cols))
                
                # Ensure column order matches training data
                X = X[self.feature_names]
            
            # Fit the scaler on all available data
            self.data_processor.scaler.fit(X)
            self.scaler_fitted = True
            
            # Save the newly fitted scaler
            scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
            os.makedirs(self.models_dir, exist_ok=True)
            joblib.dump(self.data_processor.scaler, scaler_path)
            
            logger.info("Scaler fitted on available data and saved")
            
        except Exception as e:
            logger.error(f"Error fitting scaler: {e}")
            raise
    
    def get_regional_analysis(self, lat, lon, radius=1.0):
        """
        Get historical earthquake analysis for a specific region.
        
        Args:
            lat (float): Latitude of the center point.
            lon (float): Longitude of the center point.
            radius (float): Radius in degrees to search around the center point.
            
        Returns:
            dict: Analysis results including historical data and statistics.
        """
        try:
            if self.data_processor is None:
                raise ValueError("Data processor not initialized")
                
            # Get historical data for the region
            regional_data = self.data_processor.get_historical_data_for_region(lat, lon, radius)
            
            if len(regional_data) == 0:
                return {
                    'count': 0,
                    'message': f"No historical earthquake data found within {radius} degrees of ({lat}, {lon})",
                    'data': None
                }
            
            # Calculate statistics
            stats = {
                'count': len(regional_data),
                'avg_magnitude': regional_data['Mag'].mean(),
                'max_magnitude': regional_data['Mag'].max(),
                'min_magnitude': regional_data['Mag'].min(),
                'std_magnitude': regional_data['Mag'].std(),
                'recent_events': len(regional_data[regional_data['Date'].dt.year >= datetime.now().year - 5]),
                'by_year': regional_data.groupby(regional_data['Date'].dt.year)['Mag'].agg(['count', 'mean']).to_dict(),
                'by_depth': self._analyze_by_depth(regional_data)
            }
            
            return {
                'stats': stats,
                'data': regional_data
            }
            
        except Exception as e:
            logger.error(f"Error generating regional analysis: {e}")
            raise
    
    def _analyze_by_depth(self, data):
        """
        Analyze earthquake data by depth ranges.
        
        Args:
            data (DataFrame): Earthquake data.
            
        Returns:
            dict: Statistics by depth range.
        """
        try:
            # Define depth ranges (in km)
            depth_ranges = [
                (0, 10, 'Shallow (0-10km)'),
                (10, 30, 'Intermediate (10-30km)'),
                (30, float('inf'), 'Deep (>30km)')
            ]
            
            results = {}
            
            for min_depth, max_depth, label in depth_ranges:
                mask = (data['Depth'] >= min_depth) & (data['Depth'] < max_depth)
                subset = data[mask]
                
                if len(subset) > 0:
                    results[label] = {
                        'count': len(subset),
                        'avg_magnitude': subset['Mag'].mean(),
                        'max_magnitude': subset['Mag'].max()
                    }
                else:
                    results[label] = {
                        'count': 0,
                        'avg_magnitude': 0,
                        'max_magnitude': 0
                    }
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing by depth: {e}")
            raise
    
    def generate_heatmap(self, lat=None, lon=None, radius=None):
        """
        Generate earthquake heatmap data for visualization.
        
        Args:
            lat (float, optional): Center latitude.
            lon (float, optional): Center longitude.
            radius (float, optional): Radius in degrees.
            
        Returns:
            dict: Heatmap data.
        """
        try:
            if self.data_processor is None:
                raise ValueError("Data processor not initialized")
            
            df = self.data_processor.df
            
            # Filter by region if specified
            if lat is not None and lon is not None and radius is not None:
                distance = np.sqrt((df['Lat'] - lat)**2 + (df['Lon'] - lon)**2)
                df = df[distance <= radius].copy()
            
            # Prepare data for heatmap
            heatmap_data = df[['Lat', 'Lon', 'Mag']].values.tolist()
            
            return {
                'data': heatmap_data,
                'center': {
                    'lat': lat if lat is not None else df['Lat'].mean(),
                    'lon': lon if lon is not None else df['Lon'].mean()
                },
                'count': len(heatmap_data)
            }
            
        except Exception as e:
            logger.error(f"Error generating heatmap: {e}")
            raise
    
    def get_time_series_analysis(self):
        """
        Perform time series analysis of earthquake data.
        
        Returns:
            dict: Time series analysis results.
        """
        try:
            if self.data_processor is None:
                raise ValueError("Data processor not initialized")
                
            df = self.data_processor.df
            
            # Add Year and YearMonth columns
            df['Year'] = df['Date'].dt.year
            df['YearMonth'] = df['Date'].dt.strftime('%Y-%m')
            monthly = df.groupby('YearMonth').agg({
                'Mag': ['count', 'mean', 'max', 'min']
            }).reset_index()
            
            # Create time series data
            time_series = {
                'labels': monthly['YearMonth'].tolist(),
                'count': monthly[('Mag', 'count')].tolist(),
                'avg_magnitude': monthly[('Mag', 'mean')].tolist(),
                'max_magnitude': monthly[('Mag', 'max')].tolist()
            }
            
            # Yearly trends
            yearly = df.groupby('Year').agg({
                'Mag': ['count', 'mean', 'max']
            }).reset_index()
            
            # Create yearly trend data
            yearly_trends = {
                'years': yearly['Year'].tolist(),
                'count': yearly[('Mag', 'count')].tolist(),
                'avg_magnitude': yearly[('Mag', 'mean')].tolist(),
                'max_magnitude': yearly[('Mag', 'max')].tolist()
            }
            
            return {
                'time_series': time_series,
                'yearly_trends': yearly_trends
            }
            
        except Exception as e:
            logger.error(f"Error in time series analysis: {e}")
            raise