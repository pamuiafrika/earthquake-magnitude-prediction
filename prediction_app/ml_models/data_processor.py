import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import logging
from datetime import datetime
from sklearn.cluster import KMeans

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EarthquakeDataProcessor:
    """
    Class to process earthquake data for ML training and prediction.
    """
    def __init__(self, data_path=None, df=None):
        """
        Initialize the data processor with either a path to a CSV file or a DataFrame.
        
        Args:
            data_path (str, optional): Path to CSV file.
            df (DataFrame, optional): Pandas DataFrame with earthquake data.
        """
        self.scaler = StandardScaler()
        self.df = None
        self.regions = None
        self.feature_names = None  # Store feature names from training
        self.hotspot_count = 4  # Default number of hotspots for backward compatibility
        
        # Load data from either source
        if df is not None:
            self.df = df.copy()
        elif data_path and os.path.exists(data_path):
            try:
                self.df = pd.read_csv(data_path)
                logger.info(f"Data loaded successfully from {data_path}")
            except Exception as e:
                logger.error(f"Error loading data: {e}")
                raise
        else:
            logger.warning("No data provided!")
            
    def clean_data(self):
        """Clean and preprocess the earthquake data."""
        if self.df is None:
            raise ValueError("No data available for cleaning")
            
        try:
            # Convert date components to datetime
            self.df['Date'] = pd.to_datetime(self.df[['Year', 'Month', 'Day']])
            
            # Handle missing values
            for col in ['Lat', 'Lon', 'Depth']:
                if self.df[col].isna().sum() > 0:
                    # For geographical data, filling with mean can be reasonable
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
                    logger.info(f"Filled {self.df[col].isna().sum()} missing values in {col}")
            
            # Handle depth outliers if any (cap extreme values)
            depth_q1 = self.df['Depth'].quantile(0.25)
            depth_q3 = self.df['Depth'].quantile(0.75)
            depth_iqr = depth_q3 - depth_q1
            depth_upper = depth_q3 + 1.5 * depth_iqr
            self.df['Depth'] = np.where(
                self.df['Depth'] > depth_upper,
                depth_upper,
                self.df['Depth']
            )
            
            # Remove duplicates
            initial_rows = len(self.df)
            self.df.drop_duplicates(subset=['Date', 'Lat', 'Lon', 'Depth', 'Mag'], inplace=True)
            logger.info(f"Removed {initial_rows - len(self.df)} duplicate entries")
            
            # Sort by date
            self.df.sort_values('Date', inplace=True)
            
            return self.df
            
        except Exception as e:
            logger.error(f"Error during data cleaning: {e}")
            raise
    
    def create_features(self, for_prediction=False):
        """
        Create features for ML model training.
        
        Args:
            for_prediction (bool): Whether this is for making predictions
                                  (determines hotspot count to use)
        
        Returns:
            DataFrame: Processed DataFrame with features.
        """
        if self.df is None:
            raise ValueError("No data available for feature creation")
            
        try:
            # Create a copy to avoid modifying the original
            df_features = self.df.copy()
            
            # Temporal features
            df_features['DayOfWeek'] = df_features['Date'].dt.dayofweek
            df_features['DayOfYear'] = df_features['Date'].dt.dayofyear
            df_features['Month'] = df_features['Date'].dt.month
            
            # Create season (Tanzania seasons: dry/wet seasons instead of 4 seasons)
            # Simplified: Wet season (Nov-May), Dry season (Jun-Oct)
            df_features['Season'] = df_features['Month'].apply(
                lambda x: 1 if x >= 11 or x <= 5 else 0
            )
            
            # Define regions using KMeans clustering on lat/lon
            if self.regions is None:
                kmeans = KMeans(n_clusters=5, random_state=42)
                self.regions = kmeans.fit(df_features[['Lat', 'Lon']])
            
            df_features['Region'] = self.regions.predict(df_features[['Lat', 'Lon']])
            
            # Historical patterns (time-based features)
            df_features['TimeSinceLast'] = df_features.groupby('Region')['Date'].diff().dt.days
            # Fill NaN values for first entries in each region
            df_features['TimeSinceLast'].fillna(365, inplace=True)
            
            # Rolling average magnitude by region (last 3 earthquakes)
            df_features['RollingMagnitude'] = df_features.groupby('Region')['Mag'].transform(
                lambda x: x.rolling(3, min_periods=1).mean()
            )
            
            # Distance from known seismic hotspots in Tanzania
            # These are approximate coordinates of seismic active areas
            hotspots_all = [
                (-3.35, 35.88),  # Eyasi-Wembere rift
                (-6.76, 31.01),  # Rukwa-Livingstone rift zone
                (-3.07, 36.27),  # Kilimanjaro volcanic zone
                (-2.71, 35.91),  # Ngorongoro-Natron volcanic zone
                (-8.91, 33.45),  # Mbeya triple junction
                (-4.88, 29.67),  # Lake Tanganyika rift
                (-7.92, 31.62)   # Usangu basin
            ]
            
            # Use only the number of hotspots that matches with training
            # Unless we're creating a new model
            hotspots_to_use = hotspots_all
            if for_prediction and self.feature_names is not None:
                # Count how many hotspot features were used in training
                hotspot_feature_count = sum(1 for feature in self.feature_names if feature.startswith('DistanceToHotspot'))
                if hotspot_feature_count > 0:
                    hotspots_to_use = hotspots_all[:hotspot_feature_count]
                    logger.info(f"Using {hotspot_feature_count} hotspots for prediction to match training data")
            else:
                # For training or if we don't know the feature names yet
                self.hotspot_count = len(hotspots_all)
                hotspots_to_use = hotspots_all
            
            for i, (lat, lon) in enumerate(hotspots_to_use):
                df_features[f'DistanceToHotspot{i+1}'] = np.sqrt(
                    (df_features['Lat'] - lat)**2 + (df_features['Lon'] - lon)**2
                )
            
            # Calculate nearest previous earthquake distance
            df_features['PrevLat'] = df_features['Lat'].shift(1)
            df_features['PrevLon'] = df_features['Lon'].shift(1)
            df_features['DistanceToPrevious'] = np.sqrt(
                (df_features['Lat'] - df_features['PrevLat'])**2 + 
                (df_features['Lon'] - df_features['PrevLon'])**2
            )
            df_features['DistanceToPrevious'].fillna(0, inplace=True)
            
            # Drop unnecessary columns for modeling
            drop_cols = ['Date', 'PrevLat', 'PrevLon', 'Time']
            feature_df = df_features.drop(columns=drop_cols, errors='ignore')
            
            return feature_df
            
        except Exception as e:
            logger.error(f"Error during feature creation: {e}")
            raise
    
    def prepare_train_test_data(self, test_size=0.2, random_state=42):
        """
        Prepare training and testing datasets.
        
        Args:
            test_size (float): Proportion of data to use for testing.
            random_state (int): Random seed for reproducibility.
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, feature_names)
        """
        try:
            # Create features
            df_features = self.create_features(for_prediction=False)
            
            # Define target and features
            y = df_features['Mag']
            
            # Remove target variable from features
            feature_cols = [col for col in df_features.columns if col != 'Mag']
            X = df_features[feature_cols]
            
            # Store feature names for future reference
            self.feature_names = X.columns.tolist()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Scale the data
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            logger.info(f"Data split into {X_train.shape[0]} training samples and {X_test.shape[0]} testing samples")
            
            return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()
            
        except Exception as e:
            logger.error(f"Error preparing train/test data: {e}")
            raise
    
    def preprocess_for_prediction(self, lat, lon, depth, date=None):
        """
        Preprocess a single data point for prediction.
        
        Args:
            lat (float): Latitude.
            lon (float): Longitude.
            depth (float): Depth in km.
            date (datetime, optional): Date of prediction. Defaults to current date.
            
        Returns:
            array: Preprocessed features ready for prediction.
        """
        try:
            if date is None:
                date = datetime.now()
            
            # Create a DataFrame with the same structure as training data
            data = {
                'Lat': [lat],
                'Lon': [lon],
                'Depth': [depth],
                'Year': [date.year],
                'Month': [date.month],
                'Day': [date.day],
                'Date': [date]
            }
            
            df_pred = pd.DataFrame(data)
            
            # Merge with the original dataset temporarily to compute historical features
            temp_df = pd.concat([self.df, df_pred], ignore_index=True).sort_values('Date')
            temp_processor = EarthquakeDataProcessor(df=temp_df)
            temp_processor.regions = self.regions
            
            # Transfer feature names to temporary processor for feature compatibility
            temp_processor.feature_names = self.feature_names
            
            # Get features for the last row (our prediction point)
            features_df = temp_processor.create_features(for_prediction=True)
            pred_features = features_df.iloc[-1:].drop(columns=['Mag', 'Date'], errors='ignore')
            
            # Ensure columns match training data features
            if self.feature_names is not None:
                # Check for missing columns
                missing_cols = set(self.feature_names) - set(pred_features.columns)
                for col in missing_cols:
                    pred_features[col] = 0  # Add missing columns with default values
                
                # Remove extra columns
                extra_cols = set(pred_features.columns) - set(self.feature_names)
                if extra_cols:
                    pred_features = pred_features.drop(columns=list(extra_cols))
                
                # Ensure column order matches training data
                pred_features = pred_features[self.feature_names]
            
            # Scale features using the same scaler as training data
            scaled_features = self.scaler.transform(pred_features)
            
            return scaled_features
            
        except Exception as e:
            logger.error(f"Error preprocessing prediction data: {e}")
            raise
            
    def get_historical_data_for_region(self, lat, lon, radius=1.0):
        """
        Get historical earthquake data for a given region.
        
        Args:
            lat (float): Latitude of the center point.
            lon (float): Longitude of the center point.
            radius (float): Radius in degrees to search around the center point.
            
        Returns:
            DataFrame: Historical earthquake data for the region.
        """
        try:
            if self.df is None:
                raise ValueError("No data available")
                
            # Calculate distance from the given point
            distance = np.sqrt((self.df['Lat'] - lat)**2 + (self.df['Lon'] - lon)**2)
            
            # Filter data within the specified radius
            regional_data = self.df[distance <= radius].copy()
            
            logger.info(f"Found {len(regional_data)} historical earthquakes within {radius} degrees of ({lat}, {lon})")
            
            return regional_data
            
        except Exception as e:
            logger.error(f"Error retrieving historical data: {e}")
            raise
            
    def load_feature_names(self, feature_names):
        """
        Load the feature names from a trained model.
        
        Args:
            feature_names (list): List of feature names used during training.
        """
        self.feature_names = feature_names
        
        # Count hotspots based on loaded feature names
        hotspot_features = [f for f in feature_names if f.startswith('DistanceToHotspot')]
        if hotspot_features:
            self.hotspot_count = len(hotspot_features)
            logger.info(f"Detected {self.hotspot_count} hotspot features from loaded model")
            
            
