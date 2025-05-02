import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
import warnings

warnings.filterwarnings('ignore')

class EarthquakePredictionSystem:
    def __init__(self):
        """Initialize the Earthquake Prediction System"""
        self.data = None
        self.processed_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.models = {}
        self.ensemble_weights = None
        self.scaler = StandardScaler()
        self.seq_scaler = StandardScaler()
        self.spatial_clusters = None
        self.high_activity_zones = None
        self.time_sequence = None
        
    def load_data(self, file_path):
        """
        Load earthquake data from CSV file
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pandas.DataFrame: Loaded data
        """
        try:
            self.data = pd.read_csv(file_path)
            print(f"Data loaded successfully with {self.data.shape[0]} records and {self.data.shape[1]} columns")
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
            
    def preprocess_data(self):
        """
        Preprocess the earthquake data:
        - Convert time columns to datetime
        - Handle missing values
        - Sort chronologically
        - Create basic datetime features
        
        Returns:
            pandas.DataFrame: Preprocessed data
        """
        if self.data is None:
            print("No data to preprocess. Please load data first.")
            return None
            
        # Make a copy to avoid modifying the original data
        df = self.data.copy()
        
        # Convert date and time columns to datetime
        df['DateTime'] = pd.to_datetime(df[['Year', 'Month', 'Day']].astype(str).agg('-'.join, axis=1) + ' ' + df['Time'])
        
        # Handle missing values
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if col in ['Lat', 'Lon', 'Depth', 'Mag']:
                    # For important numeric columns, use median
                    df[col] = df[col].fillna(df[col].median())
                else:
                    # For other columns, use forward fill
                    df[col] = df[col].fillna(method='ffill')
        
        # Sort chronologically
        df = df.sort_values('DateTime')
        
        # Extract basic datetime components
        df['Year_num'] = df['Year']
        df['Month_num'] = df['Month']
        df['Day_num'] = df['Day']
        df['Hour'] = df['DateTime'].dt.hour
        df['Minute'] = df['DateTime'].dt.minute
        
        # Store the processed data
        self.processed_data = df
        
        return df
        
    def identify_spatial_regions(self, n_clusters=5):
        """
        Identify spatial regions using clustering
        
        Args:
            n_clusters (int): Number of clusters to create
            
        Returns:
            pandas.DataFrame: Data with region labels
        """
        if self.processed_data is None:
            print("No processed data available. Please preprocess data first.")
            return None
            
        # Extract spatial coordinates
        coords = self.processed_data[['Lat', 'Lon', 'Depth']].values
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.processed_data['Region'] = kmeans.fit_predict(coords)
        
        # Store the cluster model for later use
        self.spatial_clusters = kmeans
        
        # Identify high activity zones (regions with highest earthquake counts)
        region_counts = self.processed_data['Region'].value_counts()
        self.high_activity_zones = region_counts.index[:3].tolist()
        
        return self.processed_data
        
    def engineer_features(self):
        """
        Perform feature engineering:
        - Spatial features
        - Temporal features
        - Combined features
        
        Returns:
            pandas.DataFrame: Data with engineered features
        """
        if self.processed_data is None:
            print("No processed data available. Please preprocess data first.")
            return None
            
        df = self.processed_data.copy()
        
        # Spatial Features
        # Distance from region centroids
        centroids = self.spatial_clusters.cluster_centers_
        for i, centroid in enumerate(centroids):
            df[f'Dist_to_region_{i}'] = np.sqrt(
                (df['Lat'] - centroid[0])**2 + 
                (df['Lon'] - centroid[1])**2 + 
                (df['Depth'] - centroid[2])**2
            )
        
        # Distance to high activity zones (top 3 regions by frequency)
        for i, region_id in enumerate(self.high_activity_zones):
            centroid = centroids[region_id]
            df[f'Dist_to_high_activity_{i}'] = np.sqrt(
                (df['Lat'] - centroid[0])**2 + 
                (df['Lon'] - centroid[1])**2 + 
                (df['Depth'] - centroid[2])**2
            )
        
        # Spatial density (number of earthquakes within 1 degree)
        def calculate_spatial_density(row):
            nearby = df[(abs(df['Lat'] - row['Lat']) < 1) & 
                        (abs(df['Lon'] - row['Lon']) < 1) & 
                        (df['DateTime'] < row['DateTime'])]
            return len(nearby)
            
        df['Spatial_density'] = df.apply(calculate_spatial_density, axis=1)
        
        # Temporal Features
        # Cyclical encoding of time components
        for col in ['Month_num', 'Day_num', 'Hour']:
            max_val = {'Month_num': 12, 'Day_num': 31, 'Hour': 24}[col]
            df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
            df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)
        
        # Time since last earthquake (overall)
        df['Time_since_last'] = df['DateTime'].diff().dt.total_seconds() / 3600  # in hours
        df['Time_since_last'] = df['Time_since_last'].fillna(0)
        
        # Time since last earthquake by region
        for region in df['Region'].unique():
            region_df = df[df['Region'] == region]
            df.loc[df['Region'] == region, 'Time_since_last_in_region'] = region_df['DateTime'].diff().dt.total_seconds() / 3600
        df['Time_since_last_in_region'] = df['Time_since_last_in_region'].fillna(0)
        
        # Historical patterns - rolling statistics
        # Overall patterns
        df['Mag_rolling_mean_5'] = df['Mag'].rolling(window=5, min_periods=1).mean()
        df['Mag_rolling_std_5'] = df['Mag'].rolling(window=5, min_periods=1).std().fillna(0)
        df['Mag_rolling_max_5'] = df['Mag'].rolling(window=5, min_periods=1).max()
        
        # Combined Features
        # Region-specific historical patterns
        for region in df['Region'].unique():
            region_mask = df['Region'] == region
            df.loc[region_mask, 'Region_Mag_mean'] = df.loc[region_mask, 'Mag'].rolling(window=3, min_periods=1).mean()
            df.loc[region_mask, 'Region_Mag_std'] = df.loc[region_mask, 'Mag'].rolling(window=3, min_periods=1).std().fillna(0)
        
        # Calculate regional frequency
        region_counts = df['Region'].value_counts()
        df['Region_frequency'] = df['Region'].map(region_counts)
        
        # Calculate energy release patterns (energy ~ 10^(1.5*magnitude))
        df['Energy'] = 10 ** (1.5 * df['Mag'])
        df['Energy_rolling_sum'] = df['Energy'].rolling(window=5, min_periods=1).sum()
        
        # Region-specific energy release
        for region in df['Region'].unique():
            region_mask = df['Region'] == region
            df.loc[region_mask, 'Region_energy_sum'] = df.loc[region_mask, 'Energy'].rolling(window=3, min_periods=1).sum()
        
        # Drop original datetime columns and create dummy variables for Region
        df = pd.get_dummies(df, columns=['Region'], prefix='Region')
        
        # Store the processed data with features
        self.processed_data = df
        
        return df
        
    def prepare_for_training(self, test_size=0.2, temporal_split=True):
        """
        Prepare data for model training
        
        Args:
            test_size (float): Portion of data to use for testing
            temporal_split (bool): Whether to split data temporally
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        if self.processed_data is None:
            print("No processed data available. Please engineer features first.")
            return None
            
        df = self.processed_data.copy()
        
        # Define features to use for training
        exclude_cols = ['Year', 'Month', 'Day', 'Time', 'DateTime', 'Mag', 'Energy']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Store feature names for later use
        self.feature_names = feature_cols
        
        # Define X and y
        X = df[feature_cols]
        y = df['Mag']
        
        # Split data
        if temporal_split:
            # Temporal split (respecting time order)
            train_size = int(len(df) * (1 - test_size))
            self.X_train = X.iloc[:train_size]
            self.y_train = y.iloc[:train_size]
            self.X_test = X.iloc[train_size:]
            self.y_test = y.iloc[train_size:]
        else:
            # Random split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Prepare sequence data for LSTM/GRU (using last 10 events to predict next)
        self.prepare_sequence_data(seq_length=10)
        
        return self.X_train, self.X_test, self.y_train, self.y_test
        
    def prepare_sequence_data(self, seq_length=10):
        """
        Prepare sequential data for LSTM/GRU models
        
        Args:
            seq_length (int): Length of sequences to create
            
        Returns:
            tuple: (X_seq_train, X_seq_test, y_seq_train, y_seq_test)
        """
        # Use scaled data
        X_scaled = self.scaler.transform(self.processed_data[self.feature_names])
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(len(X_scaled) - seq_length):
            X_sequences.append(X_scaled[i:i+seq_length])
            y_sequences.append(self.processed_data['Mag'].iloc[i+seq_length])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        # Split into train and test sets
        train_size = len(self.X_train)
        seq_train_size = train_size - seq_length
        
        self.X_seq_train = X_sequences[:seq_train_size]
        self.y_seq_train = y_sequences[:seq_train_size]
        self.X_seq_test = X_sequences[seq_train_size:]
        self.y_seq_test = y_sequences[seq_train_size:]
        
        return self.X_seq_train, self.X_seq_test, self.y_seq_train, self.y_seq_test
        
    def train_random_forest(self):
        """
        Train a Random Forest model
        
        Returns:
            object: Trained model
        """
        print("Training Random Forest model...")
        
        # Define parameter grid for grid search
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        
        # Create base model
        rf = RandomForestRegressor(random_state=42)
        
        # Use TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        # Fit the model
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        # Get best model
        best_rf = grid_search.best_estimator_
        
        # Store model
        self.models['random_forest'] = best_rf
        
        # Evaluate
        predictions = best_rf.predict(self.X_test_scaled)
        rmse = np.sqrt(mean_squared_error(self.y_test, predictions))
        mae = mean_absolute_error(self.y_test, predictions)
        r2 = r2_score(self.y_test, predictions)
        
        print(f"Random Forest - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        print(f"Best Parameters: {grid_search.best_params_}")
        
        return best_rf
        
    def train_gradient_boosting(self):
        """
        Train a Gradient Boosting model
        
        Returns:
            object: Trained model
        """
        print("Training Gradient Boosting model...")
        
        # Define parameter grid for grid search
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
            'min_samples_split': [2, 5]
        }
        
        # Create base model
        gb = GradientBoostingRegressor(random_state=42)
        
        # Use TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=gb,
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        # Fit the model
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        # Get best model
        best_gb = grid_search.best_estimator_
        
        # Store model
        self.models['gradient_boosting'] = best_gb
        
        # Evaluate
        predictions = best_gb.predict(self.X_test_scaled)
        rmse = np.sqrt(mean_squared_error(self.y_test, predictions))
        mae = mean_absolute_error(self.y_test, predictions)
        r2 = r2_score(self.y_test, predictions)
        
        print(f"Gradient Boosting - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        print(f"Best Parameters: {grid_search.best_params_}")
        
        return best_gb
        
    def train_lstm_model(self):
        """
        Train an LSTM model for sequential prediction
        
        Returns:
            object: Trained model
        """
        print("Training LSTM model...")
        
        # Create LSTM model
        input_shape = (self.X_seq_train.shape[1], self.X_seq_train.shape[2])
        
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        # Compile model
        model.compile(optimizer='adam', loss='mse')
        
        # Train model
        history = model.fit(
            self.X_seq_train, self.y_seq_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
            ]
        )
        
        # Store model
        self.models['lstm'] = model
        
        # Evaluate
        predictions = model.predict(self.X_seq_test)
        rmse = np.sqrt(mean_squared_error(self.y_seq_test, predictions))
        mae = mean_absolute_error(self.y_seq_test, predictions)
        r2 = r2_score(self.y_seq_test, predictions)
        
        print(f"LSTM - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        return model
        
    def train_gru_model(self):
        """
        Train a GRU model for sequential prediction
        
        Returns:
            object: Trained model
        """
        print("Training GRU model...")
        
        # Create GRU model
        input_shape = (self.X_seq_train.shape[1], self.X_seq_train.shape[2])
        
        model = Sequential([
            GRU(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            GRU(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        # Compile model
        model.compile(optimizer='adam', loss='mse')
        
        # Train model
        history = model.fit(
            self.X_seq_train, self.y_seq_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
            ]
        )
        
        # Store model
        self.models['gru'] = model
        
        # Evaluate
        predictions = model.predict(self.X_seq_test)
        rmse = np.sqrt(mean_squared_error(self.y_seq_test, predictions))
        mae = mean_absolute_error(self.y_seq_test, predictions)
        r2 = r2_score(self.y_seq_test, predictions)
        
        print(f"GRU - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        return model
        
    def train_gaussian_process(self):
        """
        Train a Gaussian Process model for spatial interpolation
        
        Returns:
            object: Trained model
        """
        print("Training Gaussian Process model...")
        
        # Use only spatial features for this model
        spatial_features = [col for col in self.feature_names if any(s in col for s in ['Lat', 'Lon', 'Depth', 'Dist'])]
        
        X_spatial_train = self.X_train[spatial_features].values
        X_spatial_test = self.X_test[spatial_features].values
        
        # Scale features
        X_spatial_train_scaled = self.scaler.fit_transform(X_spatial_train)
        X_spatial_test_scaled = self.scaler.transform(X_spatial_test)
        
        # Define kernel
        kernel = 1.0 * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=0.1)
        
        # Create and train model
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            random_state=42
        )
        
        # Use a subsample for training to reduce computational burden
        sample_size = min(500, len(X_spatial_train_scaled))
        sample_indices = np.random.choice(len(X_spatial_train_scaled), sample_size, replace=False)
        
        gp.fit(X_spatial_train_scaled[sample_indices], self.y_train.iloc[sample_indices])
        
        # Store model
        self.models['gaussian_process'] = gp
        
        # Evaluate
        predictions, std_dev = gp.predict(X_spatial_test_scaled, return_std=True)
        rmse = np.sqrt(mean_squared_error(self.y_test, predictions))
        mae = mean_absolute_error(self.y_test, predictions)
        r2 = r2_score(self.y_test, predictions)
        
        print(f"Gaussian Process - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        return gp, std_dev
        
    def create_ensemble(self):
        """
        Create an ensemble model using weighted averaging
        
        Returns:
            dict: Ensemble weights and performance metrics
        """
        print("Creating ensemble model...")
        
        # Get predictions from all models
        predictions = {}
        
        # Traditional models
        for model_name in ['random_forest', 'gradient_boosting']:
            if model_name in self.models:
                predictions[model_name] = self.models[model_name].predict(self.X_test_scaled)
        
        # Gaussian Process
        if 'gaussian_process' in self.models:
            spatial_features = [col for col in self.feature_names if any(s in col for s in ['Lat', 'Lon', 'Depth', 'Dist'])]
            spatial_indices = [self.feature_names.index(col) for col in spatial_features]
            gp_pred = self.models['gaussian_process'].predict(self.X_test_scaled[:, spatial_indices])
            if isinstance(gp_pred, tuple):
                gp_pred = gp_pred[0]  # Extract predictions if model returns (predictions, std)
            predictions['gaussian_process'] = gp_pred
            
        print("Creating ensemble model...gausian p")
        
        # Sequential models
        for model_name in ['lstm', 'gru']:
            if model_name in self.models:
                seq_pred = self.models[model_name].predict(self.X_seq_test)
                # Match length with other predictions
                if len(seq_pred) < len(self.y_test):
                    # Pad with NaN
                    padding = np.full(len(self.y_test) - len(seq_pred), np.nan)
                    seq_pred = np.append(seq_pred, padding)
                predictions[model_name] = seq_pred.flatten()
        
        print("Creating ensemble model..sequential.")
        # Create DataFrame of predictions
        pred_df = pd.DataFrame(predictions)
        
        # Calculate RMSE for each model
        rmse_scores = {}
        y_test_index = np.arange(len(self.y_test))
        for model_name in pred_df.columns:
            # Skip NaN values
            mask = ~np.isnan(pred_df[model_name])
            if np.sum(mask) > 0:
                # Use the same indices for both arrays
                valid_indices = y_test_index[mask]
                rmse = np.sqrt(mean_squared_error(
                    self.y_test.iloc[valid_indices],
                    pred_df[model_name].iloc[valid_indices]
                ))
                rmse_scores[model_name] = rmse
        print("Creating ensemble model... rmse")
        # Calculate weights (inversely proportional to RMSE)
        weights = {model: 1/score for model, score in rmse_scores.items()}
        sum_weights = sum(weights.values())
        normalized_weights = {model: weight/sum_weights for model, weight in weights.items()}
        
        # Initialize arrays for ensemble predictions
        ensemble_pred = np.zeros(len(self.y_test))
        weight_sum = np.zeros(len(self.y_test))
        
        # Calculate weighted predictions
        for model_name, weight in normalized_weights.items():
            mask = ~np.isnan(pred_df[model_name])
            ensemble_pred[mask] += weight * pred_df[model_name][mask]
            weight_sum[mask] += weight
        
        # Normalize by weight sum
        ensemble_pred = ensemble_pred / weight_sum
        
        # Evaluate ensemble
        rmse = np.sqrt(mean_squared_error(self.y_test, ensemble_pred))
        mae = mean_absolute_error(self.y_test, ensemble_pred)
        r2 = r2_score(self.y_test, ensemble_pred)
        
        print(f"Ensemble - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        print(f"Ensemble weights: {normalized_weights}")
        
        # Store ensemble performance
        ensemble_results = {
            'weights': normalized_weights,
            'predictions': ensemble_pred,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        self.ensemble_results = ensemble_results
        
        return ensemble_results
        
    def predict_magnitude(self, latitude, longitude, depth, year=None, month=None, day=None, hour=None):
        """
        Predict earthquake magnitude for given location and time
        
        Args:
            latitude (float): Latitude coordinate
            longitude (float): Longitude coordinate
            depth (float): Depth in km
            year (int, optional): Year
            month (int, optional): Month
            day (int, optional): Day
            hour (int, optional): Hour
            
        Returns:
            tuple: (predicted_magnitude, confidence_interval)
        """
        # Use current date/time if not provided
        if year is None:
            current_dt = datetime.now()
            year = current_dt.year
            month = month or current_dt.month
            day = day or current_dt.day
            hour = hour or current_dt.hour
        
        # Create input features
        input_data = pd.DataFrame({
            'Lat': [latitude],
            'Lon': [longitude],
            'Depth': [depth],
            'Year_num': [year],
            'Month_num': [month],
            'Day_num': [day],
            'Hour': [hour],
        })
        
        # Cyclical encoding of time components
        input_data['Month_num_sin'] = np.sin(2 * np.pi * input_data['Month_num'] / 12)
        input_data['Month_num_cos'] = np.cos(2 * np.pi * input_data['Month_num'] / 12)
        input_data['Day_num_sin'] = np.sin(2 * np.pi * input_data['Day_num'] / 31)
        input_data['Day_num_cos'] = np.cos(2 * np.pi * input_data['Day_num'] / 31)
        input_data['Hour_sin'] = np.sin(2 * np.pi * input_data['Hour'] / 24)
        input_data['Hour_cos'] = np.cos(2 * np.pi * input_data['Hour'] / 24)
        
        # Assign to region
        coords = np.array([[latitude, longitude, depth]])
        region = self.spatial_clusters.predict(coords)[0]
        
        # Create one-hot encoding for region
        for i in range(len(self.spatial_clusters.cluster_centers_)):
            input_data[f'Region_Region_{i}'] = 1 if i == region else 0
        
        # Calculate distance to region centroids
        centroids = self.spatial_clusters.cluster_centers_
        for i, centroid in enumerate(centroids):
            input_data[f'Dist_to_region_{i}'] = np.sqrt(
                (input_data['Lat'] - centroid[0])**2 + 
                (input_data['Lon'] - centroid[1])**2 + 
                (input_data['Depth'] - centroid[2])**2
            )
        
        # Calculate distance to high activity zones
        for i, region_id in enumerate(self.high_activity_zones):
            centroid = centroids[region_id]
            input_data[f'Dist_to_high_activity_{i}'] = np.sqrt(
                (input_data['Lat'] - centroid[0])**2 + 
                (input_data['Lon'] - centroid[1])**2 + 
                (input_data['Depth'] - centroid[2])**2
            )
        
        # Fill remaining features with median values from training data
        for feature in self.feature_names:
            if feature not in input_data.columns:
                input_data[feature] = self.X_train[feature].median()
        
        # Ensure all features are in the correct order
        input_data = input_data[self.feature_names]
        
        # Scale features
        input_scaled = self.scaler.transform(input_data)
        
        # Get predictions from all models
        predictions = {}
        uncertainties = {}
        
        # Random Forest
        if 'random_forest' in self.models:
            rf_pred = self.models['random_forest'].predict(input_scaled)
            # Get prediction intervals using quantiles from tree predictions
            rf_preds_all = np.array([tree.predict(input_scaled) for tree in self.models['random_forest'].estimators_])
            rf_lower = np.percentile(rf_preds_all, 5)
            rf_upper = np.percentile(rf_preds_all, 95)
            predictions['random_forest'] = rf_pred[0]
            uncertainties['random_forest'] = (rf_upper - rf_lower) / 2
        
        # Gradient Boosting
        if 'gradient_boosting' in self.models:
            gb_pred = self.models['gradient_boosting'].predict(input_scaled)
            predictions['gradient_boosting'] = gb_pred[0]
            # Use training error for uncertainty estimate
            gb_errors = np.abs(self.models['gradient_boosting'].predict(self.X_train_scaled) - self.y_train)
            uncertainties['gradient_boosting'] = np.std(gb_errors) * 1.96  # 95% confidence interval
        
        # Gaussian Process
        if 'gaussian_process' in self.models:
            # Extract spatial features
            spatial_features = [col for col in self.feature_names if any(s in col for s in ['Lat', 'Lon', 'Depth', 'Dist'])]
            spatial_indices = [self.feature_names.index(col) for col in spatial_features]
            input_spatial = input_scaled[:, spatial_indices]
            
            gp_pred, gp_std = self.models['gaussian_process'].predict(input_spatial, return_std=True)
            predictions['gaussian_process'] = gp_pred[0]
            uncertainties['gaussian_process'] = gp_std[0] * 1.96  # 95% confidence interval
        
        # LSTM and GRU need sequence data
        # For simplicity, we'll use recent data points from the training set
        if 'lstm' in self.models or 'gru' in self.models:
            # Create a sequence using the most recent data points plus the new input
            recent_data = self.X_train_scaled[-9:]  # Get last 9 points from training
            seq_input = np.vstack([recent_data, input_scaled])
            seq_input = seq_input.reshape(1, 10, -1)  # Reshape to (1, seq_length, n_features)
            
            if 'lstm' in self.models:
                lstm_pred = self.models['lstm'].predict(seq_input)
                predictions['lstm'] = lstm_pred[0][0]
                # Use training error for uncertainty
                lstm_errors = np.abs(self.models['lstm'].predict(self.X_seq_train) - self.y_seq_train)
                uncertainties['lstm'] = np.std(lstm_errors) * 1.96
            
            if 'gru' in self.models:
                gru_pred = self.models['gru'].predict(seq_input)
                predictions['gru'] = gru_pred[0][0]
                # Use training error for uncertainty
                gru_errors = np.abs(self.models['gru'].predict(self.X_seq_train) - self.y_seq_train)
                uncertainties['gru'] = np.std(gru_errors) * 1.96
        
        # Calculate ensemble prediction
        ensemble_pred = 0
        ensemble_weight_sum = 0
        
        for model_name, weight in self.ensemble_weights.items():
            if model_name in predictions:
                ensemble_pred += weight * predictions[model_name]
                ensemble_weight_sum += weight
        
        if ensemble_weight_sum > 0:
            ensemble_pred /= ensemble_weight_sum
        
        # Calculate ensemble uncertainty (weighted average of individual uncertainties)
        ensemble_uncertainty = 0
        for model_name, weight in self.ensemble_weights.items():
            if model_name in uncertainties:
                ensemble_uncertainty += weight * uncertainties[model_name]
        
        if ensemble_weight_sum > 0:
            ensemble_uncertainty /= ensemble_weight_sum
        
        # Create confidence interval
        confidence_interval = (ensemble_pred - ensemble_uncertainty, ensemble_pred + ensemble_uncertainty)
        
        return ensemble_pred, confidence_interval, predictions, uncertainties
    
    def visualize_predictions(self, test_only=True):
        """
        Visualize actual vs predicted magnitudes
        
        Args:
            test_only (bool): Whether to visualize only test data
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        # Get ensemble predictions for test data
        ensemble_pred = self.ensemble_results['predictions']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot actual vs predicted
        ax.scatter(self.y_test, ensemble_pred, alpha=0.7)
        
        # Add perfect prediction line
        min_val = min(self.y_test.min(), ensemble_pred.min())
        max_val = max(self.y_test.max(), ensemble_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # Add labels and title
        ax.set_xlabel('Actual Magnitude')
        ax.set_ylabel('Predicted Magnitude')
        ax.set_title('Earthquake Magnitude Prediction: Actual vs Predicted')
        
        # Add performance metrics
        rmse = self.ensemble_results['rmse']
        r2 = self.ensemble_results['r2']
        ax.text(0.05, 0.95, f'RMSE: {rmse:.4f}\nR²: {r2:.4f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def create_geographic_heatmap(self, grid_size=50):
        """
        Create a geographic heatmap of predicted earthquake magnitudes
        
        Args:
            grid_size (int): Size of the grid for prediction
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        # Get boundaries from data
        lat_min, lat_max = self.processed_data['Lat'].min(), self.processed_data['Lat'].max()
        lon_min, lon_max = self.processed_data['Lon'].min(), self.processed_data['Lon'].max()
        
        # Create grid
        lat_grid = np.linspace(lat_min, lat_max, grid_size)
        lon_grid = np.linspace(lon_min, lon_max, grid_size)
        
        # Use median depth
        depth = self.processed_data['Depth'].median()
        
        # Create prediction grid
        predictions = np.zeros((grid_size, grid_size))
        uncertainties = np.zeros((grid_size, grid_size))
        
        for i, lat in enumerate(lat_grid):
            for j, lon in enumerate(lon_grid):
                pred, conf_int, _, _ = self.predict_magnitude(lat, lon, depth)
                predictions[i, j] = pred
                uncertainties[i, j] = (conf_int[1] - conf_int[0]) / 2
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot magnitude heatmap
        im1 = ax1.imshow(predictions, origin='lower', extent=[lon_min, lon_max, lat_min, lat_max],
                      cmap='YlOrRd', aspect='auto')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title('Predicted Earthquake Magnitude')
        fig.colorbar(im1, ax=ax1, label='Magnitude')
        
        # Plot uncertainty heatmap
        im2 = ax2.imshow(uncertainties, origin='lower', extent=[lon_min, lon_max, lat_min, lat_max],
                      cmap='Blues', aspect='auto')
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        ax2.set_title('Prediction Uncertainty (95% CI)')
        fig.colorbar(im2, ax=ax2, label='Uncertainty')
        
        # Plot actual earthquake locations
        ax1.scatter(self.processed_data['Lon'], self.processed_data['Lat'], 
                   s=self.processed_data['Mag']*2, c='black', alpha=0.3, marker='.')
        
        plt.tight_layout()
        return fig
    
    def train_all_models(self):
        """
        Train all models and create ensemble
        
        Returns:
            dict: Dictionary with trained models and ensemble results
        """
        self.train_random_forest()
        self.train_gradient_boosting()
        self.train_lstm_model()
        self.train_gru_model()
        self.train_gaussian_process()
        self.create_ensemble()
        
        return {
            'models': self.models,
            'ensemble_weights': self.ensemble_weights,
            'ensemble_results': self.ensemble_results
        }

def main():
    """
    Main function to demonstrate the earthquake prediction system
    """
    # Create prediction system
    system = EarthquakePredictionSystem()
    
    # Load and preprocess data
    # Replace with actual file path
    data = pd.read_csv('earthquake_data.csv')
    system.data = data
    
    # Preprocessing steps
    system.preprocess_data()
    system.identify_spatial_regions(n_clusters=5)
    system.engineer_features()
    system.prepare_for_training(test_size=0.2, temporal_split=True)
    
    # Train models
    system.train_all_models()
    
    # Visualize results
    fig1 = system.visualize_predictions()
    fig2 = system.create_geographic_heatmap()
    
    # Show plots
    plt.show()
    
    # Example prediction
    lat, lon, depth = -5.4, 35.0, 10.0
    pred_mag, conf_int, model_preds, model_uncertainties = system.predict_magnitude(lat, lon, depth)
    
    print(f"\nPrediction for location ({lat}, {lon}, {depth}km):")
    print(f"Predicted magnitude: {pred_mag:.2f}")
    print(f"95% confidence interval: [{conf_int[0]:.2f}, {conf_int[1]:.2f}]")
    print(f"\nIndividual model predictions:")
    for model, pred in model_preds.items():
        print(f"  {model}: {pred:.2f} ± {model_uncertainties[model]:.2f}")
    
if __name__ == "__main__":
    main()