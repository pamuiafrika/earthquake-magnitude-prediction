import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import folium
from folium.plugins import HeatMap
import io
import base64
from IPython.display import HTML, display
import EarthquakePredictionSystem

class EarthquakePredictionInterface:
    """
    Interface for the Earthquake Prediction System with visualization capabilities
    """
    def __init__(self, model_system=None):
        """
        Initialize the interface
        
        Args:
            model_system (EarthquakePredictionSystem, optional): Pre-trained model system
        """
        self.system = model_system or EarthquakePredictionSystem()
        self.data = None
        self.test_predictions = None
    
    def load_and_prepare_data(self, file_path):
        """
        Load data and prepare the model system
        
        Args:
            file_path (str): Path to CSV file
            
        Returns:
            bool: Success status
        """
        try:
            # Load data
            self.data = self.system.load_data(file_path)
            
            # Process data
            self.system.preprocess_data()
            self.system.identify_spatial_regions(n_clusters=5)
            self.system.engineer_features()
            self.system.prepare_for_training(test_size=0.2, temporal_split=True)
            
            print("Data loaded and prepared successfully!")
            return True
        except Exception as e:
            print(f"Error in data preparation: {e}")
            return False
    
    def train_models(self):
        """
        Train all models in the system
        
        Returns:
            bool: Success status
        """
        try:
            results = self.system.train_all_models()
            self.test_predictions = self.system.ensemble_results['predictions']
            print("All models trained successfully!")
            return True
        except Exception as e:
            print(f"Error training models: {e}")
            return False
    
    def predict_earthquake(self, latitude, longitude, depth, year=None, month=None, day=None, hour=None):
        """
        Make a prediction for a specific location and time
        
        Args:
            latitude (float): Latitude coordinate
            longitude (float): Longitude coordinate
            depth (float): Depth in km
            year (int, optional): Year
            month (int, optional): Month
            day (int, optional): Day
            hour (int, optional): Hour
            
        Returns:
            tuple: (predicted_magnitude, confidence_interval, model_predictions, model_uncertainties)
        """
        try:
            return self.system.predict_magnitude(latitude, longitude, depth, year, month, day, hour)
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None, None, None, None
    
    def visualize_model_performance(self):
        """
        Visualize model performance (actual vs predicted)
        
        Returns:
            matplotlib.figure.Figure: Performance visualization
        """
        if self.test_predictions is None:
            print("No predictions available. Please train models first.")
            return None
        
        try:
            return self.system.visualize_predictions()
        except Exception as e:
            print(f"Error creating visualization: {e}")
            return None
    
    def visualize_geographic_heatmap(self, grid_size=30):
        """
        Create a geographic heatmap of predicted earthquake magnitudes
        
        Args:
            grid_size (int): Size of the grid for prediction
            
        Returns:
            matplotlib.figure.Figure: Heatmap visualization
        """
        try:
            return self.system.create_geographic_heatmap(grid_size)
        except Exception as e:
            print(f"Error creating heatmap: {e}")
            return None
    
    def create_interactive_map(self):
        """
        Create an interactive map with earthquake data
        
        Returns:
            folium.Map: Interactive map object
        """
        if self.data is None:
            print("No data available. Please load data first.")
            return None
        
        try:
            # Get processed data
            df = self.system.processed_data
            
            # Calculate center point
            center_lat = df['Lat'].mean()
            center_lon = df['Lon'].mean()
            
            # Create map
            m = folium.Map(location=[center_lat, center_lon], zoom_start=5)
            
            # Normalize magnitude for color scaling
            norm = Normalize(vmin=df['Mag'].min(), vmax=df['Mag'].max())
            cmap = cm.get_cmap('YlOrRd')
            
            # Add markers for earthquakes
            for idx, row in df.iterrows():
                # Skip if more than 100 points to avoid overloading the map
                if idx > 100:
                    break
                    
                # Calculate color based on magnitude
                color = '#%02x%02x%02x' % tuple(int(c * 255) for c in cmap(norm(row['Mag']))[:3])
                
                # Create popup text
                popup_text = f"""
                <b>Magnitude:</b> {row['Mag']:.1f}<br>
                <b>Depth:</b> {row['Depth']} km<br>
                <b>Date:</b> {int(row['Year'])}-{int(row['Month'])}-{int(row['Day'])}<br>
                <b>Time:</b> {row['Time'] if 'Time' in row else 'N/A'}<br>
                """
                
                # Add marker
                folium.CircleMarker(
                    location=[row['Lat'], row['Lon']],
                    radius=row['Mag'] * 2,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    popup=folium.Popup(popup_text, max_width=300)
                ).add_to(m)
            
            # Add heatmap
            heat_data = [[row['Lat'], row['Lon'], row['Mag']] for _, row in df.iterrows()]
            HeatMap(heat_data, radius=15, gradient={0.4: 'blue', 0.65: 'lime', 0.8: 'orange', 1: 'red'}).add_to(m)
            
            return m
        except Exception as e:
            print(f"Error creating interactive map: {e}")
            return None
    
    def display_feature_importance(self):
        """
        Display feature importance from tree-based models
        
        Returns:
            matplotlib.figure.Figure: Feature importance visualization
        """
        if 'random_forest' not in self.system.models:
            print("Random Forest model not available. Please train models first.")
            return None
        
        try:
            # Get feature importance from Random Forest model
            rf_model = self.system.models['random_forest']
            feature_importance = rf_model.feature_importances_
            feature_names = self.system.feature_names
            
            # Sort by importance
            indices = np.argsort(feature_importance)[::-1]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot top 20 features
            top_n = min(20, len(feature_names))
            ax.barh(range(top_n), feature_importance[indices][:top_n], align='center')
            ax.set_yticks(range(top_n))
            ax.set_yticklabels([feature_names[i] for i in indices[:top_n]])
            ax.set_xlabel('Feature Importance')
            ax.set_title('Top 20 Feature Importance from Random Forest')
            
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Error creating feature importance visualization: {e}")
            return None
    
    def visualize_regional_analysis(self):
        """
        Visualize earthquake patterns by region
        
        Returns:
            matplotlib.figure.Figure: Regional analysis visualization
        """
        if self.system.processed_data is None:
            print("No processed data available. Please load and prepare data first.")
            return None
            
        try:
            df = self.system.processed_data
            
            # Extract region columns 
            region_cols = [col for col in df.columns if col.startswith('Region_Region_')]
            
            # Create a single region label for visualization
            df['Region_Label'] = np.nan
            for i, col in enumerate(region_cols):
                df.loc[df[col] == 1, 'Region_Label'] = i
            
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            
            # Plot 1: Magnitude distribution by region
            sns.boxplot(x='Region_Label', y='Mag', data=df, ax=axes[0, 0])
            axes[0, 0].set_title('Magnitude Distribution by Region')
            axes[0, 0].set_xlabel('Region')
            axes[0, 0].set_ylabel('Magnitude')
            
            # Plot 2: Depth distribution by region
            sns.boxplot(x='Region_Label', y='Depth', data=df, ax=axes[0, 1])
            axes[0, 1].set_title('Depth Distribution by Region')
            axes[0, 1].set_xlabel('Region')
            axes[0, 1].set_ylabel('Depth (km)')
            
            # Plot 3: Earthquake count by region
            region_counts = df['Region_Label'].value_counts().sort_index()
            axes[1, 0].bar(region_counts.index, region_counts.values)
            axes[1, 0].set_title('Earthquake Count by Region')
            axes[1, 0].set_xlabel('Region')
            axes[1, 0].set_ylabel('Count')
            
            # Plot 4: Average magnitude over time by region
            # Group by year and region, calculate mean magnitude
            if 'Year_num' in df.columns:
                time_region_avg = df.groupby(['Year_num', 'Region_Label'])['Mag'].mean().reset_index()
                
                # Plot for each region
                for region in df['Region_Label'].unique():
                    region_data = time_region_avg[time_region_avg['Region_Label'] == region]
                    axes[1, 1].plot(region_data['Year_num'], region_data['Mag'], marker='o', label=f'Region {region}')
                
                axes[1, 1].set_title('Average Magnitude Over Time by Region')
                axes[1, 1].set_xlabel('Year')
                axes[1, 1].set_ylabel('Average Magnitude')
                axes[1, 1].legend()
            
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Error creating regional analysis: {e}")
            return None
    
    def create_prediction_report(self, latitude, longitude, depth):
        """
        Create a comprehensive prediction report for a given location
        
        Args:
            latitude (float): Latitude coordinate
            longitude (float): Longitude coordinate
            depth (float): Depth in km
            
        Returns:
            dict: Prediction report data
        """
        if self.system.models == {}:
            print("No trained models available. Please train models first.")
            return None
            
        try:
            # Get prediction
            pred_mag, conf_int, model_preds, uncertainties = self.predict_earthquake(latitude, longitude, depth)
            
            # Find similar historical earthquakes
            df = self.system.processed_data
            
            # Calculate spatial distance
            df['dist_to_query'] = np.sqrt(
                (df['Lat'] - latitude)**2 + 
                (df['Lon'] - longitude)**2 + 
                (df['Depth'] - depth)**2
            )
            
            # Get 5 most similar earthquakes
            similar_quakes = df.sort_values('dist_to_query').head(5)
            
            # Extract relevant columns
            similar_quakes = similar_quakes[['Year', 'Month', 'Day', 'Time', 'Lat', 'Lon', 'Depth', 'Mag']]
            
            # Determine confidence level based on uncertainty
            uncertainty = (conf_int[1] - conf_int[0]) / 2
            if uncertainty < 0.3:
                confidence_level = "High"
            elif uncertainty < 0.5:
                confidence_level = "Medium"
            else:
                confidence_level = "Low"
                
            # Create report
            report = {
                'prediction': {
                    'magnitude': float(pred_mag),
                    'confidence_interval': [float(conf_int[0]), float(conf_int[1])],
                    'uncertainty': float(uncertainty),
                    'confidence_level': confidence_level
                },
                'location': {
                    'latitude': latitude,
                    'longitude': longitude,
                    'depth': depth
                },
                'model_predictions': {model: float(pred) for model, pred in model_preds.items()},
                'model_uncertainties': {model: float(unc) for model, unc in uncertainties.items()},
                'similar_historical_events': similar_quakes.to_dict('records')
            }
            
            return report
        except Exception as e:
            print(f"Error creating prediction report: {e}")
            return None
    
def main():
    """
    Main function to demonstrate the earthquake prediction interface
    """
    # Create interface
    interface = EarthquakePredictionInterface()
    
    # Load and prepare data (replace with actual file path)
    interface.load_and_prepare_data('earthquake_data.csv')
    
    # Train models
    interface.train_models()
    
    # Make a prediction
    lat, lon, depth = -5.4, 35.0, 10.0
    pred_mag, conf_int, model_preds, uncertainties = interface.predict_earthquake(lat, lon, depth)
    
    print(f"\nPrediction for location ({lat}, {lon}, {depth}km):")
    print(f"Predicted magnitude: {pred_mag:.2f}")
    print(f"95% confidence interval: [{conf_int[0]:.2f}, {conf_int[1]:.2f}]")
    
    # Show visualizations
    interface.visualize_model_performance()
    interface.visualize_geographic_heatmap()
    interface.display_feature_importance()
    interface.visualize_regional_analysis()
    
    # Generate comprehensive report
    report = interface.create_prediction_report(lat, lon, depth)
    print("\nPrediction Report:")
    print(f"Predicted Magnitude: {report['prediction']['magnitude']:.2f}")
    print(f"Confidence Level: {report['prediction']['confidence_level']}")
    print(f"Similar Historical Events: {len(report['similar_historical_events'])}")
    
    plt.show()
    
if __name__ == "__main__":
    main()