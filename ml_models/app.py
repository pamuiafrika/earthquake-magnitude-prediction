import pandas as pd
import matplotlib.pyplot as plt
import os
from data_loader import load_sample_data
from EarthquakePredictionSystem import EarthquakePredictionSystem
from EarthquakePredictionInterface import EarthquakePredictionInterface

def run_prediction_system(data_file='Earthquake_Events_1970_2025_083913.csv'):
    """
    Run the earthquake prediction system with synthetic or provided data.
    
    Args:
        data_file (str): Path to the input data file.
    
    Returns:
        dict: Results including predictions and visualization paths.
    """
    results = {
        'status': 'failed',
        'message': '',
        'prediction': None,
        'visualizations': [],
        'report': None
    }
    
    try:
        # Step 1: Load data
        if not os.path.exists(data_file):
            print(f"Data file {data_file} not found. Generating synthetic data...")
            data = load_sample_data(output_file=data_file)
        else:
            data = pd.read_csv(data_file)
        
        if data is None:
            raise ValueError("Failed to load data")
        
        print(f"Loaded dataset with {len(data)} records")
        
        # Step 2: Initialize the prediction system and interface
        system = EarthquakePredictionSystem()
        interface = EarthquakePredictionInterface(model_system=system)
        
        # Step 3: Load data into the system
        temp_file = 'temp_earthquake_data.csv'
        data.to_csv(temp_file, index=False)
        success = interface.load_and_prepare_data(temp_file)
        
        if not success:
            raise ValueError("Failed to prepare data")
        
        # Step 4: Train models
        success = interface.train_models()
        if not success:
            raise ValueError("Failed to train models")
        
        # Step 5: Make a sample prediction
        lat, lon, depth = -5.4, 35.0, 10.0
        pred_mag, conf_int, model_preds, uncertainties = interface.predict_earthquake(lat, lon, depth)
        
        results['prediction'] = {
            'latitude': lat,
            'longitude': lon,
            'depth': depth,
            'magnitude': float(pred_mag),
            'confidence_interval': [float(conf_int[0]), float(conf_int[1])],
            'model_predictions': {model: float(pred) for model, pred in model_preds.items()},
            'model_uncertainties': {model: float(unc) for model, unc in uncertainties.items()}
        }
        
        print(f"\nPrediction for location (Lat: {lat}, Lon: {lon}, Depth: {depth}km):")
        print(f"Predicted magnitude: {pred_mag:.2f}")
        print(f"95% confidence interval: [{conf_int[0]:.2f}, {conf_int[1]:.2f}]")
        print("\nIndividual model predictions:")
        for model, pred in model_preds.items():
            print(f"  {model}: {pred:.2f} ± {uncertainties[model]:.2f}")
        
        # Step 6: Generate visualizations
        output_dir = 'static/visualizations'
        os.makedirs(output_dir, exist_ok=True)
        
        visualizations = []
        
        # Model performance
        fig1 = interface.visualize_model_performance()
        if fig1:
            fig1_path = os.path.join(output_dir, 'model_performance.png')
            fig1.savefig(fig1_path)
            plt.close(fig1)
            visualizations.append(fig1_path)
            print(f"Model performance visualization saved as {fig1_path}")
        
        # Geographic heatmap
        fig2 = interface.visualize_geographic_heatmap(grid_size=30)
        if fig2:
            fig2_path = os.path.join(output_dir, 'geographic_heatmap.png')
            fig2.savefig(fig2_path)
            plt.close(fig2)
            visualizations.append(fig2_path)
            print(f"Geographic heatmap saved as {fig2_path}")
        
        # Feature importance
        fig3 = interface.display_feature_importance()
        if fig3:
            fig3_path = os.path.join(output_dir, 'feature_importance.png')
            fig3.savefig(fig3_path)
            plt.close(fig3)
            visualizations.append(fig3_path)
            print(f"Feature importance visualization saved as {fig3_path}")
        
        # Regional analysis
        fig4 = interface.visualize_regional_analysis()
        if fig4:
            fig4_path = os.path.join(output_dir, 'regional_analysis.png')
            fig4.savefig(fig4_path)
            plt.close(fig4)
            visualizations.append(fig4_path)
            print(f"Regional analysis visualization saved as {fig4_path}")
        
        # Interactive map
        interactive_map = interface.create_interactive_map()
        if interactive_map:
            map_path = os.path.join(output_dir, 'interactive_map.html')
            interactive_map.save(map_path)
            visualizations.append(map_path)
            print(f"Interactive map saved as {map_path}")
        
        results['visualizations'] = visualizations
        
        # Step 7: Generate prediction report
        report = interface.create_prediction_report(lat, lon, depth)
        if report:
            results['report'] = report
            print("\nPrediction Report:")
            print(f"Predicted Magnitude: {report['prediction']['magnitude']:.2f}")
            print(f"Confidence Level: {report['prediction']['confidence_level']}")
            print(f"Number of Similar Historical Events: {len(report['similar_historical_events'])}")
            if report['similar_historical_events']:
                print("First similar event:", report['similar_historical_events'][0])
        
        results['status'] = 'success'
        results['message'] = 'Prediction system ran successfully'
        
        return results
    
    except Exception as e:
        results['message'] = f"Error running prediction system: {str(e)}"
        print(results['message'])
        return results

if __name__ == "__main__":
    results = run_prediction_system()
    if results['status'] == 'success':
        print("\nPrediction system completed successfully!")
    else:
        print("\nPrediction system failed:", results['message'])