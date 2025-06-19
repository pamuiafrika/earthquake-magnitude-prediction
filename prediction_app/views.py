import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from django.conf import settings
from django.urls import reverse
from django.core.paginator import Paginator
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import csv

from .models import Earthquake, PredictionLog, ModelMetrics
from .ml_models.data_processor import EarthquakeDataProcessor
from .ml_models.model_trainer import EarthquakeModelTrainer
from .ml_models.predictor import EarthquakePredictionService

# Initialize prediction service
DATA_PATH = os.path.join(settings.BASE_DIR, 'prediction_app/static/data/earthquake_data.csv')
MODELS_DIR = os.path.join(settings.BASE_DIR, 'prediction_app/ml_models/models')

# Create prediction service as a global variable
prediction_service = None

def initialize_prediction_service():
    """Initialize the prediction service if not already done."""
    global prediction_service
    try:
        if prediction_service is None:
            prediction_service = EarthquakePredictionService(DATA_PATH, MODELS_DIR)
        return True
    except Exception as e:
        print(f"Error initializing prediction service: {e}")
        return False

def index(request):
    """Home page view."""
    # Initialize prediction service
    service_ready = initialize_prediction_service()
    
    # Get earthquake statistics
    stats = {}
    if service_ready:
        try:
            df = prediction_service.data_processor.df
            stats = {
                'total_earthquakes': len(df),
                'avg_magnitude': df['Mag'].mean(),
                'max_magnitude': df['Mag'].max(),
                'recent_count': len(df[df['Year'] >= datetime.now().year - 1]),
            }
        except Exception as e:
            messages.error(request, f"Error loading statistics: {e}")
    
    context = {
        'service_ready': service_ready,
        'stats': stats,
    }
    return render(request, 'ai/index.html', context)

@csrf_exempt
def predict(request):
    """View for earthquake prediction."""
    if request.method == 'POST':
        try:
            # Initialize prediction service if needed
            if not initialize_prediction_service():
                return JsonResponse({
                    'error': 'Prediction service not available'
                }, status=500)
            
            # Get input parameters
            data = json.loads(request.body) if request.body else request.POST
            
            lat = float(data.get('latitude'))
            lon = float(data.get('longitude'))
            depth = float(data.get('depth'))
            
            # Validate input
            if not (-90 <= lat <= 90 and -180 <= lon <= 180 and depth >= 0):
                return JsonResponse({
                    'error': 'Invalid input parameters. Latitude must be between -90 and 90, longitude between -180 and 180, and depth must be positive.'
                }, status=400)
            
            # Make prediction
            magnitude = prediction_service.predict_magnitude(lat, lon, depth)
            
            # Log prediction
            PredictionLog.objects.create(
                latitude=lat,
                longitude=lon,
                depth=depth,
                predicted_magnitude=magnitude,
                model_version=getattr(prediction_service.model_trainer, 'best_model_name', 'unknown')
            )
            
            # Get regional analysis
            regional_analysis = prediction_service.get_regional_analysis(lat, lon, radius=1.0)
            
            response_data = {
                'magnitude': round(magnitude, 2),
                'latitude': lat,
                'longitude': lon,
                'depth': depth,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'regional_data': {
                    'count': regional_analysis.get('stats', {}).get('count', 0),
                    'avg_magnitude': round(regional_analysis.get('stats', {}).get('avg_magnitude', 0), 2),
                    'max_magnitude': regional_analysis.get('stats', {}).get('max_magnitude', 0),
                }
            }
            
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse(response_data)
            else:
                context = {
                    'prediction': response_data,
                    'regional_analysis': regional_analysis.get('stats', {}),
                }
                return render(request, 'ai/prediction_result.html', context)
                
        except Exception as e:
            error_message = f"Error making prediction: {str(e)}"
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'error': error_message}, status=500)
            else:
                messages.error(request, error_message)
                return redirect('ai:predict')
    
    # GET request - show prediction form
    return render(request, 'ai/prediction.html')

def analytics(request):
    """View for earthquake analytics dashboard."""
    if not initialize_prediction_service():
        messages.error(request, "Analytics service not available")
        return redirect('ai:index')
    
    try:
        # Get filter parameters
        lat = request.GET.get('latitude')
        lon = request.GET.get('longitude')
        radius = request.GET.get('radius', 1.0)
        
        if lat and lon:
            lat = float(lat)
            lon = float(lon)
            radius = float(radius)
            
            # Get regional analysis
            regional_data = prediction_service.get_regional_analysis(lat, lon, radius)
            
            # Get heatmap data
            heatmap_data = prediction_service.generate_heatmap(lat, lon, radius)
        else:
            # Get overall analysis
            df = prediction_service.data_processor.df
            
            regional_data = {
                'stats': {
                    'count': len(df),
                    'avg_magnitude': df['Mag'].mean(),
                    'max_magnitude': df['Mag'].max(),
                    'min_magnitude': df['Mag'].min(),
                    'std_magnitude': df['Mag'].std(),
                    'recent_events': len(df[df['Year'] >= datetime.now().year - 5]),
                }
            }
            
            # Get heatmap data for all Tanzania
            heatmap_data = prediction_service.generate_heatmap()
        
        # Get time series analysis
        time_series = prediction_service.get_time_series_analysis()
        
        context = {
            'regional_data': regional_data.get('stats', {}),
            'heatmap_data': json.dumps(heatmap_data),
            'time_series': json.dumps(time_series),
            'filter': {
                'latitude': lat,
                'longitude': lon,
                'radius': radius,
            } if lat and lon else None
        }
        
        return render(request, 'ai/analytics.html', context)
        
    except Exception as e:
        messages.error(request, f"Error loading analytics: {str(e)}")
        return redirect('ai:index')

def train_model(request):
    """View for training/retraining ML models."""
    if not request.user.is_superuser:
        messages.error(request, "You don't have permission to train models")
        return redirect('users:home')
    
    if request.method == 'POST':
        try:
            # Initialize prediction service if needed
            if not initialize_prediction_service():
                messages.error(request, "Prediction service not available")
                return redirect('ai:train_model')
            
            # Get training parameters
            test_size = float(request.POST.get('test_size', 0.2))
            random_state = int(request.POST.get('random_state', 42))
            
            # Train models
            results = prediction_service.train_new_model(test_size, random_state)
            
            # Save model metrics
            for model_name, metrics in results.items():
                ModelMetrics.objects.create(
                    model_name=model_name,
                    version=datetime.now().strftime('%Y%m%d_%H%M%S'),
                    rmse=metrics['rmse'],
                    mae=metrics['mae'],
                    r2_score=metrics['r2'],
                    mse=metrics['mse'],
                    mape=metrics['mape'],
                    explained_variance=metrics['explained_variance'],
                    feature_count=len(prediction_service.data_processor.create_features().columns) - 1,
                    training_samples=int(len(prediction_service.data_processor.df) * (1 - test_size)),
                    is_active=(model_name == prediction_service.model_trainer.best_model_name)
                )
            
            messages.success(request, f"Models trained successfully. Best model: {prediction_service.model_trainer.best_model_name}")
            return redirect('ai:model_metrics')
            
        except Exception as e:
            messages.error(request, f"Error training models: {str(e)}")
            return redirect('ai:train_model')
    
    return render(request, 'ai/train_model.html')

def model_metrics(request):
    """View for model performance metrics."""
    metrics = ModelMetrics.objects.all().order_by('-trained_date')
    models = ModelMetrics.objects.all().count()
    
    # Paginate results
    paginator = Paginator(metrics, 10)
    page = request.GET.get('page', 1)
    metrics_page = paginator.get_page(page)
    
    context = {
        'metrics': metrics_page,
        'metrics_count': models
    }
    return render(request, 'ai/model_metrics.html', context)


def api_earthquake_data(request):
    """View for displaying earthquake data."""
    if not initialize_prediction_service():
        messages.error(request, "Data service not available")
        return redirect('users:home')
    
    try:
        df = prediction_service.data_processor.df
        
        # Filter parameters
        year = request.GET.get('year')
        min_mag = request.GET.get('min_mag')
        
        if year:
            df = df[df['Year'] == int(year)]
        
        if min_mag:
            df = df[df['Mag'] >= float(min_mag)]
        
        # Convert to list of dictionaries for template
        earthquakes = df.to_dict('records')
        
        # Paginate results
        paginator = Paginator(earthquakes, 50)
        page = request.GET.get('page', 1)
        earthquakes_page = paginator.get_page(page)
        
        # Get unique years for filter
        years = sorted(df['Year'].unique(), reverse=True)
        
        context = {
            'earthquakes': earthquakes_page,
            'years': years,
            'filters': {
                'year': year,
                'min_mag': min_mag,
            }
        }
        
        return render(request, 'prediction/historical_data.html', context)
        
    except Exception as e:
        messages.error(request, f"Error loading earthquake data: {str(e)}")
        return redirect('users:home')

def earthquake_data(request):
    """View for displaying earthquake data."""
    if not initialize_prediction_service():
        messages.error(request, "Data service not available")
        return redirect('ai:index')
    
    try:
        df = prediction_service.data_processor.df
        
        # Filter parameters
        year = request.GET.get('year')
        min_mag = request.GET.get('min_mag')
        
        if year:
            df = df[df['Year'] == int(year)]
        
        if min_mag:
            df = df[df['Mag'] >= float(min_mag)]
        
        # Convert to list of dictionaries for template
        earthquakes = df.to_dict('records')
        
        # Paginate results
        paginator = Paginator(earthquakes, 50)
        page = request.GET.get('page', 1)
        earthquakes_page = paginator.get_page(page)
        
        # Get unique years for filter
        years = sorted(df['Year'].unique(), reverse=True)
        
        context = {
            'earthquakes': earthquakes_page,
            'years': years,
            'filters': {
                'year': year,
                'min_mag': min_mag,
            }
        }
        
        return render(request, 'ai/earthquake_data.html', context)
        
    except Exception as e:
        messages.error(request, f"Error loading earthquake data: {str(e)}")
        return redirect('ai:index')

def export_earthquake_csv(request):
    """View for exporting earthquake data to CSV."""
    if not initialize_prediction_service():
        messages.error(request, "Data service not available")
        return redirect('users:home')
        
    try:
        import csv
        from django.http import HttpResponse
        
        # Get the data from the prediction service
        df = prediction_service.data_processor.df
        
        # Apply filters if provided
        year = request.GET.get('year')
        min_mag = request.GET.get('min_mag')
        
        if year:
            df = df[df['Year'] == int(year)]
            
        if min_mag:
            df = df[df['Mag'] >= float(min_mag)]
        
        # Create the HttpResponse object with CSV header
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="earthquake_data.csv"'
        
        # Create the CSV writer
        writer = csv.writer(response)
        
        # Write headers
        writer.writerow(df.columns)
        
        # Write data rows
        for _, row in df.iterrows():
            writer.writerow(row)
        
        return response
        
    except Exception as e:
        messages.error(request, f"Error exporting earthquake data: {str(e)}")
        return redirect('ai:api_earthquake_data')
    
def api_heatmap(request):
    """API endpoint for heatmap data."""
    if not initialize_prediction_service():
        return JsonResponse({'error': 'Service not available'}, status=500)
    
    try:
        lat = request.GET.get('lat')
        lon = request.GET.get('lon')
        radius = request.GET.get('radius', 1.0)
        
        if lat and lon:
            lat = float(lat)
            lon = float(lon)
            radius = float(radius)
            heatmap_data = prediction_service.generate_heatmap(lat, lon, radius)
        else:
            heatmap_data = prediction_service.generate_heatmap()
        
        return JsonResponse(heatmap_data)
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def api_time_series(request):
    """API endpoint for time series data."""
    if not initialize_prediction_service():
        return JsonResponse({'error': 'Service not available'}, status=500)
    
    try:
        time_series = prediction_service.get_time_series_analysis()
        return JsonResponse(time_series)
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def import_data(request):
    """View for importing earthquake data."""
    if not request.user.is_superuser:
        messages.error(request, "You don't have permission to import data")
        return redirect('users:home')
    
    if request.method == 'POST' and request.FILES.get('data_file'):
        try:
            data_file = request.FILES['data_file']
            
            # Save the uploaded file
            file_path = os.path.join(settings.MEDIA_ROOT, 'uploads', data_file.name)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'wb+') as destination:
                for chunk in data_file.chunks():
                    destination.write(chunk)
            
            # Load data
            df = pd.read_csv(file_path)
            
            # Validate data structure
            required_columns = ['Year', 'Month', 'Day', 'Time', 'Lat', 'Lon', 'Depth', 'Mag']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in data file")
            
            # Import into database (optional)
            count = 0
            for _, row in df.iterrows():
                try:
                    date_str = f"{int(row['Year'])}-{int(row['Month']):02d}-{int(row['Day']):02d}"
                    time_str = row['Time']
                    
                    Earthquake.objects.update_or_create(
                        date=date_str,
                        time=time_str,
                        latitude=row['Lat'],
                        longitude=row['Lon'],
                        defaults={
                            'depth': row['Depth'],
                            'magnitude': row['Mag'],
                        }
                    )
                    count += 1
                except Exception as e:
                    print(f"Error importing row: {e}")
            
            # Update main data file
            df.to_csv(DATA_PATH, index=False)
            
            # Reinitialize prediction service
            global prediction_service
            prediction_service = None
            initialize_prediction_service()
            
            messages.success(request, f"Data imported successfully: {count} records")
            return redirect('ai:api_earthquake_data')
            
        except Exception as e:
            messages.error(request, f"Error importing data: {str(e)}")
            return redirect('ai:train_model')
    
    return render(request, 'ai/import_data.html')
