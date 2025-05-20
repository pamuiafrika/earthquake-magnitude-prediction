from rest_framework import viewsets, status
from rest_framework.decorators import api_view, permission_classes, action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, IsAdminUser, AllowAny
from django.shortcuts import get_object_or_404
from django.conf import settings
from datetime import datetime
import os
import pandas as pd
import json

from .models import Earthquake, PredictionLog, ModelMetrics
from .serializers import (
    EarthquakeSerializer, 
    PredictionLogSerializer, 
    ModelMetricsSerializer,
    PredictionRequestSerializer
)
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


class EarthquakeViewSet(viewsets.ReadOnlyModelViewSet):
    """API endpoint for earthquake data."""
    queryset = Earthquake.objects.all()
    serializer_class = EarthquakeSerializer

    def get_queryset(self):
        queryset = Earthquake.objects.all()
        # Filter parameters
        start = self.request.query_params.get('start_date')
        end = self.request.query_params.get('end_date')
        min_mag = self.request.query_params.get('min_magnitude')
        region = self.request.query_params.get('region')

        if start:
            queryset = queryset.filter(date__gte=start)
        if end:
            queryset = queryset.filter(date__lte=end)
        if min_mag and float(min_mag) > 0:
            queryset = queryset.filter(magnitude__gte=float(min_mag))
        if region and region != 'all':
            queryset = queryset.filter(region__iexact=region)

        return queryset.order_by('-date', '-time')

    @action(detail=False, methods=['get'])
    def years(self, request):
        """Get unique years for filtering."""
        years = Earthquake.objects.dates('date', 'year').values_list('date__year', flat=True)
        return Response(sorted(years, reverse=True))

class PredictionLogViewSet(viewsets.ReadOnlyModelViewSet):
    """API endpoint for prediction logs."""
    queryset = PredictionLog.objects.all().order_by('-timestamp')
    serializer_class = PredictionLogSerializer
    permission_classes = [IsAuthenticated]


class ModelMetricsViewSet(viewsets.ReadOnlyModelViewSet):
    """API endpoint for model metrics."""
    queryset = ModelMetrics.objects.all().order_by('-trained_date')
    serializer_class = ModelMetricsSerializer
    permission_classes = [IsAdminUser]


@api_view(['GET'])
def api_statistics(request):
    """API endpoint for earthquake statistics."""
    if not initialize_prediction_service():
        return Response({'error': 'Service not available'}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
    
    try:
        df = prediction_service.data_processor.df
        
        # Get basic stats
        stats = {
            'total_earthquakes': len(df),
            'avg_magnitude': round(df['Mag'].mean(), 2),
            'max_magnitude': float(df['Mag'].max()),
            'recent_count': len(df[df['Year'] >= datetime.now().year - 1]),
        }
        
        # Get 10 latest events
        latest_events = df.sort_values(['Year', 'Month', 'Day'], ascending=False).head(5)
        stats['latest_events'] = [
            {
                'date': f"{int(row['Year'])}-{int(row['Month']):02d}-{int(row['Day']):02d}",
                'magnitude': row['Mag'],
                'latitude': row['Lat'],
                'longitude': row['Lon'],
                'depth': row['Depth']
            }
            for _, row in latest_events.iterrows()
        ]
        
        return Response(stats)
        
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
def api_predict(request):
    """API endpoint for earthquake prediction."""
    if not initialize_prediction_service():
        return Response({'error': 'Prediction service not available'}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
    
    try:
        serializer = PredictionRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        lat = serializer.validated_data['latitude']
        lon = serializer.validated_data['longitude']
        depth = serializer.validated_data['depth']
        
        # Make prediction
        magnitude = prediction_service.predict_magnitude(lat, lon, depth)
        model = ModelMetrics.objects.filter(is_active=True).first()
        
        if not model:
            return Response({'error': 'No active model found. Please contact your System Administrator'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # Log prediction
        prediction_log = PredictionLog.objects.create(
            latitude=lat,
            longitude=lon,
            depth=depth,
            predicted_magnitude=magnitude,
            model_version=f"{str(model.get_active_model_name())} v{model.get_active_model_version()}"
        )
        prediction_log.save()
            
        print(f"Prediction logged: {getattr(prediction_service.model_trainer, 'best_model_name', 'unknown')}")
        # Get regional analysis
        regional_analysis = prediction_service.get_regional_analysis(lat, lon, radius=1.0)
        
        response_data = {
            'id': prediction_log.id,
            'magnitude': round(magnitude, 2),
            'model_version': prediction_log.model_version,
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
        
        return Response(response_data)
            
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)



@api_view(['GET'])
def api_heatmap(request):
    """API endpoint for heatmap data."""
    if not initialize_prediction_service():
        return Response({'error': 'Service not available'}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
    
    try:
        lat = request.query_params.get('lat')
        lon = request.query_params.get('lon')
        radius = request.query_params.get('radius')
        
        # Default values if parameters are not provided
        if lat is None or lon is None:
            # Default center of Tanzania
            lat, lon = -6.0, 35.0
        else:
            lat = float(lat)
            lon = float(lon)
        
        if radius is None:
            radius = 5.0  # Default radius
        else:
            radius = float(radius)
        
        # Generate heatmap data
        heatmap_data = prediction_service.generate_heatmap(lat, lon, radius)
        
        # Ensure the response has the required structure
        if 'data' not in heatmap_data:
            heatmap_data['data'] = []
        
        if 'center' not in heatmap_data:
            heatmap_data['center'] = {'lat': lat, 'lon': lon}
        
        if 'count' not in heatmap_data:
            heatmap_data['count'] = len(heatmap_data['data'])
        
        return Response(heatmap_data)
    
    except ValueError as e:
        return Response({
            'error': f'Invalid parameter format: {str(e)}',
            'data': [],
            'center': {'lat': -6.0, 'lon': 35.0},
            'count': 0
        }, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        print(f"Heatmap API error: {e}")
        return Response({
            'error': f'Internal server error: {str(e)}',
            'data': [],
            'center': {'lat': -6.0, 'lon': 35.0},
            'count': 0
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def api_time_series(request):
    """API endpoint for time series data."""
    if not initialize_prediction_service():
        return Response({'error': 'Service not available'}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
    
    try:
        time_series = prediction_service.get_time_series_analysis()
        return Response(time_series)
        
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def api_regional_analysis(request):
    """API endpoint for regional analysis."""
    if not initialize_prediction_service():
        return Response({'error': 'Service not available'}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
    
    try:
        lat = request.query_params.get('lat')
        lon = request.query_params.get('lon')
        radius = request.query_params.get('radius', 1.0)
        
        if lat and lon:
            lat = float(lat)
            lon = float(lon)
            radius = float(radius)
            
            # Get regional analysis
            regional_data = prediction_service.get_regional_analysis(lat, lon, radius)
            return Response(regional_data)
        else:
            # Get overall analysis
            df = prediction_service.data_processor.df
            
            regional_data = {
                'stats': {
                    'count': len(df),
                    'avg_magnitude': round(df['Mag'].mean(), 2),
                    'max_magnitude': float(df['Mag'].max()),
                    'min_magnitude': float(df['Mag'].min()),
                    'std_magnitude': round(df['Mag'].std(), 2),
                    'recent_events': len(df[df['Year'] >= datetime.now().year - 5]),
                }
            }
            return Response(regional_data)
        
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@permission_classes([IsAdminUser])
def api_train_model(request):
    """API endpoint for training/retraining ML models."""
    if not initialize_prediction_service():
        return Response({'error': 'Prediction service not available'}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
    
    try:
        # Get training parameters
        test_size = float(request.data.get('test_size', 0.2))
        random_state = int(request.data.get('random_state', 42))
        
        # Train models
        results = prediction_service.train_new_model(test_size, random_state)
        
        # Save model metrics
        metrics_list = []
        for model_name, metrics in results.items():
            model_metric = ModelMetrics.objects.create(
                model_name=model_name,
                version=datetime.now().strftime('%Y%m%d'),
                rmse=metrics['rmse'],
                mae=metrics['mae'],
                r2_score=metrics['r2'],
                feature_count=len(prediction_service.data_processor.create_features().columns) - 1,
                training_samples=int(len(prediction_service.data_processor.df) * (1 - test_size)),
                is_active=(model_name == prediction_service.model_trainer.best_model_name)
            )
            metrics_list.append(ModelMetricsSerializer(model_metric).data)
        
        return Response({
            'success': True,
            'best_model': prediction_service.model_trainer.best_model_name,
            'metrics': metrics_list
        })
        
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@permission_classes([IsAdminUser])
def api_import_data(request):
    """API endpoint for importing earthquake data."""
    if not request.FILES.get('data_file'):
        return Response({'error': 'No file uploaded'}, status=status.HTTP_400_BAD_REQUEST)
    
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
                return Response({'error': f"Required column '{col}' not found in data file"}, 
                               status=status.HTTP_400_BAD_REQUEST)
        
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
        
        return Response({
            'success': True,
            'records_imported': count
        })
        
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)