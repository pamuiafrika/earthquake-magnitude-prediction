from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views
from . import api_views

app_name = 'ai'

# API Router
router = DefaultRouter()
router.register(r'earthquakes', api_views.EarthquakeViewSet)
router.register(r'prediction-logs', api_views.PredictionLogViewSet)
router.register(r'model-metrics', api_views.ModelMetricsViewSet)

urlpatterns = [
    path('', views.index, name='index'),
    path('predict/', views.predict, name='predict'),
    path('analytics/', views.analytics, name='analytics'),
    path('train-model/', views.train_model, name='train_model'),
    path('model-metrics/', views.model_metrics, name='model_metrics'),
    path('earthquake-data/', views.earthquake_data, name='earthquake_data'),
    path('import-data/', views.import_data, name='import_data'),
    
    # # API endpoints
    # path('api/heatmap/', views.api_heatmap, name='api_heatmap'),
    # path('api/time-series/', views.api_time_series, name='api_time_series'),

    # API Endpoints
    path('api/', include(router.urls)),
    # earthquake data
    path('api/earthquake-data/', views.api_earthquake_data, name='api_earthquake_data'),
    path('export-earthquake-csv/', views.export_earthquake_csv, name='export_earthquake_csv'),
    
    
    path('api/statistics/', api_views.api_statistics, name='api_statistics'),
    path('api/predict/', api_views.api_predict, name='api_predict'),
    path('api/heatmap/', api_views.api_heatmap, name='api_heatmap'),
    path('api/time-series/', api_views.api_time_series, name='api_time_series'),
    path('api/regional-analysis/', api_views.api_regional_analysis, name='api_regional_analysis'),
    path('api/train-model/', api_views.api_train_model, name='api_train_model'),
    path('api/import-data/', api_views.api_import_data, name='api_import_data'),
]