from django.urls import path
from . import views

app_name = 'ai'

urlpatterns = [
    path('', views.index, name='index'),
    path('predict/', views.predict, name='predict'),
    path('analytics/', views.analytics, name='analytics'),
    path('train-model/', views.train_model, name='train_model'),
    path('model-metrics/', views.model_metrics, name='model_metrics'),
    path('earthquake-data/', views.earthquake_data, name='earthquake_data'),
    path('import-data/', views.import_data, name='import_data'),
    
    # API endpoints
    path('api/heatmap/', views.api_heatmap, name='api_heatmap'),
    path('api/time-series/', views.api_time_series, name='api_time_series'),
]
