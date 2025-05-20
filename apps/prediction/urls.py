
from django.urls import path
from . import views

app_name = 'prediction'

urlpatterns = [
    path('dashboard/', views.index, name="dashboard"),
    path('historical-data/', views.historical_data, name="historical_data"),
    path('maps/', views.maps, name="maps"),
    path('scale/', views.scale, name="scale"),
    path('about/', views.about, name="about"),
    path('prediction/', views.prediction, name="prediction"),
]
