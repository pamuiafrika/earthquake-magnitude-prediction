from django.contrib import admin
from .models import Earthquake, PredictionLog,ModelMetrics

# Register your models here.
admin.site.register(Earthquake)
admin.site.register(PredictionLog)
admin.site.register(ModelMetrics)