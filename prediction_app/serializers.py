from rest_framework import serializers
from .models import Earthquake, PredictionLog, ModelMetrics


class EarthquakeSerializer(serializers.ModelSerializer):
    """Serializer for Earthquake model."""
    class Meta:
        model = Earthquake
        fields = '__all__'


class PredictionLogSerializer(serializers.ModelSerializer):
    """Serializer for PredictionLog model."""
    class Meta:
        model = PredictionLog
        fields = '__all__'


class ModelMetricsSerializer(serializers.ModelSerializer):
    """Serializer for ModelMetrics model."""
    class Meta:
        model = ModelMetrics
        fields = '__all__'


class PredictionRequestSerializer(serializers.Serializer):
    """Serializer for prediction requests."""
    latitude = serializers.FloatField(min_value=-90, max_value=90)
    longitude = serializers.FloatField(min_value=-180, max_value=180)
    depth = serializers.FloatField(min_value=0)
    
    def validate(self, data):
        """Validate the prediction request."""
        # Additional validation can be added here if needed
        return data