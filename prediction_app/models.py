from django.db import models
from django.utils import timezone
from django.urls import reverse

class Earthquake(models.Model):
    """
    Model to store earthquake data.
    """
    date = models.DateField()
    time = models.TimeField()
    latitude = models.FloatField()
    longitude = models.FloatField()
    depth = models.FloatField(help_text="Depth in kilometers")
    magnitude = models.FloatField()
    region = models.CharField(max_length=100, blank=True, null=True)
    imported_on = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ['-date', '-time']
        indexes = [
            models.Index(fields=['latitude', 'longitude']),
            models.Index(fields=['date']),
            models.Index(fields=['magnitude']),
        ]

    def __str__(self):
        return f"M{self.magnitude} at ({self.latitude:.2f}, {self.longitude:.2f}) on {self.date}"

    @property
    def datetime(self):
        """Combine date and time into a single datetime string."""
        return timezone.datetime.combine(self.date, self.time)

    @property
    def location(self):
        """Use region or fallback to lat/lon."""
        return self.region or f"({self.latitude:.2f}, {self.longitude:.2f})"

    @property
    def tectonic_plate(self):
        """Placeholder: return region or Unknown."""
        # Implement your tectonic plate lookup here
        return getattr(self, 'region', 'Unknown')
        return f"M{self.magnitude} at ({self.latitude:.2f}, {self.longitude:.2f}) on {self.date}"


class PredictionLog(models.Model):
    """
    Model to store prediction logs.
    """
    timestamp = models.DateTimeField(default=timezone.now)
    latitude = models.FloatField()
    longitude = models.FloatField()
    depth = models.FloatField(help_text="Depth in kilometers")
    predicted_magnitude = models.FloatField()
    
    # Additional context
    model_version = models.CharField(max_length=100, blank=True, null=True)
    confidence = models.FloatField(blank=True, null=True)
    
    class Meta:
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"Prediction: M{self.predicted_magnitude:.2f} at ({self.latitude:.2f}, {self.longitude:.2f})"


class ModelMetrics(models.Model):
    """
    Model to store ML model performance metrics.
    """
    model_name = models.CharField(max_length=100)
    version = models.CharField(max_length=50)
    trained_date = models.DateTimeField(default=timezone.now)
    
    # Performance metrics
    rmse = models.FloatField(help_text="Root Mean Square Error")
    mae = models.FloatField(help_text="Mean Absolute Error")
    r2_score = models.FloatField(help_text="R-squared")
    
    # Model details
    feature_count = models.IntegerField()
    training_samples = models.IntegerField()
    is_active = models.BooleanField(default=False)
            
    class Meta:
        ordering = ['-trained_date']
            
    def __str__(self):
        return f"{self.model_name} v{self.version} ({self.trained_date.strftime('%Y-%m-%d')})"
    
    def save(self, *args, **kwargs):
        if self.is_active:
            # Set all other models to inactive
            ModelMetrics.objects.all().update(is_active=False)
        super().save(*args, **kwargs)

    @classmethod
    def get_active_model_name(cls):
        active_model = cls.objects.filter(is_active=True).first()
        if not active_model:
            return None
        return ' '.join(word.capitalize() for word in active_model.model_name.split('_'))

    @classmethod
    def get_active_model_version(cls):
        active_model = cls.objects.filter(is_active=True).first()
        return active_model.version if active_model else None