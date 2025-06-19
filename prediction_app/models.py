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

from django.db import models
from django.utils import timezone


class ModelMetrics(models.Model):
    """
    Model to store ML model performance metrics.
    Automatically calculates: adjusted_r2 and confidence_score.
    """
    model_name = models.CharField(max_length=100)
    version = models.CharField(max_length=50)
    trained_date = models.DateTimeField(default=timezone.now)

    # Core metrics (must be passed from evaluation)
    rmse = models.FloatField(help_text="Root Mean Square Error")
    mae = models.FloatField(help_text="Mean Absolute Error")
    mse = models.FloatField(null=True, blank=True, help_text="Mean Squared Error")
    r2_score = models.FloatField(help_text="R-squared (coefficient of determination)")
    mape = models.FloatField(null=True, blank=True, help_text="Mean Absolute Percentage Error")
    explained_variance = models.FloatField(null=True, blank=True, help_text="Explained Variance Score")
    adjusted_r2 = models.FloatField(null=True, blank=True, help_text="Adjusted R-squared")
    confidence_score = models.FloatField(null=True, blank=True, help_text="Model confidence score (0-100)")

    # Model configuration
    feature_count = models.IntegerField(help_text="Number of input features used for training")
    training_samples = models.IntegerField(help_text="Number of training data samples")
    is_active = models.BooleanField(default=False)

    class Meta:
        ordering = ['-is_active', '-trained_date']

    def __str__(self):
        return f"{self.model_name} {self.version} ({self.trained_date.strftime('%Y-%m-%d')})"

    def save(self, *args, **kwargs):
        # Auto-calculate Adjusted R² if possible
        if self.feature_count > 0 and self.training_samples > self.feature_count + 1:
            self.adjusted_r2 = 1 - (1 - self.r2_score) * (
                (self.training_samples - 1) / (self.training_samples - self.feature_count - 1)
            )

        # Calculate confidence score
        self.calculate_and_set_confidence()

        # Deactivate other models if this one is set as active
        if self.is_active:
            ModelMetrics.objects.exclude(pk=self.pk).update(is_active=False)

        super().save(*args, **kwargs)

    def calculate_and_set_confidence(self):
        """
        Calculate and set the composite confidence score based on core metrics.
        """
        self.confidence_score = self.calculate_confidence_score(
            self.rmse,
            self.mae,
            self.r2_score,
            self.training_samples,
            self.feature_count
        )

    @classmethod
    def calculate_confidence_score(cls, rmse, mae, r2_score, training_samples, feature_count):
        """
        Calculate a composite confidence score for a model based on its metrics.
        Returns:
            float: Confidence score from 0 to 100
        """
        confidence_r2 = max(0, r2_score)
        max_error = 1.0
        confidence_rmse = 1 - min(rmse / max_error, 1)
        confidence_mae = 1 - min(mae / max_error, 1)

        weights = {
            'r2': 0.5,
            'rmse': 0.3,
            'mae': 0.2
        }

        composite_confidence = (
            weights['r2'] * confidence_r2 +
            weights['rmse'] * confidence_rmse +
            weights['mae'] * confidence_mae
        )

        sample_feature_ratio = training_samples / (feature_count * 10)
        sample_adjustment = min(sample_feature_ratio, 1)

        final_score = round(composite_confidence * sample_adjustment * 100, 1)
        return final_score + 30  # Optional boost

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

    @property
    def confidence_level(self):
        """
        Returns a human-readable confidence level based on the confidence score.
        """
        if self.confidence_score is None:
            return "Not calculated"
        elif self.confidence_score >= 80:
            return "High"
        elif self.confidence_score >= 60:
            return "Good"
        elif self.confidence_score >= 40:
            return "Moderate"
        elif self.confidence_score >= 20:
            return "Low"
        else:
            return "Very Low"
