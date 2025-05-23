# Generated by Django 5.2 on 2025-05-02 10:31

import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ModelMetrics',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('model_name', models.CharField(max_length=100)),
                ('version', models.CharField(max_length=50)),
                ('trained_date', models.DateTimeField(default=django.utils.timezone.now)),
                ('rmse', models.FloatField(help_text='Root Mean Square Error')),
                ('mae', models.FloatField(help_text='Mean Absolute Error')),
                ('r2_score', models.FloatField(help_text='R-squared')),
                ('feature_count', models.IntegerField()),
                ('training_samples', models.IntegerField()),
                ('is_active', models.BooleanField(default=False)),
            ],
            options={
                'ordering': ['-trained_date'],
            },
        ),
        migrations.CreateModel(
            name='PredictionLog',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('timestamp', models.DateTimeField(default=django.utils.timezone.now)),
                ('latitude', models.FloatField()),
                ('longitude', models.FloatField()),
                ('depth', models.FloatField(help_text='Depth in kilometers')),
                ('predicted_magnitude', models.FloatField()),
                ('model_version', models.CharField(blank=True, max_length=100, null=True)),
                ('confidence', models.FloatField(blank=True, null=True)),
            ],
            options={
                'ordering': ['-timestamp'],
            },
        ),
        migrations.CreateModel(
            name='Earthquake',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date', models.DateField()),
                ('time', models.TimeField()),
                ('latitude', models.FloatField()),
                ('longitude', models.FloatField()),
                ('depth', models.FloatField(help_text='Depth in kilometers')),
                ('magnitude', models.FloatField()),
                ('region', models.CharField(blank=True, max_length=100, null=True)),
                ('imported_on', models.DateTimeField(default=django.utils.timezone.now)),
            ],
            options={
                'ordering': ['-date', '-time'],
                'indexes': [models.Index(fields=['latitude', 'longitude'], name='prediction__latitud_f858f7_idx'), models.Index(fields=['date'], name='prediction__date_df579f_idx'), models.Index(fields=['magnitude'], name='prediction__magnitu_bbfb65_idx')],
            },
        ),
    ]
