# users/urls.py
from django.urls import path
from django.contrib.auth.views import LogoutView
from . import views

app_name = 'users'

urlpatterns = [
    # Authentication URLs
    path('', views.CustomLoginView.as_view(), name='login'),
    path('users/auth/register/', views.RegisterView.as_view(), name='register'),
    path('users/auth/logout/', LogoutView.as_view(), name='logout'),
    
    # Password Reset URLs
    path('users/auth/password-reset/', views.CustomPasswordResetView.as_view(), name='password_reset'),
    path('users/auth/password-reset/done/', views.CustomPasswordResetDoneView.as_view(), name='password_reset_done'),
    path('users/auth/password-reset/<uidb64>/<token>/', views.CustomPasswordResetConfirmView.as_view(), name='password_reset_confirm'),
    path('users/auth/password-reset/complete/', views.CustomPasswordResetCompleteView.as_view(), name='password_reset_complete'),
    
    # Profile URL
    path('users/profile/', views.profile_view, name='profile'),
    path('dashboard/', views.dashboard, name='home')
]