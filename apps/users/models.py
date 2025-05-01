from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils.translation import gettext_lazy as _

class CustomUser(AbstractUser):
    """
    Custom user model that extends Django's AbstractUser
    """
    email = models.EmailField(_('email address'), unique=True)
    profile_picture = models.ImageField(upload_to='users/profile_pics/', null=True, blank=True)
    bio = models.TextField(max_length=500, blank=True)
    date_of_birth = models.DateField(null=True, blank=True)
    
    USERNAME_FIELD = 'username'
    REQUIRED_FIELDS = ['email']
    
    def __str__(self):
        return self.email