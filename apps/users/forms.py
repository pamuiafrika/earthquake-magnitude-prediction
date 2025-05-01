from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm, PasswordResetForm, SetPasswordForm
from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError

User = get_user_model()

class CustomUserCreationForm(UserCreationForm):
    """
    Form for user registration with custom styling
    """
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={
            'class': 'w-full px-3 py-4 border border-gray-300 rounded-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500'
        })
    )
    
    password1 = forms.CharField(
        label="Password",
        widget=forms.PasswordInput(attrs={
            'class': 'w-full px-3 py-4 border border-gray-300 rounded-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500'
        })
    )
    
    password2 = forms.CharField(
        label="Confirm Password",
        widget=forms.PasswordInput(attrs={
            'class': 'w-full px-3 py-4 border border-gray-300 rounded-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500'
        })
    )
    
    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2')
        widgets = {
            'username': forms.TextInput(attrs={
                'class': 'w-full px-3 py-4 border border-gray-300 rounded-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500'
            }),
        }
    
    def clean_email(self):
        email = self.cleaned_data['email']
        if User.objects.filter(email=email).exists():
            raise ValidationError("A user with that email already exists.")
        return email


class CustomAuthenticationForm(AuthenticationForm):
    """
    Form for user login with custom styling
    """
    username = forms.CharField(
        widget=forms.TextInput(attrs={
            'class': 'w-full px-3 py-4 border border-gray-300 rounded-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500'
        })
    )
    
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'w-full px-3 py-4 border border-gray-300 rounded-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500'
        })
    )


class CustomPasswordResetForm(PasswordResetForm):
    """
    Form for password reset request with custom styling
    """
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={
            'class': 'w-full px-3 py-4 border border-gray-300 rounded-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500'
        })
    )


class CustomSetPasswordForm(SetPasswordForm):
    """
    Form for setting a new password with custom styling
    """
    new_password1 = forms.CharField(
        label="New password",
        widget=forms.PasswordInput(attrs={
            'class': 'w-full px-3 py-4 border border-gray-300 rounded-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500'
        })
    )
    
    new_password2 = forms.CharField(
        label="Confirm new password",
        widget=forms.PasswordInput(attrs={
            'class': 'w-full px-3 py-4 border border-gray-300 rounded-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500'
        })
    )


class UserProfileUpdateForm(forms.ModelForm):
    """
    Form for updating user profile information
    """
    class Meta:
        model = User
        fields = ('username', 'email', 'profile_picture', 'bio', 'date_of_birth')
        widgets = {
            'username': forms.TextInput(attrs={
                'class': 'w-full px-3 py-4 border border-gray-300 rounded-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500'
            }),
            'email': forms.EmailInput(attrs={
                'class': 'w-full px-3 py-4 border border-gray-300 rounded-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500'
            }),
            'bio': forms.Textarea(attrs={
                'class': 'w-full px-3 py-4 border border-gray-300 rounded-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500'
            }),
            'date_of_birth': forms.DateInput(attrs={
                'class': 'w-full px-3 py-4 border border-gray-300 rounded-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500',
                'type': 'date'
            }),
            'profile_picture': forms.FileInput(attrs={
                'class': 'w-full px-3 py-4 border border-gray-300 rounded-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500'
            }),
        }