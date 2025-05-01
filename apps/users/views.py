from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, get_user_model
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.urls import reverse_lazy
from django.views import generic
from django.contrib.auth.views import (
    LoginView, LogoutView, 
    PasswordResetView, PasswordResetDoneView, 
    PasswordResetConfirmView, PasswordResetCompleteView
)
from .forms import (
    CustomUserCreationForm, CustomAuthenticationForm, 
    CustomPasswordResetForm, CustomSetPasswordForm, 
    UserProfileUpdateForm
)

User = get_user_model()

class RegisterView(generic.CreateView):
    form_class = CustomUserCreationForm
    template_name = 'auth/register.html'
    success_url = reverse_lazy('users:login')
    
    def form_valid(self, form):
        response = super().form_valid(form)
        print(response)
        messages.success(self.request, "Your account has been created successfully! Please log in.")
        return response


class CustomLoginView(LoginView):
    form_class = CustomAuthenticationForm
    template_name = 'auth/login.html'
    
    def form_valid(self, form):
        response = super().form_valid(form)
        messages.success(self.request, f"Welcome back, {self.request.user.username}!")
        return response


class CustomPasswordResetView(PasswordResetView):
    form_class = CustomPasswordResetForm
    template_name = 'auth/forgot_password.html'
    email_template_name = 'auth/password_reset_email.html'
    subject_template_name = 'auth/password_reset_subject.txt'
    success_url = reverse_lazy('users:password_reset_done')


class CustomPasswordResetDoneView(PasswordResetDoneView):
    template_name = 'auth/password_reset_done.html'


class CustomPasswordResetConfirmView(PasswordResetConfirmView):
    form_class = CustomSetPasswordForm
    template_name = 'auth/password_reset_confirm.html'
    success_url = reverse_lazy('users:password_reset_complete')


class CustomPasswordResetCompleteView(PasswordResetCompleteView):
    template_name = 'auth/password_reset_complete.html'


@login_required
def profile_view(request):
    """View for displaying and updating user profile"""
    if request.method == 'POST':
        form = UserProfileUpdateForm(request.POST, request.FILES, instance=request.user)
        if form.is_valid():
            form.save()
            messages.success(request, "Your profile has been updated successfully!")
            return redirect('users:profile')
    else:
        form = UserProfileUpdateForm(instance=request.user)
    
    return render(request, 'users/profile.html', {'form': form})


# HOME
@login_required
def dashboard(request):
    return render(request, 'prediction/index.html')