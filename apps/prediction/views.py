from django.shortcuts import render

# Create your views here.
def index(request):
    return render(request, 'prediction/index.html')

def historical_data(request):
    return render(request, 'prediction/historical_data.html')

def maps(request):
    return render(request, 'prediction/maps.html')

def scale(request):
    return render(request, 'prediction/scale.html')

def prediction(request):
    return render(request, 'prediction/prediction.html')