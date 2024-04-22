from django.urls import path
from . import views
from .views import predict_business

urlpatterns = [
    path('predict/', predict_business, name='predict_business'),
]