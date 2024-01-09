from django.urls import path, include
from .views import predict,give_value

urlpatterns = [
    path('predict',predict),
    path('give_value',give_value)
]