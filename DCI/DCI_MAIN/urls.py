from django.urls import path, include
from .views import predict_with_arg,give_value

urlpatterns = [
    path('predict=<str:stock_name_request>',predict_with_arg),
    path('give_value',give_value)
]