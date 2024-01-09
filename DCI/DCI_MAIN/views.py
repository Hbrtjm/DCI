from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import HttpResponse, render
from rest_framework.views import APIView
from rest_framework import generics
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .coin_search import coin_search
from requests import get
import json
# Create your views here.

@api_view(['GET'])
def predict(request):
    print(request)
    context = {
        "name":"thing"
    }
    return Response(context,status=status.HTTP_200_OK) 

@api_view(['GET'])
def give_value(request):
    context = coin_search()
    return Response(context,status=status.HTTP_200_OK)