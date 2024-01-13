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
from .Model_and_others.trading_model import train_model, predict
from .Model_and_others.data_request import StockDatabaseManager
from pathlib import Path
import json
import os
# Create your views here.

@api_view(['GET'])
def predict(request):
    try:
        data = josn.loads(request)
        stock_name = data.get('stock_name')
        try:
            db_manager = StockDatabaseManager(f'stock_data.db')
        except Exception as e:
            print(f"An error has occured: {e}")
            return Response(status=status.HTTP_500_BAD_REQUEST)
    except Exception as e:
        print(f"An error has occured: {e}")
        return Response(status=status.HTTP_500_BAD_REQUEST)

@csrf_exempt
@api_view(['GET'])
def predict_with_arg(request,stock_name_request):
    # try:
    print(request)
    stock_name = stock_name_request
    print(stock_name)
    # try:
    # Build paths inside the project like this: BASE_DIR / 'subdir'.
    BASE_DIR = Path(__file__).resolve().parent.parent
    print("Getting data")
    db_manager = StockDatabaseManager(os.path.join(BASE_DIR,"/Model_and_others/stock_data.db"))
    # Bad checking if stock_name is in the database
    print("Stock name")
    # stock_name = db_manager.load_database()[stock_name]
    print("Prediction")
    predictions_list = predict(stock_name)
    predictions_list = [(1,"Buy")]
    print(predictions_list)
    context = {
        "prediction":predictions_list[len(predictions_list)-1][1]
    }
        # except Exception as e:
        #     print(f"Due to exception {e} the model will be trained")
        #     context = {
        #         "model":"trained"
        #     }
        #     train_model(stock_name)
    return Response(context,status=status.HTTP_200_OK) 
    # except Exception as e:
    #     print(f"An err0r occured {e}")
    #     return Response(status=status.HTTP_404_NOT_FOUND)
    
@api_view(['GET'])
def give_value(request):
    context = coin_search()
    return Response(context,status=status.HTTP_200_OK)