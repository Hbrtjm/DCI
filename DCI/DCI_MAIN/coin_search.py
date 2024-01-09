import requests 
from datetime import datetime, timedelta
def coin_search(coin_name):
    url = f"https://www.binance.com/pl/trade/{coin_name}_USDT?_from=markets&type=spot"
    # try:
    def debug():
       
        # nonlocal url
        # response = requests.get(url)
        # content = response.content
        # print(response)
        # print(content)

        # Fetch the top 100 cryptocurrencies from CoinGecko API
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 100,
            "page": 1,
            "sparkline": False
        }
        response = requests.get(url, params=params)
        data = response.json()

        # Extract the name and symbol for each cryptocurrency
        crypto_list = ['BTC', 'ETH']
        for crypto in data:
            name = crypto['name']
            symbol = crypto['symbol'].upper()
            crypto_list.append(f"{name} ({symbol})")
        print(crypto_list[:5])  # Display the first 5 entries as a sample
                # Convert the current time to Unix timestamp and subtract one day (86400 seconds)
        end_time = int(datetime.now().timestamp()) - 86400
        start_time = end_time - 86400

        # API URL for historical data
        url = f"https://api.coingecko.com/api/v3/coins/{coin_name}/market_chart/range"
        params = {
            'vs_currency': 'usd',
            'from': start_time,
            'to': end_time
        }

        response = requests.get(url, params=params)
        data = response.json()

        # The price data is usually in a list of [timestamp, price] pairs.
        # You might want to average the prices or take the first/last price of the day.
        # Here, I'm taking the last recorded price of the day.
        last_recorded_price = data['prices'][-1][1]
        print(data['prices'])
        for name in crypto_list:
            name = name.split(' ')[0].lower()
                    # API URL for historical data
            url = f"https://api.coingecko.com/api/v3/coins/{name}/market_chart/range"
            params = {
                'vs_currency': 'usd',
                'from': start_time,
                'to': end_time
            }

            response = requests.get(url, params=params)
            data = response.json()

            # The price data is usually in a list of [timestamp, price] pairs.
            # You might want to average the prices or take the first/last price of the day.
            # Here, I'm taking the last recorded price of the day.
            last_recorded_price = data['prices'][-1][1]
            print(name)
            print(data['prices'])
        # except:
    #     print("Something went wrong with sending the request")
    # finally:
    #     return []
    debug()
# coin_search("bitcoin")