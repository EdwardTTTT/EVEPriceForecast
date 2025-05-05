import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import smtplib
from datetime import timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from sklearn.preprocessing import MinMaxScaler

# 1. Fetch Jita (live) price
def fetch_jita_price(type_id):
    buy_url = f"https://esi.evetech.net/latest/markets/10000002/orders/?order_type=buy&type_id={type_id}"
    sell_url = f"https://esi.evetech.net/latest/markets/10000002/orders/?order_type=sell&type_id={type_id}"
    buy_data = requests.get(buy_url).json()
    sell_data = requests.get(sell_url).json()
    highest_buy = max(order['price'] for order in buy_data if order['location_id'] == 60003760)
    lowest_sell = min(order['price'] for order in sell_data if order['location_id'] == 60003760)
    return (highest_buy + lowest_sell) / 2

# 2. Fetch price history
def fetch_price_history(type_id):
    url = f'https://esi.evetech.net/latest/markets/10000002/history/?type_id={type_id}'
    df = pd.DataFrame(requests.get(url).json())
    df['date'] = pd.to_datetime(df['date'])
    df['price'] = (df['highest'] + df['lowest']) / 2
    return df[['date', 'price']]

# 3. Create sequences for model
def create_sequences(data, window=30):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

# 4. Forecast using GRU or LSTM
def forecast_lstm(df, days=14, window=30, model_type="gru"):
    series = df.set_index('date').asfreq('D').interpolate()['price']
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))
    X, y = create_sequences(scaled, window)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    RNN = GRU if model_type == "gru" else LSTM
    model.add(RNN(50, input_shape=(window, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, verbose=0)

    forecast_scaled = []
    last_window = scaled[-window:].reshape(1, window, 1)
    for _ in range(days):
        pred = model.predict(last_window, verbose=0)[0][0]
        forecast_scaled.append(pred)
        last_window = np.append(last_window[:, 1:, :], [[[pred]]], axis=1)

    forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()
    start = series.index[-1] + pd.Timedelta(days=1)
    return pd.Series(forecast, index=pd.date_range(start=start, periods=days))

# 5. Plotting
def create_plot(name, df, forecast):
    full_df = df.set_index('date').asfreq('D').interpolate()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(full_df.index, full_df['price'], label="Historical", color='blue')
    ax.plot(forecast.index, forecast.values, label="Forecast", color='red')
    ax.set_title(f"{name} (Jita): Historical + 14-Day Forecast")
    ax.set_ylabel("ISK")
    ax.legend()
    ax.grid(True)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf

# 6. Email
def send_combined_email(results):
    msg = MIMEMultipart()
    msg['Subject'] = "EVE Online Market Forecast (PLEX, LSI, SE)"
    msg['From'] = 'Your email address'
    msg['To'] = 'Your email address'

    html = "<h2>EVE Online Daily Price Report (GRU Forecast)</h2>"
    for name, today_price, forecast, cid in results:
        html += f"""
        <h3>{name} (Jita)</h3>
        <ul>
            <li>Today's Price: {today_price:,.2f} ISK</li>
            <li>Tomorrow: {forecast[1]:,.2f} ISK</li>
            <li>In 1 Week: {forecast[6]:,.2f} ISK</li>
            <li>In 2 Weeks: {forecast[13]:,.2f} ISK</li>
        </ul>
        <img src="cid:{cid}" width="700"><br><br>
        """
    msg.attach(MIMEText(html, 'html'))

    for name, _, forecast, cid in results:
        df = fetch_price_history(type_ids[name])
        buf = create_plot(name, df, forecast)
        img = MIMEImage(buf.read())
        img.add_header('Content-ID', f'<{cid}>')
        msg.attach(img)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login('Your email address', 'Your gmail app password')
        server.send_message(msg)

# 7. Main
type_ids = {
    'PLEX': 44992,
    'LSI': 40520,
    'SE': 40519
}

def run():
    results = []
    for i, (name, type_id) in enumerate(type_ids.items()):
        try:
            print(f"Processing {name}...")
            today_price = fetch_jita_price(type_id)
            df = fetch_price_history(type_id)
            forecast = forecast_lstm(df, days=14, model_type="gru")
            cid = f"chart_{i}"
            results.append((name, today_price, forecast, cid))
        except Exception as e:
            print(f"Error processing {name}: {e}")
    send_combined_email(results)
    print("Email sent.")

if __name__ == "__main__":
    run()
