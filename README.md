# EVE Online Price Forecasting System

This script forecasts the next 14 days of prices for PLEX, Large Skill Injector, and Skill Extractor on the Jita market using a GRU neural network (TensorFlow). It sends daily forecast results via email with embedded price charts.

## 📦 Features

- Live price scrape from ESI API (Jita 4-4)
- Historical price forecasting using GRU
- Inline charts via `matplotlib`
- Single email with all forecast summaries and charts
- Ready for daily automation

## ⚙️ Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/EVEPriceForecast.git
   cd EVEPriceForecast