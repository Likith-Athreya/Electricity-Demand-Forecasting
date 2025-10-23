# 24-Hour Electricity Demand Forecasting

Complete electricity demand forecasting solution for Bareilly, India.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run forecast
python forecast.py --city Bareilly --with_weather true
```

## Project Structure

```
assignment/
├── forecast.py             # Main script (150 lines)
├── requirements.txt         # Dependencies
├── README.md               # This file
├── run_forecast.bat        # Windows batch
├── run_forecast.sh         # Unix shell
├── results/                # Generated outputs
│   ├── forecast_T_plus_24.csv
│   ├── metrics.csv
│   ├── forecast_overlay.png
│   └── horizon_mae.png
└── reports/                # Generated reports
    └── fast_track_report.md
```

## Features

- **Data Processing**: 3-minute → hourly aggregation with outlier handling
- **Models**: Seasonal naive baseline + Ridge regression ML model
- **Weather**: Open-Meteo API integration (optional)
- **Outputs**: 24-hour forecast with confidence intervals
- **Visualization**: 3-day actuals + forecast, horizon-wise MAE
- **Report**: Professional 2-page analysis

## Usage

```bash
python forecast.py [OPTIONS]

Options:
  --city TEXT              City name (default: Bareilly)
  --with_weather BOOLEAN   Include weather data (default: true)
```

## Outputs

- **forecast_T_plus_24.csv**: 24-hour predictions with quantiles
- **metrics.csv**: Model performance (MAE, WAPE)
- **forecast_overlay.png**: 3-day actuals + forecast plot
- **horizon_mae.png**: Horizon-wise error analysis
- **fast_track_report.md**: Professional analysis report

## Data Sources

- **Demand Data**: [Kaggle Smart Meter Dataset](https://www.kaggle.com/datasets/jehanbhathena/smart-meter-datamathura-and-bareilly)
- **Weather Data**: [Open-Meteo API](https://open-meteo.com/en/docs)

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn, matplotlib, requests

## Notes

- Uses synthetic data if Kaggle dataset unavailable
- Single command execution for reproducibility
- All outputs saved to `results/` folder