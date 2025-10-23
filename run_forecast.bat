@echo off
echo 24-Hour Electricity Demand Forecasting
echo =====================================
echo.

pip install -r requirements.txt
python forecast.py --city Bareilly --with_weather true

echo.
echo =====================================
echo COMPLETE - Check results/ folder
echo =====================================
pause