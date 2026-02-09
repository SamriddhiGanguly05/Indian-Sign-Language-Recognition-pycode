@echo off
chcp 65001 >nul
echo ======================================================================
echo TRAINING MODEL - 200 EPOCHS
echo ======================================================================
echo.
echo Training will run in THIS window so you can see progress
echo Press Ctrl+C to stop training
echo.
echo Best model will be saved automatically
echo.
echo ======================================================================
echo.
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
python train-with-available-data.py --data-dir training-data --epochs 200 --batch-size 8
echo.
echo ======================================================================
echo Training completed or stopped
echo ======================================================================
pause

