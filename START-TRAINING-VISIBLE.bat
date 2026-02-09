@echo off
chcp 65001 >nul
echo ======================================================================
echo STARTING TRAINING - 100 EPOCHS
echo ======================================================================
echo.
echo Training will run in THIS window so you can see progress
echo Press Ctrl+C to stop training
echo.
echo ======================================================================
echo.
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
python train-with-available-data.py --data-dir training-data --epochs 100 --batch-size 8
echo.
echo ======================================================================
echo Training completed or stopped
echo ======================================================================
pause

