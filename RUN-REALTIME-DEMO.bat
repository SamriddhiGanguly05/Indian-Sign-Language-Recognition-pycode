@echo off
echo ======================================================================
echo REAL-TIME SIGN LANGUAGE DETECTION
echo ======================================================================
echo.
echo Starting webcam detection with landmarks...
echo Press 'q' to quit
echo.
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
python realtime-detection.py
pause

