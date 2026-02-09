@echo off
chcp 65001 >nul
echo ======================================================================
echo REAL-TIME SIGN LANGUAGE DETECTION - OPENCV
echo ======================================================================
echo.
echo Starting webcam with landmark visualization...
echo.
echo Features:
echo   - MediaPipe landmarks (pose, hands)
echo   - Real-time predictions
echo   - Continuous detection (no delay)
echo.
echo Press 'q' to quit
echo.
echo ======================================================================
echo.
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
python realtime-detection.py
echo.
echo ======================================================================
echo Demo ended
echo ======================================================================
pause

