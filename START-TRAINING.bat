@echo off
echo ======================================================================
echo STARTING TRAINING - 100 EPOCHS
echo ======================================================================
echo.
echo Training will run in the background
echo Check lstm-model-trained folder for progress
echo.
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
python train-with-available-data.py --data-dir training-data --epochs 100 --batch-size 8
pause

