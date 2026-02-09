@echo off
echo ========================================
echo Starting Advanced Model Training
echo ========================================
echo.
echo This will train with:
echo   - Advanced data augmentation
echo   - Cross-validation (3 folds)
echo   - Better architecture
echo   - Comprehensive evaluation
echo.
echo Training will run in the background...
echo.

set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
python train-advanced-model.py --data-dir training-data --epochs 200 --batch-size 16 --augment 5 --folds 3

pause

