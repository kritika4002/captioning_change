@echo off
REM ============================================================
REM  GPU Training Launcher — RTX 3050 (tf_gpu_env)
REM  Uses: C:\project-major\tf_gpu_env\Scripts\python.exe
REM ============================================================

set PYTHON=C:\project-major\tf_gpu_env\Scripts\python.exe
set PROJECT=C:\project-major\captioning_change

echo.
echo ============================================================
echo  Disaster Change Captioning — GPU Training Run
echo  Python  : %PYTHON%
echo  Project : %PROJECT%
echo ============================================================
echo.

REM Check GPU availability
%PYTHON% -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND')"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Cannot load PyTorch. Aborting.
    pause
    exit /b 1
)

echo.
echo Installing / verifying required packages...
%PYTHON% -m pip install --quiet transformers>=5.0 peft>=0.19 open-clip-torch sacrebleu nltk rouge-score pillow numpy pandas pyyaml tqdm

echo.
echo Starting training...
echo Logs will be written to: %PROJECT%\logs\training.log
echo.

cd /d %PROJECT%
%PYTHON% scripts\train.py --config configs\config.yaml

echo.
echo ============================================================
echo  Training complete! Checkpoint saved in: checkpoints\best\
echo ============================================================
pause
