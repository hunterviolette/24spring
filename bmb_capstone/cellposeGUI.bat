@echo off

REM Check if venv\Scripts\activate exists
if not exist venv\Scripts\activate (
    echo Virtual environment not found. Creating and installing Cellpose[gui]...
    python -m venv venv
    call venv\Scripts\activate
    pip install cellpose[gui] cellpose==2.2.3
) else (
    REM Activate virtual environment
    call venv\Scripts\activate
)

REM Run cellpose
python -m cellpose

REM Deactivate virtual environment
deactivate