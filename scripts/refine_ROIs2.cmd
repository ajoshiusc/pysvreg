@echo off
setlocal
if "%PYTHON%"=="" set PYTHON=python
"%PYTHON%" "%~dp0..\dev\pysvreg\compat_cli.py" refine_ROIs2 %*
exit /b %ERRORLEVEL%
