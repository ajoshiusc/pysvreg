@echo off
setlocal
if "%PYTHON%"=="" set PYTHON=python
"%PYTHON%" "%~dp0..\dev\pysvreg\compat_cli.py" svreg_prepare_files %*
exit /b %ERRORLEVEL%
