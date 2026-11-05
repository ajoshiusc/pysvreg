@echo off
setlocal
if "%PYTHON%"=="" set PYTHON=python
"%PYTHON%" "%~dp0..\dev\pysvreg\compat_cli.py" generate_vol_param_stats_xls %*
exit /b %ERRORLEVEL%
