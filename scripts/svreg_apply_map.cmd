@echo off
setlocal
if "%PYTHON%"=="" set PYTHON=python
"%PYTHON%" "%~dp0..\dev\pysvreg\compat_cli.py" svreg_apply_map %*
exit /b %ERRORLEVEL%
