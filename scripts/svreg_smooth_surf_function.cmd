@echo off
setlocal
if "%PYTHON%"=="" set PYTHON=python
"%PYTHON%" "%~dp0..\dev\pysvreg\compat_cli.py" svreg_smooth_surf_function %*
exit /b %ERRORLEVEL%
