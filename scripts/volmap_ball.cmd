@echo off
setlocal
if "%PYTHON%"=="" set PYTHON=python
"%PYTHON%" "%~dp0..\dev\pysvreg\compat_cli.py" volmap_ball %*
exit /b %ERRORLEVEL%
