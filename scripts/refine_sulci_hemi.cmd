@echo off
setlocal
if "%PYTHON%"=="" set PYTHON=python
"%PYTHON%" "%~dp0..\dev\pysvreg\compat_cli.py" refine_sulci_hemi %*
exit /b %ERRORLEVEL%
