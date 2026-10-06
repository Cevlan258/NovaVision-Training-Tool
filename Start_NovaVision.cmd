@echo off
cd /d "%~dp0"
"%~dp0python.exe" -B "%~dp0portable_launch.py"
if errorlevel 1 pause
