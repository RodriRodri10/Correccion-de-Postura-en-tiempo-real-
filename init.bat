@echo off
REM Inicializacion rapida del proyecto (Windows, cmd.exe).
REM
REM Uso:
REM   init.bat            Crea el entorno virtual (.venv) e instala dependencias.
REM   init.bat --docker   Ademas levanta la base de datos (PostgreSQL + PostgREST).
setlocal
cd /d "%~dp0"

REM 1) Localizar Python 3.11 (preferido por mediapipe==0.10.14).
set "PY=py -3.11"
%PY% --version >nul 2>&1 || set "PY=python"
%PY% --version >nul 2>&1 || (
  echo ERROR: no se encontro Python. Instala Python 3.11 desde python.org
  exit /b 1
)

REM 1b) Advertir si no es Python 3.11 (mediapipe==0.10.14 no es confiable en 3.12/3.13).
for /f "tokens=2 delims= " %%v in ('%PY% --version 2^>^&1') do set "PYVER=%%v"
echo %PYVER% | findstr /b "3.11" >nul || (
  echo ADVERTENCIA: usando Python %PYVER% ^(se recomienda 3.11^).
  echo   mediapipe==0.10.14 puede fallar al instalar fuera de Python 3.11.
)

REM 2) Entorno virtual.
echo == Entorno virtual (.venv) ==
if not exist .venv ( %PY% -m venv .venv )

REM 3) Dependencias.
echo == Instalando dependencias ==
".venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
".venv\Scripts\python.exe" -m pip install -r requirements.txt

REM 4) Base de datos opcional (Docker).
if "%~1"=="--docker" (
  echo == Levantando base de datos (Docker) ==
  where docker >nul 2>&1 || ( echo ERROR: Docker no esta instalado. & exit /b 1 )
  if not exist .env ( copy .env.example .env >nul )
  docker compose up -d db postgrest
)

echo.
echo Listo. Siguientes pasos:
echo   .venv\Scripts\activate
echo   streamlit run app.py
if not "%~1"=="--docker" (
  echo.
  echo Persistencia opcional ^(base de datos^): docker compose up -d db postgrest
)
endlocal
