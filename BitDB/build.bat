@echo off
setlocal enabledelayedexpansion

:: Check if user requested watchdog mode
if /I "%1"=="watch" goto :watch_mode
goto :start_build

:watch_mode
if exist "build\Watchdog.exe" (
    echo [*] Starting BitDB Native C++ Auto-Sync Watchdog on ./ingestor...
    build\Watchdog.exe %2 %3 %4 %5
    exit /b %ERRORLEVEL%
)
echo [*] Starting BitDB Auto-Sync Watchdog on ./ingestor [Python fallback]...
python scripts\db_watchdog.py %2 %3 %4 %5
exit /b %ERRORLEVEL%

:start_build

echo.
echo ==================================================
echo   BitDB Prototype-4 - Unified Build System
echo ==================================================
echo.

:: Ensure required directories exist
if not exist "build" mkdir "build"
if not exist "ingestor" mkdir "ingestor"
if not exist "DataStorage" mkdir "DataStorage"

:: Check if CMake is available
where cmake >nul 2>&1
:: Skip CMake as it has issues with NMake
goto :fallback
if %ERRORLEVEL% equ 0 (
    echo [*] CMake detected. Building via CMake...
    if not exist "build_cmake" mkdir "build_cmake"
    cmake -B build_cmake -S . -DCMAKE_BUILD_TYPE=Release
    if %ERRORLEVEL% neq 0 (
        echo [!] CMake configuration failed.
        goto :fallback
    )
    cmake --build build_cmake --config Release
    if %ERRORLEVEL% neq 0 (
        echo [!] CMake build failed.
        goto :fallback
    )
    goto :success
)

:fallback
echo [*] Building directly with GCC / Clang / MSVC...

:: Auto-detect Python include and libs (robust across virtual environments and system installs)
for /f "delims=" %%i in ('python -c "import sys, sysconfig, os; inc = sysconfig.get_path('include'); base_inc = os.path.join(sys.base_prefix, 'include'); print(base_inc if os.path.isfile(os.path.join(base_inc, 'Python.h')) else inc)" 2^>nul') do set PY_INC=%%i
for /f "delims=" %%i in ('python -c "import sys, os; p = os.path.join(sys.base_prefix, 'libs'); print(p if os.path.isdir(p) else os.path.join(sys.prefix, 'libs'))" 2^>nul') do set PY_LIB=%%i
for /f "delims=" %%i in ('python -c "import sys; print(f'python{sys.version_info.major}{sys.version_info.minor}')" 2^>nul') do set PY_NAME=%%i

if not defined PY_INC (
    echo [!] ERROR: Python not found in PATH. Please install Python 3.
    exit /b 1
)

echo   Python Include: %PY_INC%
echo   Python Lib:     %PY_LIB%

:: Compiler detection
set CXX=g++
where g++ >nul 2>&1
if %ERRORLEVEL% neq 0 (
    where clang++ >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        set CXX=clang++
    ) else (
        set CXX=cl
    )
)

echo   Compiler:       %CXX%
echo.

set FLAGS=-std=c++17 -O2 -mavx2 -mpopcnt -I"src" -I"%PY_INC%"
set LIBS=-L"%PY_LIB%" -l%PY_NAME%

:: Terminate any lingering instances holding file locks on binaries
taskkill /F /IM BitDBSearch.exe >nul 2>&1
taskkill /F /IM Build.exe >nul 2>&1
taskkill /F /IM print_catalog.exe >nul 2>&1
taskkill /F /IM print_segment_dir.exe >nul 2>&1
taskkill /F /IM test_suite.exe >nul 2>&1
taskkill /F /IM Watchdog.exe >nul 2>&1
taskkill /F /IM HaltonProbes.exe >nul 2>&1

echo [1/7] Compiling Build.exe...
%CXX% %FLAGS% src\Build.cpp src\embed.cpp %LIBS% -o build\Build.exe
if %ERRORLEVEL% neq 0 (
    echo [!] FAILED to compile Build.exe.
    echo     Tip: If 'Permission denied', check if Build.exe is currently running.
    exit /b 1
)

echo [2/7] Compiling BitDBSearch.exe...
%CXX% %FLAGS% src\Search.cpp src\embed.cpp %LIBS% -o build\BitDBSearch.exe
if %ERRORLEVEL% neq 0 (
    echo [!] FAILED to compile BitDBSearch.exe.
    echo     Tip: If 'Permission denied', check if BitDBSearch.exe is currently running in another terminal.
    exit /b 1
)

echo [3/7] Compiling Watchdog.exe...
%CXX% -std=c++17 -O2 -I"src" src\Watchdog.cpp -o build\Watchdog.exe
if %ERRORLEVEL% neq 0 ( echo [!] FAILED to compile Watchdog.exe & exit /b 1 )

echo [4/7] Compiling HaltonProbes.exe...
%CXX% -std=c++17 -O2 -I"src" src\HaltonProbes.cpp -o build\HaltonProbes.exe
if %ERRORLEVEL% neq 0 ( echo [!] FAILED to compile HaltonProbes.exe & exit /b 1 )

echo [5/7] Compiling print_catalog.exe...
%CXX% -std=c++17 -O2 -I"src" printers\print_catalog.cpp -o build\print_catalog.exe
if %ERRORLEVEL% neq 0 ( echo [!] FAILED to compile print_catalog.exe & exit /b 1 )

echo [6/7] Compiling print_segment_dir.exe...
%CXX% -std=c++17 -O2 -I"src" printers\print_segment_dir.cpp -o build\print_segment_dir.exe
if %ERRORLEVEL% neq 0 ( echo [!] FAILED to compile print_segment_dir.exe & exit /b 1 )

echo [7/7] Compiling test_suite.exe...
%CXX% -std=c++17 -O2 -mavx2 -mpopcnt -I"src" src\test_suite.cpp -o build\test_suite.exe
if %ERRORLEVEL% neq 0 ( echo [!] FAILED to compile test_suite.exe & exit /b 1 )

:success
echo.
echo ==================================================
echo   BUILD SUCCESSFUL
echo ==================================================
echo.
echo   Binaries placed in:  Prototype-4\build\
echo.
echo   USAGE:
echo     1. Drop PDFs:   Prototype-4\ingestor\
echo     2. Ingest:      build\Build.exe
echo     3. Query:       build\BitDBSearch.exe "your search query" [N] [probes]
echo        List Files:  build\BitDBSearch.exe --list
echo        Interactive: build\BitDBSearch.exe --interactive --probes 4
echo     4. Rebuild:     build\Build.exe --rebuild
echo     5. Watchdog:    build\Build.exe --watch (or build.bat watch)
echo     6. Catalog:     build\print_catalog.exe
echo     7. Segments:    build\print_segment_dir.exe
echo ==================================================
echo.
exit /b 0
