@echo off
setlocal enabledelayedexpansion

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

:: Auto-detect Python include and libs
for /f "delims=" %%i in ('python -c "import sysconfig; print(sysconfig.get_path('include'))" 2^>nul') do set PY_INC=%%i
for /f "delims=" %%i in ('python -c "import sysconfig, os; print(os.path.join(sysconfig.get_config_var('base'), 'libs'))" 2^>nul') do set PY_LIB=%%i
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

echo [1/4] Compiling Build.exe...
%CXX% %FLAGS% src\Build.cpp src\embed.cpp %LIBS% -o build\Build.exe
if %ERRORLEVEL% neq 0 ( echo [!] FAILED to compile Build.exe & exit /b 1 )

echo [2/4] Compiling BitDBSearch.exe...
%CXX% %FLAGS% src\Search.cpp src\embed.cpp %LIBS% -o build\BitDBSearch.exe
if %ERRORLEVEL% neq 0 ( echo [!] FAILED to compile BitDBSearch.exe & exit /b 1 )

echo [3/4] Compiling print_catalog.exe...
%CXX% -std=c++17 -O2 -I"src" printers\print_catalog.cpp -o build\print_catalog.exe
if %ERRORLEVEL% neq 0 ( echo [!] FAILED to compile print_catalog.exe & exit /b 1 )

echo [4/5] Compiling print_segment_dir.exe...
%CXX% -std=c++17 -O2 -I"src" printers\print_segment_dir.cpp -o build\print_segment_dir.exe
if %ERRORLEVEL% neq 0 ( echo [!] FAILED to compile print_segment_dir.exe & exit /b 1 )

echo [5/5] Compiling test_suite.exe...
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
echo        Interactive: build\BitDBSearch.exe --interactive --probes 4
echo     4. Rebuild:     build\Build.exe --rebuild
echo     5. Catalog:     build\print_catalog.exe
echo     6. Segments:    build\print_segment_dir.exe
echo ==================================================
echo.
exit /b 0
