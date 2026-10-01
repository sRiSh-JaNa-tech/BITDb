@echo off
setlocal enabledelayedexpansion

REM =============================================================================
REM BitDB Prototype-4 Unified Build System
REM Supports: GCC (g++), Clang (clang++), MSVC (cl)
REM Targets: AVX2 Accelerated & Portable / Scalar Fallback
REM =============================================================================

REM Check if user requested watchdog mode
if /I "%~1"=="watch" goto :watch_mode
goto :start_build

:watch_mode
if exist "build\Watchdog.exe" goto :run_watchdog

echo [*] Compiling native Watchdog.exe...
if not exist "build" mkdir "build"

where g++ >nul 2>&1
if %ERRORLEVEL% equ 0 (
    g++ -std=c++17 -O2 -Wall -I"src" src\Watchdog.cpp -o build\Watchdog.exe
    goto :check_watch_compiled
)

where clang++ >nul 2>&1
if %ERRORLEVEL% equ 0 (
    clang++ -std=c++17 -O2 -Wall -I"src" src\Watchdog.cpp -o build\Watchdog.exe
    goto :check_watch_compiled
)

where cl >nul 2>&1
if %ERRORLEVEL% equ 0 (
    cl /std:c++17 /O2 /W3 /EHsc /I"src" src\Watchdog.cpp /Fe:build\Watchdog.exe /Fo:build\
    goto :check_watch_compiled
)

:check_watch_compiled
if %ERRORLEVEL% neq 0 (
    echo [!] FAILED to compile Watchdog.exe
    exit /b 1
)

:run_watchdog
echo [*] Starting BitDB Native C++ Auto-Sync Watchdog on ./ingestor...
build\Watchdog.exe %2 %3 %4 %5
exit /b %ERRORLEVEL%

:start_build

echo.
echo ==================================================
echo   BitDB Prototype-4 - Unified Build System
echo ==================================================
echo.

REM Ensure required directories exist
if not exist "build" mkdir "build"
if not exist "ingestor" mkdir "ingestor"
if not exist "DataStorage" mkdir "DataStorage"

REM Auto-detect Python include and libs using python generator
python -c "import sys, sysconfig, os; inc = os.path.join(sys.base_prefix, 'include'); inc = inc if os.path.isfile(os.path.join(inc, 'Python.h')) else sysconfig.get_path('include'); lib = os.path.join(sys.base_prefix, 'libs') if os.path.isdir(os.path.join(sys.base_prefix, 'libs')) else os.path.join(sys.prefix, 'libs'); name = f'python{sys.version_info[0]}{sys.version_info[1]}'; print(f'set \"PY_INC={inc}\"\nset \"PY_LIB={lib}\"\nset \"PY_NAME={name}\"')" > "%TEMP%\_bitdb_py.bat"
call "%TEMP%\_bitdb_py.bat"
if exist "%TEMP%\_bitdb_py.bat" del "%TEMP%\_bitdb_py.bat"

if not defined PY_INC goto :no_python
goto :has_python

:no_python
echo [!] ERROR: Python not found in PATH. Please install Python 3.
exit /b 1

:has_python
echo   Python Include: %PY_INC%
echo   Python Lib:     %PY_LIB%
echo   Python Name:    %PY_NAME%

REM Check build mode: portable vs avx2 (default)
set BUILD_MODE=AVX2
if /I "%~1"=="portable" set BUILD_MODE=PORTABLE
if /I "%~1"=="scalar"   set BUILD_MODE=PORTABLE
if /I "%~1"=="--portable" set BUILD_MODE=PORTABLE

REM Compiler detection
set CXX=
set COMPILER_FAMILY=

where g++ >nul 2>&1
if %ERRORLEVEL% equ 0 goto :found_gcc

where clang++ >nul 2>&1
if %ERRORLEVEL% equ 0 goto :found_clang

where cl >nul 2>&1
if %ERRORLEVEL% equ 0 goto :found_msvc

goto :no_compiler

:found_gcc
set CXX=g++
set COMPILER_FAMILY=GCC
goto :compiler_detected

:found_clang
set CXX=clang++
set COMPILER_FAMILY=CLANG
goto :compiler_detected

:found_msvc
set CXX=cl
set COMPILER_FAMILY=MSVC
goto :compiler_detected

:no_compiler
echo [!] ERROR: No suitable C++ compiler found (g++, clang++, or cl).
exit /b 1

:compiler_detected
echo   Compiler:       %CXX% (%COMPILER_FAMILY%)
echo   Build Mode:     %BUILD_MODE%
echo(

REM Configure compiler-specific flag sets
if "%COMPILER_FAMILY%"=="MSVC" goto :setup_msvc_flags
if "%BUILD_MODE%"=="PORTABLE" goto :setup_gcc_portable_flags

:setup_gcc_avx2_flags
set FLAGS_UTIL=-std=c++17 -O2 -Wall -I"src"
set FLAGS_CORE=-std=c++17 -O2 -Wall -mavx2 -mpopcnt -I"src" -I"%PY_INC%"
set FLAGS_TEST=-std=c++17 -O2 -Wall -mavx2 -mpopcnt -I"src"
set LIBS=-L"%PY_LIB%" -l%PY_NAME%
goto :flags_configured

:setup_gcc_portable_flags
set FLAGS_UTIL=-std=c++17 -O2 -Wall -I"src"
set FLAGS_CORE=-std=c++17 -O2 -Wall -I"src" -I"%PY_INC%"
set FLAGS_TEST=-std=c++17 -O2 -Wall -I"src"
set LIBS=-L"%PY_LIB%" -l%PY_NAME%
goto :flags_configured

:setup_msvc_flags
set FLAGS_UTIL=/std:c++17 /O2 /W3 /EHsc /I"src"
if "%BUILD_MODE%"=="PORTABLE" goto :setup_msvc_portable_flags

set FLAGS_CORE=/std:c++17 /O2 /W3 /EHsc /MD /arch:AVX2 /I"src" /I"%PY_INC%"
set FLAGS_TEST=/std:c++17 /O2 /W3 /EHsc /arch:AVX2 /I"src"
set LIBS=/link /LIBPATH:"%PY_LIB%" "%PY_NAME%.lib"
goto :flags_configured

:setup_msvc_portable_flags
set FLAGS_CORE=/std:c++17 /O2 /W3 /EHsc /MD /I"src" /I"%PY_INC%"
set FLAGS_TEST=/std:c++17 /O2 /W3 /EHsc /I"src"
set LIBS=/link /LIBPATH:"%PY_LIB%" "%PY_NAME%.lib"
goto :flags_configured

:flags_configured

REM Terminate any lingering instances holding file locks on binaries
taskkill /F /IM BitDBSearch.exe >nul 2>&1
taskkill /F /IM Build.exe >nul 2>&1
taskkill /F /IM print_catalog.exe >nul 2>&1
taskkill /F /IM print_segment_dir.exe >nul 2>&1
taskkill /F /IM test_suite.exe >nul 2>&1
taskkill /F /IM Watchdog.exe >nul 2>&1
taskkill /F /IM HaltonProbes.exe >nul 2>&1

if "%COMPILER_FAMILY%"=="MSVC" goto :build_msvc
goto :build_gcc_clang

:build_msvc
echo [1/7] Compiling Build.exe...
%CXX% %FLAGS_CORE% src\Build.cpp src\embed.cpp /Fe:build\Build.exe /Fo:build\ %LIBS%
if %ERRORLEVEL% neq 0 (
    echo [!] FAILED to compile Build.exe
    exit /b 1
)

echo [2/7] Compiling BitDBSearch.exe...
%CXX% %FLAGS_CORE% src\Search.cpp src\embed.cpp /Fe:build\BitDBSearch.exe /Fo:build\ %LIBS%
if %ERRORLEVEL% neq 0 (
    echo [!] FAILED to compile BitDBSearch.exe
    exit /b 1
)

echo [3/7] Compiling Watchdog.exe...
%CXX% %FLAGS_UTIL% src\Watchdog.cpp /Fe:build\Watchdog.exe /Fo:build\
if %ERRORLEVEL% neq 0 (
    echo [!] FAILED to compile Watchdog.exe
    exit /b 1
)

echo [4/7] Compiling HaltonProbes.exe...
%CXX% %FLAGS_UTIL% src\HaltonProbes.cpp /Fe:build\HaltonProbes.exe /Fo:build\
if %ERRORLEVEL% neq 0 (
    echo [!] FAILED to compile HaltonProbes.exe
    exit /b 1
)

echo [5/7] Compiling print_catalog.exe...
%CXX% %FLAGS_UTIL% printers\print_catalog.cpp /Fe:build\print_catalog.exe /Fo:build\
if %ERRORLEVEL% neq 0 (
    echo [!] FAILED to compile print_catalog.exe
    exit /b 1
)

echo [6/7] Compiling print_segment_dir.exe...
%CXX% %FLAGS_UTIL% printers\print_segment_dir.cpp /Fe:build\print_segment_dir.exe /Fo:build\
if %ERRORLEVEL% neq 0 (
    echo [!] FAILED to compile print_segment_dir.exe
    exit /b 1
)

echo [7/7] Compiling test_suite.exe...
%CXX% %FLAGS_TEST% src\test_suite.cpp /Fe:build\test_suite.exe /Fo:build\
if %ERRORLEVEL% neq 0 (
    echo [!] FAILED to compile test_suite.exe
    exit /b 1
)
goto :success

:build_gcc_clang
echo [1/7] Compiling Build.exe...
%CXX% %FLAGS_CORE% src\Build.cpp src\embed.cpp %LIBS% -o build\Build.exe
if %ERRORLEVEL% neq 0 (
    echo [!] FAILED to compile Build.exe
    exit /b 1
)

echo [2/7] Compiling BitDBSearch.exe...
%CXX% %FLAGS_CORE% src\Search.cpp src\embed.cpp %LIBS% -o build\BitDBSearch.exe
if %ERRORLEVEL% neq 0 (
    echo [!] FAILED to compile BitDBSearch.exe
    exit /b 1
)

echo [3/7] Compiling Watchdog.exe...
%CXX% %FLAGS_UTIL% src\Watchdog.cpp -o build\Watchdog.exe
if %ERRORLEVEL% neq 0 (
    echo [!] FAILED to compile Watchdog.exe
    exit /b 1
)

echo [4/7] Compiling HaltonProbes.exe...
%CXX% %FLAGS_UTIL% src\HaltonProbes.cpp -o build\HaltonProbes.exe
if %ERRORLEVEL% neq 0 (
    echo [!] FAILED to compile HaltonProbes.exe
    exit /b 1
)

echo [5/7] Compiling print_catalog.exe...
%CXX% %FLAGS_UTIL% printers\print_catalog.cpp -o build\print_catalog.exe
if %ERRORLEVEL% neq 0 (
    echo [!] FAILED to compile print_catalog.exe
    exit /b 1
)

echo [6/7] Compiling print_segment_dir.exe...
%CXX% %FLAGS_UTIL% printers\print_segment_dir.cpp -o build\print_segment_dir.exe
if %ERRORLEVEL% neq 0 (
    echo [!] FAILED to compile print_segment_dir.exe
    exit /b 1
)

echo [7/7] Compiling test_suite.exe...
%CXX% %FLAGS_TEST% src\test_suite.cpp -o build\test_suite.exe
if %ERRORLEVEL% neq 0 (
    echo [!] FAILED to compile test_suite.exe
    exit /b 1
)
goto :success

:success
echo.
echo ==================================================
echo   BUILD SUCCESSFUL (%COMPILER_FAMILY% - %BUILD_MODE%)
echo ==================================================
echo.
echo   Binaries placed in:  build\
echo.
echo   USAGE:
echo     1. Drop PDFs:   ingestor\
echo     2. Ingest:      build\Build.exe
echo     3. Query:       build\BitDBSearch.exe "your search query" [N] [probes]
echo        List Files:  build\BitDBSearch.exe --list
echo        Interactive: build\BitDBSearch.exe --interactive --probes 4
echo     4. Rebuild:     build\Build.exe --rebuild
echo     5. Watchdog:    build\Build.exe --watch (or build.bat watch)
echo     6. Catalog:     build\print_catalog.exe
echo     7. Segments:    build\print_segment_dir.exe
echo     8. Test:        build\test_suite.exe
echo ==================================================
echo.
exit /b 0
