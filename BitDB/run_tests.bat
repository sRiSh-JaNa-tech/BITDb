@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo.
echo ==================================================
echo   BitDB Automated Test ^& Invariant Verification
echo ==================================================
echo.

:: Ensure build directory exists
if not exist "build" (
    echo [*] build directory missing. Running build.bat first...
    call build.bat
    if errorlevel 1 (
        echo [-] Build failed. Cannot run tests.
        exit /b 1
    )
)

:: Compile test suite if missing
if not exist "build\test_suite.exe" (
    echo [*] Compiling test_suite.exe...
    g++ -std=c++17 -O2 -mavx2 -mpopcnt -I"src" src\test_suite.cpp -o build\test_suite.exe
    if errorlevel 1 (
        echo [-] Failed to compile test_suite.exe
        exit /b 1
    )
)

:: Compile HaltonProbes if missing
if not exist "build\HaltonProbes.exe" (
    echo [*] Compiling HaltonProbes.exe...
    g++ -std=c++17 -O2 -I"src" src\HaltonProbes.cpp -o build\HaltonProbes.exe
    if errorlevel 1 (
        echo [-] Failed to compile HaltonProbes.exe
        exit /b 1
    )
)

set TESTS_FAILED=0

echo --------------------------------------------------
echo  [STEP 1/5] Invariant ^& Unit Test Suite (11 tests)
echo --------------------------------------------------
build\test_suite.exe
if errorlevel 1 (
    echo [FAIL] test_suite.exe failed.
    set /a TESTS_FAILED+=1
) else (
    echo [OK] All unit and mathematical invariant tests passed.
)
echo.

echo --------------------------------------------------
echo  [STEP 2/5] Native Halton Probes Generator
echo --------------------------------------------------
build\HaltonProbes.exe
if errorlevel 1 (
    echo [FAIL] HaltonProbes.exe failed.
    set /a TESTS_FAILED+=1
) else (
    echo [OK] Halton probe generation verified.
)
echo.

echo --------------------------------------------------
echo  [STEP 3/5] Document Catalog Integrity Audit
echo --------------------------------------------------
if exist "build\print_catalog.exe" (
    build\print_catalog.exe
    if errorlevel 1 (
        echo [FAIL] print_catalog.exe failed.
        set /a TESTS_FAILED+=1
    ) else (
        echo [OK] Catalog integrity verified.
    )
) else (
    echo [*] print_catalog.exe not built yet. Run build.bat.
)
echo.

echo --------------------------------------------------
echo  [STEP 4/5] 256-Segment Balance ^& Skew Audit
echo --------------------------------------------------
if exist "build\print_segment_dir.exe" (
    build\print_segment_dir.exe
    if errorlevel 1 (
        echo [FAIL] print_segment_dir.exe failed.
        set /a TESTS_FAILED+=1
    ) else (
        echo [OK] Segment directory structure verified.
    )
) else (
    echo [*] print_segment_dir.exe not built yet. Run build.bat.
)
echo.

echo --------------------------------------------------
echo  [STEP 5/5] Live End-to-End Query Verification
echo --------------------------------------------------
if exist "build\BitDBSearch.exe" (
    build\BitDBSearch.exe "vector index on SSD" 3 4
    if errorlevel 1 (
        echo [FAIL] BitDBSearch.exe query failed.
        set /a TESTS_FAILED+=1
    ) else (
        echo [OK] Live query and passage retrieval verified.
    )
) else (
    echo [*] BitDBSearch.exe not built yet. Run build.bat.
)
echo.

echo ==================================================
if %TESTS_FAILED% equ 0 (
    echo   ALL TESTS PASSED SUCCESSFULLY - 0 Failures
) else (
    echo   TEST RUN FINISHED WITH %TESTS_FAILED% FAILURE(S).
)
echo ==================================================
echo.

exit /b %TESTS_FAILED%
