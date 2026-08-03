@echo off
echo Building Node.exe...
C:\msys64\ucrt64\bin\g++.exe -std=c++17 -O2 -I"C:\Users\srish\AppData\Local\Programs\Python\Python311\Include" -L"C:\Users\srish\AppData\Local\Programs\Python\Python311\libs" src\Node.cpp src\embed.cpp -lpython311 -o build\Node.exe

echo Building printer utilities...
C:\msys64\ucrt64\bin\g++.exe -std=c++17 -O2 printers\print_nodes.cpp -o build\print_nodes.exe
C:\msys64\ucrt64\bin\g++.exe -std=c++17 -O2 printers\print_children.cpp -o build\print_children.exe
C:\msys64\ucrt64\bin\g++.exe -std=c++17 -O2 printers\print_centroids.cpp -o build\print_centroids.exe
C:\msys64\ucrt64\bin\g++.exe -std=c++17 -O2 printers\print_embeddings.cpp -o build\print_embeddings.exe
C:\msys64\ucrt64\bin\g++.exe -std=c++17 -O2 printers\print_leaf_centroids.cpp -o build\print_leaf_centroids.exe

echo Building Search.exe...
C:\msys64\ucrt64\bin\g++.exe -std=c++17 -O2 -I"C:\Users\srish\AppData\Local\Programs\Python\Python311\Include" -L"C:\Users\srish\AppData\Local\Programs\Python\Python311\libs" src\Search.cpp src\embed.cpp -lpython311 -o build\Search.exe

echo Building test_search.exe...
C:\msys64\ucrt64\bin\g++.exe -std=c++17 -O2 -I"C:\Users\srish\AppData\Local\Programs\Python\Python311\Include" -L"C:\Users\srish\AppData\Local\Programs\Python\Python311\libs" src\test_search.cpp src\embed.cpp -lpython311 -o build\test_search.exe

echo Building ingest.exe...
C:\msys64\ucrt64\bin\g++.exe -std=c++17 -O2 -I"C:\Users\srish\AppData\Local\Programs\Python\Python311\Include" -L"C:\Users\srish\AppData\Local\Programs\Python\Python311\libs" src\ingest.cpp src\embed.cpp -lpython311 -o build\ingest.exe

if %ERRORLEVEL% equ 0 (
    echo Build successful.
    echo All programs compiled into build/
) else (
    echo Build failed.
)
