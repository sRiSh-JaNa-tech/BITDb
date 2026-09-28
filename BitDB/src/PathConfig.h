#pragma once
// ════════════════════════════════════════════════════════════════════════
// PathConfig.h — Dynamic cross-platform path resolver for BitDB
//
// Automatically discovers the project root and manages all directory paths
// (DataStorage, ingestor, scripts, models, python venv) without any hardcoded
// system or user paths.
// ════════════════════════════════════════════════════════════════════════

#include <filesystem>
#include <string>
#include <cstdlib>
#include <iostream>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#elif defined(__linux__)
#include <unistd.h>
#include <limits.h>
#endif

namespace fs = std::filesystem;

class PathConfig {
public:
    // Returns the root directory of the BitDB project
    static fs::path getProjectRoot() {
        static fs::path root = discoverProjectRoot();
        return root;
    }

    static fs::path getDataStorageDir() {
        return getProjectRoot() / "DataStorage";
    }

    static fs::path getIngestorDir() {
        return getProjectRoot() / "ingestor";
    }

    static fs::path getScriptsDir() {
        return getProjectRoot() / "scripts";
    }

    static fs::path getModelsDir() {
        return getProjectRoot() / "models";
    }

    // Binary index files
    static fs::path getSegmentDirFile() {
        return getDataStorageDir() / "segment_dir.bin";
    }

    static fs::path getSegmentExtentsFile() {
        return getDataStorageDir() / "segment_extents.bin";
    }

    static fs::path getChunkStoreFile() {
        return getDataStorageDir() / "chunk_store.bin";
    }

    static fs::path getPdfTextFile() {
        return getDataStorageDir() / "pdf_text.bin";
    }

    static fs::path getDocCatalogFile() {
        return getDataStorageDir() / "doc_catalog.bin";
    }

    static fs::path getMihTableFile() {
        return getDataStorageDir() / "mih_table.bin";
    }

    // Resolves virtual environment site-packages directory
    static fs::path getVenvSitePackagesDir() {
        fs::path root = getProjectRoot();
        
        // Common virtualenv names
        const std::string venvNames[] = {"bitdb", "venv", ".venv", "env"};
        
        for (const auto& name : venvNames) {
            fs::path candidateWin = root / name / "Lib" / "site-packages";
            if (fs::exists(candidateWin)) return candidateWin;

            // Unix / Linux / macOS style
            fs::path candidateUnixLib = root / name / "lib";
            if (fs::exists(candidateUnixLib)) {
                for (const auto& entry : fs::directory_iterator(candidateUnixLib)) {
                    if (entry.is_directory() && entry.path().filename().string().rfind("python", 0) == 0) {
                        fs::path sp = entry.path() / "site-packages";
                        if (fs::exists(sp)) return sp;
                    }
                }
            }
        }
        return root / "bitdb" / "Lib" / "site-packages";
    }

    // Ensures necessary data directories exist
    static void ensureDirectories() {
        fs::create_directories(getDataStorageDir());
        fs::create_directories(getIngestorDir());
    }

private:
    static fs::path getExecutableDirectory() {
#if defined(_WIN32)
        wchar_t path[MAX_PATH];
        DWORD length = GetModuleFileNameW(NULL, path, MAX_PATH);
        if (length > 0) {
            return fs::path(path).parent_path();
        }
#elif defined(__APPLE__)
        char path[1024];
        uint32_t size = sizeof(path);
        if (_NSGetExecutablePath(path, &size) == 0) {
            return fs::weakly_canonical(fs::path(path)).parent_path();
        }
#elif defined(__linux__)
        char path[PATH_MAX];
        ssize_t length = readlink("/proc/self/exe", path, sizeof(path) - 1);
        if (length != -1) {
            path[length] = '\0';
            return fs::path(path).parent_path();
        }
#endif
        return fs::current_path();
    }

    static fs::path discoverProjectRoot() {
        // 1. Check environment variable override
        const char* envRoot = std::getenv("BITDB_ROOT");
        if (envRoot && *envRoot) {
            fs::path p(envRoot);
            if (fs::exists(p)) return fs::weakly_canonical(p);
        }

        // 2. Search upward starting from the executable directory and current directory
        fs::path searchStarts[] = { getExecutableDirectory(), fs::current_path() };

        for (const auto& start : searchStarts) {
            fs::path cur = fs::weakly_canonical(start);
            while (!cur.empty()) {
                // Look for project root markers
                if (fs::exists(cur / "src" / "PathConfig.h") ||
                    fs::exists(cur / "scripts" / "vendor.py") ||
                    fs::exists(cur / "CMakeLists.txt") ||
                    (fs::exists(cur / "src") && fs::exists(cur / "DataStorage"))) {
                    return cur;
                }
                if (cur == cur.parent_path()) break; // reached filesystem root
                cur = cur.parent_path();
            }
        }

        // 3. Fallback to current working directory
        return fs::current_path();
    }
};
