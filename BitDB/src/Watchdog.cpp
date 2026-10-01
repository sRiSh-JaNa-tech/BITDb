// ════════════════════════════════════════════════════════════════════════
// Watchdog.cpp — Native BitDB Ingestor File Watcher & Auto-Sync Engine
// ════════════════════════════════════════════════════════════════════════

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <filesystem>
#include <chrono>
#include <thread>
#include <cstdlib>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include "PathConfig.h"

using namespace std;
namespace fs = std::filesystem;

struct FileSnapshot {
    fs::file_time_type last_write;
    uintmax_t file_size;
};

static map<string, FileSnapshot> get_ingestor_snapshot(const fs::path& dir) {
    map<string, FileSnapshot> snapshot;
    if (!fs::exists(dir) || !fs::is_directory(dir)) {
        return snapshot;
    }

    std::error_code ec;
    for (const auto& entry : fs::recursive_directory_iterator(dir, fs::directory_options::skip_permission_denied, ec)) {
        if (ec) continue;
        std::error_code fileEc;
        if (entry.is_regular_file(fileEc)) {
            string ext = entry.path().extension().string();
            for (auto& c : ext) c = static_cast<char>(tolower(c));
            if (ext == ".pdf") {
                string rel = fs::relative(entry.path(), dir).generic_string();
                snapshot[rel] = {entry.last_write_time(fileEc), entry.file_size(fileEc)};
            }
        }
    }
    return snapshot;
}

static fs::path find_build_binary() {
    fs::path root = PathConfig::getProjectRoot();
    vector<fs::path> candidates = {
        root / "build" / "Build.exe",
        root / "build" / "Build",
        root / "build" / "Release" / "Build.exe",
        root / "build_cmake" / "Build.exe",
        root / "build_cmake" / "Release" / "Build.exe",
        root / "build_cmake" / "Build",
    };

    for (const auto& p : candidates) {
        if (fs::exists(p) && fs::is_regular_file(p)) {
            return p;
        }
    }
    return root / "build" / "Build.exe";
}

int main(int argc, char* argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    ios::sync_with_stdio(true);

    double interval_sec = 2.5;
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if ((arg == "--interval" || arg == "-i") && i + 1 < argc) {
            interval_sec = stod(argv[++i]);
        }
    }

    fs::path ingestorDir = PathConfig::getIngestorDir();
    fs::path projectRoot = PathConfig::getProjectRoot();
    fs::path buildExe    = find_build_binary();

    std::error_code ec;
    fs::create_directories(ingestorDir, ec);

    cout << "==================================================\n";
    cout << "  BitDB Native C++ Auto-Sync Watchdog\n";
    cout << "==================================================\n";
    cout << "[*] Project root:      " << projectRoot.string() << "\n";
    cout << "[*] Monitoring dir:    " << ingestorDir.string() << "\n";
    cout << "[*] Engine executable: " << buildExe.string() << "\n";
    cout << "[*] Polling interval:  " << interval_sec << "s\n";
    cout << "[*] Ready. Listening for file additions, edits, or deletions...\n";
    cout << "    (Press Ctrl+C to stop)\n\n";

    auto current_snapshot = get_ingestor_snapshot(ingestorDir);

    while (true) {
        this_thread::sleep_for(chrono::milliseconds(static_cast<long long>(interval_sec * 1000.0)));

        auto new_snapshot = get_ingestor_snapshot(ingestorDir);

        vector<string> added;
        vector<string> deleted;
        vector<string> modified;

        for (const auto& [path, snap] : new_snapshot) {
            auto it = current_snapshot.find(path);
            if (it == current_snapshot.end()) {
                added.push_back(path);
            } else if (it->second.last_write != snap.last_write || it->second.file_size != snap.file_size) {
                modified.push_back(path);
            }
        }

        for (const auto& [path, _] : current_snapshot) {
            if (new_snapshot.find(path) == new_snapshot.end()) {
                deleted.push_back(path);
            }
        }

        if (!added.empty() || !deleted.empty() || !modified.empty()) {
            if (!added.empty()) {
                cout << "[Watchdog] Detected " << added.size() << " new file(s):\n";
                for (const auto& f : added) cout << "   + " << f << "\n";
            }
            if (!deleted.empty()) {
                cout << "[Watchdog] Detected " << deleted.size() << " deleted file(s):\n";
                for (const auto& f : deleted) cout << "   - " << f << "\n";
            }
            if (!modified.empty()) {
                cout << "[Watchdog] Detected " << modified.size() << " modified file(s):\n";
                for (const auto& f : modified) cout << "   ~ " << f << "\n";
            }

            cout << "[Watchdog] Synchronizing database...\n";
            this_thread::sleep_for(chrono::milliseconds(1000)); // Allow OS write buffers to settle

            if (!fs::exists(buildExe)) {
                cerr << "[Watchdog] ERROR: Engine executable not found at '" << buildExe.string() << "'\n";
            } else {
                string cmd = "\"" + buildExe.string() + "\"";
                int ret = system(cmd.c_str());
                if (ret == 0) {
                    cout << "[Watchdog] Database sync completed successfully.\n\n";
                    current_snapshot = new_snapshot;
                } else {
                    cerr << "[Watchdog] Engine exited with code " << ret << "\n\n";
                }
            }
        }
    }

    return 0;
}
