#include <iostream>
#include <fstream>
#include <cstdint>
#include <cstring>
#include "../src/PathConfig.h"

using namespace std;

#pragma pack(push, 1)
struct CatalogHeader {
    uint32_t num_docs;
    uint32_t active_docs;
};

struct DocEntry {
    uint32_t doc_id;
    uint32_t page_count;
    uint64_t chunk_store_start;
    uint32_t chunk_count;
    uint32_t is_deleted;
    char     filename[232];
};
#pragma pack(pop)

int main() {
    const string path = PathConfig::getDocCatalogFile().string();
    ifstream f(path, ios::binary);
    if (!f) {
        cerr << "Cannot open " << path << "\n";
        return 1;
    }

    CatalogHeader hdr;
    f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    cout << "doc_catalog.bin — " << hdr.num_docs << " documents (" << hdr.active_docs << " active)\n\n";
    cout << " ID  | Pages | Chunks | Status    | Filename\n";
    cout << "-----|-------|--------|-----------|" << string(50,'-') << "\n";

    for (uint32_t i = 0; i < hdr.num_docs; ++i) {
        DocEntry de;
        f.read(reinterpret_cast<char*>(&de), sizeof(de));
        printf("%4u | %5u | %6u | %-9s | %s\n",
               de.doc_id, de.page_count, de.chunk_count,
               (de.is_deleted ? "DELETED" : "ACTIVE"),
               de.filename);
    }
    return 0;
}
