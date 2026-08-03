#include <Python.h>
#include "embed.h"
#include <iostream>

// ──────────────────────────────────────────────
// Module-level state (lives for the lifetime of the program)
// ──────────────────────────────────────────────
static PyObject* g_pModule = nullptr;
static PyObject* g_pChunkFunc = nullptr;
static PyObject* g_pChunkOffsetsFunc = nullptr;
static PyObject* g_pEmbedFunc = nullptr;

// ──────────────────────────────────────────────
// init_python / finalize_python
// ──────────────────────────────────────────────

void init_python() {
    Py_Initialize();

    // Add the scripts directory and the venv site-packages to Python path
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.insert(0, 'C:/Users/srish/Desktop/BitDB/Prototype-1/scripts')");
    PyRun_SimpleString("sys.path.insert(0, 'C:/Users/srish/Desktop/BitDB/bitdb/Lib/site-packages')");

    // Import the vendor module
    PyObject* pName = PyUnicode_DecodeFSDefault("vendor");
    g_pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (!g_pModule) {
        PyErr_Print();
        std::cerr << "[embed.cpp] FATAL: Failed to import vendor module" << std::endl;
        return;
    }

    // Cache function references
    g_pChunkFunc = PyObject_GetAttrString(g_pModule, "chunk_file");
    if (!g_pChunkFunc || !PyCallable_Check(g_pChunkFunc)) {
        PyErr_Print();
        std::cerr << "[embed.cpp] FATAL: Cannot find vendor.chunk_file()" << std::endl;
    }

    g_pChunkOffsetsFunc = PyObject_GetAttrString(g_pModule, "chunk_file_with_offsets");
    if (!g_pChunkOffsetsFunc || !PyCallable_Check(g_pChunkOffsetsFunc)) {
        PyErr_Print();
        std::cerr << "[embed.cpp] FATAL: Cannot find vendor.chunk_file_with_offsets()" << std::endl;
    }

    g_pEmbedFunc = PyObject_GetAttrString(g_pModule, "embed_chunks");
    if (!g_pEmbedFunc || !PyCallable_Check(g_pEmbedFunc)) {
        PyErr_Print();
        std::cerr << "[embed.cpp] FATAL: Cannot find vendor.embed_chunks()" << std::endl;
    }

    std::cout << "[embed.cpp] Python initialized, vendor module loaded." << std::endl;
}

void finalize_python() {
    Py_XDECREF(g_pEmbedFunc);
    Py_XDECREF(g_pChunkOffsetsFunc);
    Py_XDECREF(g_pChunkFunc);
    Py_XDECREF(g_pModule);
    g_pEmbedFunc = nullptr;
    g_pChunkOffsetsFunc = nullptr;
    g_pChunkFunc = nullptr;
    g_pModule = nullptr;

    Py_Finalize();
    std::cout << "[embed.cpp] Python finalized." << std::endl;
}

// ──────────────────────────────────────────────
// chunk_file — calls vendor.chunk_file(filepath)
// ──────────────────────────────────────────────

std::vector<std::string> chunk_file(const std::string& filepath) {
    std::vector<std::string> result;

    if (!g_pChunkFunc) {
        std::cerr << "[embed.cpp] chunk_file: function not loaded" << std::endl;
        return result;
    }

    // Build arguments: (filepath,)
    PyObject* pArgs = PyTuple_New(1);
    PyObject* pPath = PyUnicode_FromString(filepath.c_str());
    PyTuple_SetItem(pArgs, 0, pPath); // steals reference

    // Call
    PyObject* pReturn = PyObject_CallObject(g_pChunkFunc, pArgs);
    Py_DECREF(pArgs);

    if (!pReturn) {
        PyErr_Print();
        std::cerr << "[embed.cpp] chunk_file call failed for: " << filepath << std::endl;
        return result;
    }

    // pReturn is a Python list of strings
    if (PyList_Check(pReturn)) {
        Py_ssize_t size = PyList_Size(pReturn);
        for (Py_ssize_t i = 0; i < size; ++i) {
            PyObject* pItem = PyList_GetItem(pReturn, i); // borrowed ref
            const char* str = PyUnicode_AsUTF8(pItem);
            if (str) {
                result.push_back(std::string(str));
            }
        }
    }

    Py_DECREF(pReturn);
    return result;
}

// ──────────────────────────────────────────────
// chunk_file_with_offsets — calls vendor.chunk_file_with_offsets(filepath)
// ──────────────────────────────────────────────

std::vector<ChunkInfo> chunk_file_with_offsets(const std::string& filepath) {
    std::vector<ChunkInfo> result;

    if (!g_pChunkOffsetsFunc) {
        std::cerr << "[embed.cpp] chunk_file_with_offsets: function not loaded" << std::endl;
        return result;
    }

    PyObject* pArgs = PyTuple_New(1);
    PyObject* pPath = PyUnicode_FromString(filepath.c_str());
    PyTuple_SetItem(pArgs, 0, pPath); // steals reference

    PyObject* pReturn = PyObject_CallObject(g_pChunkOffsetsFunc, pArgs);
    Py_DECREF(pArgs);

    if (!pReturn) {
        PyErr_Print();
        std::cerr << "[embed.cpp] chunk_file_with_offsets call failed for: " << filepath << std::endl;
        return result;
    }

    // pReturn is a Python list of tuples (str, int, int)
    if (PyList_Check(pReturn)) {
        Py_ssize_t size = PyList_Size(pReturn);
        for (Py_ssize_t i = 0; i < size; ++i) {
            PyObject* pTuple = PyList_GetItem(pReturn, i); // borrowed ref
            if (PyTuple_Check(pTuple) && PyTuple_Size(pTuple) == 3) {
                PyObject* pStr = PyTuple_GetItem(pTuple, 0); // borrowed ref
                PyObject* pOffset = PyTuple_GetItem(pTuple, 1);
                PyObject* pLength = PyTuple_GetItem(pTuple, 2);

                const char* str = PyUnicode_AsUTF8(pStr);
                uint64_t offset = PyLong_AsUnsignedLongLong(pOffset);
                uint32_t length = static_cast<uint32_t>(PyLong_AsUnsignedLong(pLength));

                if (str) {
                    result.push_back({std::string(str), offset, length});
                }
            }
        }
    }

    Py_DECREF(pReturn);
    return result;
}

// ──────────────────────────────────────────────
// embed_chunks — calls vendor.embed_chunks(sentences)
// ──────────────────────────────────────────────

std::vector<std::vector<int8_t>> embed_chunks(const std::vector<std::string>& sentences) {
    std::vector<std::vector<int8_t>> result;

    if (!g_pEmbedFunc) {
        std::cerr << "[embed.cpp] embed_chunks: function not loaded" << std::endl;
        return result;
    }

    if (sentences.empty()) return result;

    // Build a Python list of strings
    PyObject* pList = PyList_New(sentences.size());
    for (size_t i = 0; i < sentences.size(); ++i) {
        PyObject* pStr = PyUnicode_FromString(sentences[i].c_str());
        PyList_SetItem(pList, i, pStr); // steals reference
    }

    // Build arguments: (list,)
    PyObject* pArgs = PyTuple_New(1);
    PyTuple_SetItem(pArgs, 0, pList); // steals reference

    // Call
    PyObject* pReturn = PyObject_CallObject(g_pEmbedFunc, pArgs);
    Py_DECREF(pArgs);

    if (!pReturn) {
        PyErr_Print();
        std::cerr << "[embed.cpp] embed_chunks call failed" << std::endl;
        return result;
    }

    // pReturn is a Python list of lists of ints
    if (PyList_Check(pReturn)) {
        Py_ssize_t numChunks = PyList_Size(pReturn);
        for (Py_ssize_t i = 0; i < numChunks; ++i) {
            PyObject* pEmbedding = PyList_GetItem(pReturn, i); // borrowed ref
            std::vector<int8_t> emb;

            if (PyList_Check(pEmbedding)) {
                Py_ssize_t dim = PyList_Size(pEmbedding);
                emb.reserve(dim);
                for (Py_ssize_t j = 0; j < dim; ++j) {
                    PyObject* pVal = PyList_GetItem(pEmbedding, j); // borrowed ref
                    int8_t val = static_cast<int8_t>(PyLong_AsLong(pVal));
                    emb.push_back(val);
                }
            }

            result.push_back(std::move(emb));
        }
    }

    Py_DECREF(pReturn);
    return result;
}
