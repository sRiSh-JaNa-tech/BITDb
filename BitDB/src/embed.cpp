#include <Python.h>
#include "embed.h"
#include "PathConfig.h"
#include <iostream>
#include <sstream>
#include <fstream>

// ──────────────────────────────────────────────────────────────────
// Module-level state (lives for the lifetime of the program)
// ──────────────────────────────────────────────────────────────────
static PyObject* g_pModule              = nullptr;
static PyObject* g_pChunkTextFunc       = nullptr;   // chunk_text_with_offsets
static PyObject* g_pEmbedFunc           = nullptr;
static PyObject* g_pPdfToTextPagesFunc  = nullptr;   // pdf_to_text_pages

// ──────────────────────────────────────────────────────────────────
// init_python / finalize_python
// ──────────────────────────────────────────────────────────────────

void init_python() {
    Py_Initialize();

    std::string scriptsDir = PathConfig::getScriptsDir().generic_string();
    fs::path venvPath      = PathConfig::getVenvSitePackagesDir();

    std::ostringstream ss;
    ss << "import sys, site, os\n"
       << "sys.path.insert(0, '" << scriptsDir << "')\n";

    if (fs::exists(venvPath)) {
        ss << "sys.path.insert(0, '" << venvPath.generic_string() << "')\n";
    }

    PyRun_SimpleString(ss.str().c_str());

    PyObject* pName = PyUnicode_DecodeFSDefault("vendor");
    g_pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (!g_pModule) {
        PyErr_Print();
        std::cerr << "[embed.cpp] FATAL: Failed to import vendor module" << std::endl;
        return;
    }

    g_pEmbedFunc = PyObject_GetAttrString(g_pModule, "embed_chunks");
    if (!g_pEmbedFunc || !PyCallable_Check(g_pEmbedFunc)) {
        PyErr_Print();
        std::cerr << "[embed.cpp] FATAL: Cannot find vendor.embed_chunks()" << std::endl;
    }

    g_pChunkTextFunc = PyObject_GetAttrString(g_pModule, "chunk_text_with_offsets");
    if (!g_pChunkTextFunc || !PyCallable_Check(g_pChunkTextFunc)) {
        PyErr_Print();
        std::cerr << "[embed.cpp] FATAL: Cannot find vendor.chunk_text_with_offsets()" << std::endl;
    }

    g_pPdfToTextPagesFunc = PyObject_GetAttrString(g_pModule, "pdf_to_text_pages");
    if (!g_pPdfToTextPagesFunc || !PyCallable_Check(g_pPdfToTextPagesFunc)) {
        PyErr_Print();
        std::cerr << "[embed.cpp] FATAL: Cannot find vendor.pdf_to_text_pages()" << std::endl;
    }

    std::cout << "[embed.cpp] Python initialized, vendor module loaded." << std::endl;
}

void finalize_python() {
    Py_XDECREF(g_pPdfToTextPagesFunc);
    Py_XDECREF(g_pChunkTextFunc);
    Py_XDECREF(g_pEmbedFunc);
    Py_XDECREF(g_pModule);
    g_pPdfToTextPagesFunc  = nullptr;
    g_pChunkTextFunc       = nullptr;
    g_pEmbedFunc           = nullptr;
    g_pModule              = nullptr;
    Py_Finalize();
    std::cout << "[embed.cpp] Python finalized." << std::endl;
}

// ──────────────────────────────────────────────────────────────────
// chunk_file (Native C++ text reader)
// ──────────────────────────────────────────────────────────────────

std::vector<std::string> chunk_file(const std::string& filepath) {
    std::ifstream f(filepath, std::ios::in | std::ios::binary);
    if (!f.is_open()) return {};
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto infos = chunk_text_with_offsets(text);
    std::vector<std::string> result;
    result.reserve(infos.size());
    for (const auto& ci : infos) {
        result.push_back(ci.text);
    }
    return result;
}

// ──────────────────────────────────────────────────────────────────
// chunk_file_with_offsets (Native C++ text reader)
// ──────────────────────────────────────────────────────────────────

std::vector<ChunkInfo> chunk_file_with_offsets(const std::string& filepath) {
    std::ifstream f(filepath, std::ios::in | std::ios::binary);
    if (!f.is_open()) return {};
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return chunk_text_with_offsets(text);
}

// ──────────────────────────────────────────────────────────────────
// chunk_text_with_offsets
// Takes a raw text string (not a file path)
// ──────────────────────────────────────────────────────────────────

std::vector<ChunkInfo> chunk_text_with_offsets(const std::string& text) {
    std::vector<ChunkInfo> result;
    if (!g_pChunkTextFunc) return result;

    PyObject* pArgs = PyTuple_New(1);
    PyTuple_SetItem(pArgs, 0, PyUnicode_FromString(text.c_str()));
    PyObject* pReturn = PyObject_CallObject(g_pChunkTextFunc, pArgs);
    Py_DECREF(pArgs);

    if (!pReturn) { PyErr_Print(); return result; }
    if (PyList_Check(pReturn)) {
        Py_ssize_t sz = PyList_Size(pReturn);
        for (Py_ssize_t i = 0; i < sz; ++i) {
            PyObject* pTup = PyList_GetItem(pReturn, i);
            if (PyTuple_Check(pTup) && PyTuple_Size(pTup) == 3) {
                const char* s  = PyUnicode_AsUTF8(PyTuple_GetItem(pTup, 0));
                uint64_t off   = PyLong_AsUnsignedLongLong(PyTuple_GetItem(pTup, 1));
                uint32_t len   = static_cast<uint32_t>(PyLong_AsUnsignedLong(PyTuple_GetItem(pTup, 2)));
                if (s) result.push_back({std::string(s), off, len});
            }
        }
    }
    Py_DECREF(pReturn);
    return result;
}

// ──────────────────────────────────────────────────────────────────
// embed_chunks
// ──────────────────────────────────────────────────────────────────

std::vector<std::vector<int8_t>> embed_chunks(const std::vector<std::string>& sentences) {
    std::vector<std::vector<int8_t>> result;
    if (!g_pEmbedFunc || sentences.empty()) return result;

    PyObject* pList = PyList_New(sentences.size());
    for (size_t i = 0; i < sentences.size(); ++i) {
        PyList_SetItem(pList, i, PyUnicode_FromString(sentences[i].c_str()));
    }
    PyObject* pArgs = PyTuple_New(1);
    PyTuple_SetItem(pArgs, 0, pList);
    PyObject* pReturn = PyObject_CallObject(g_pEmbedFunc, pArgs);
    Py_DECREF(pArgs);

    if (!pReturn) { PyErr_Print(); return result; }
    if (PyList_Check(pReturn)) {
        Py_ssize_t n = PyList_Size(pReturn);
        for (Py_ssize_t i = 0; i < n; ++i) {
            PyObject* pEmb = PyList_GetItem(pReturn, i);
            std::vector<int8_t> emb;
            if (PyList_Check(pEmb)) {
                Py_ssize_t dim = PyList_Size(pEmb);
                emb.reserve(dim);
                for (Py_ssize_t j = 0; j < dim; ++j) {
                    emb.push_back(static_cast<int8_t>(PyLong_AsLong(PyList_GetItem(pEmb, j))));
                }
            }
            result.push_back(std::move(emb));
        }
    }
    Py_DECREF(pReturn);
    return result;
}

// ──────────────────────────────────────────────────────────────────
// pdf_to_text_pages
// ──────────────────────────────────────────────────────────────────

std::vector<PageText> pdf_to_text_pages(const std::string& filepath) {
    std::vector<PageText> result;
    if (!g_pPdfToTextPagesFunc) return result;

    PyObject* pArgs = PyTuple_New(1);
    PyTuple_SetItem(pArgs, 0, PyUnicode_FromString(filepath.c_str()));
    PyObject* pReturn = PyObject_CallObject(g_pPdfToTextPagesFunc, pArgs);
    Py_DECREF(pArgs);

    if (!pReturn) { PyErr_Print(); return result; }
    // pReturn: list of (page_num: int, text: str) tuples
    if (PyList_Check(pReturn)) {
        Py_ssize_t n = PyList_Size(pReturn);
        for (Py_ssize_t i = 0; i < n; ++i) {
            PyObject* pTup = PyList_GetItem(pReturn, i);
            if (PyTuple_Check(pTup) && PyTuple_Size(pTup) == 2) {
                int pageNum      = static_cast<int>(PyLong_AsLong(PyTuple_GetItem(pTup, 0)));
                const char* text = PyUnicode_AsUTF8(PyTuple_GetItem(pTup, 1));
                if (text) result.push_back({pageNum, std::string(text)});
            }
        }
    }
    Py_DECREF(pReturn);
    return result;
}
