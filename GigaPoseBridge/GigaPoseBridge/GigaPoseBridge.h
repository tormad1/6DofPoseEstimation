#pragma once
#include <cstdint>

#ifdef _WIN32
#define GIGAPOSE_EXPORT __declspec(dllexport)
#else
#define GIGAPOSE_EXPORT
#endif

extern "C" {

    // Initialises the embedded Python interpreter
    // python_home: path to your Python installation e.g. C:\Users\...\miniconda3
    // Returns 1 on success, 0 on failure
    GIGAPOSE_EXPORT int __cdecl InitPython(const char* python_home);

    // Shuts down the embedded Python interpreter
    // Call this when Unity unloads the plugin
    GIGAPOSE_EXPORT void __cdecl ShutdownPython();

    // passes image_path to Python, which opens it with Pillow
    // and writes the result into out_buf as a null-terminated string
    // out_buf_len: size of the buffer you're passing in
    // Returns 1 on success, 0 on failure
    GIGAPOSE_EXPORT int __cdecl OpenImageTest(const char* image_path, char* out_buf, int out_buf_len);

}