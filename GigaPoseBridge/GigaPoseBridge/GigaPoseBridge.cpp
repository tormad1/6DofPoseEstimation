#include "pch.h"
#include "GigaPoseBridge.h"
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string>

extern "C" {

    int __cdecl InitPython(const char* python_home)
    {
        try
        {
            if (Py_IsInitialized()) return 1;

            PyConfig config;
            PyConfig_InitPythonConfig(&config);

            // Convert python_home from char* to wchar_t*
            wchar_t home_buf[512];
            size_t converted = 0;
            mbstowcs_s(&converted, home_buf, 512, python_home, _TRUNCATE);

            PyConfig_SetString(&config, &config.home, home_buf);

            PyStatus status = Py_InitializeFromConfig(&config);
            PyConfig_Clear(&config);

            if (PyStatus_Exception(status)) return 0;

            return Py_IsInitialized() ? 1 : 0;
        }
        catch (...)
        {
            return 0;
        }
    }

    void __cdecl ShutdownPython()
    {
        try
        {
            if (Py_IsInitialized()) Py_Finalize();
        }
        catch (...) {}
    }

    int __cdecl OpenImageTest(const char* image_path, char* out_buf, int out_buf_len)
    {
        if (!Py_IsInitialized() || !image_path || !out_buf || out_buf_len <= 0)
            return 0;

        try
        {
            // Build and run a small Python snippet inline
            // Imports Pillow, opens the image, writes width x height into result
            std::string script =
                "import sys\n"
                "try:\n"
                "    from PIL import Image\n"
                "    img = Image.open(r'" + std::string(image_path) + "')\n"
                "    _result = f'{img.width}x{img.height}'\n"
                "except Exception as e:\n"
                "    _result = f'ERROR: {e}'\n";

            // Run the script in a fresh dict so we can read _result back
            PyObject* globals = PyDict_New();
            PyObject* locals = PyDict_New();
            PyRun_String(script.c_str(), Py_file_input, globals, locals);

            // Pull _result out of locals
            PyObject* result_obj = PyDict_GetItemString(locals, "_result");
            if (!result_obj)
            {
                Py_DECREF(globals);
                Py_DECREF(locals);
                return 0;
            }

            const char* result_str = PyUnicode_AsUTF8(result_obj);
            if (!result_str)
            {
                Py_DECREF(globals);
                Py_DECREF(locals);
                return 0;
            }

            strncpy_s(out_buf, out_buf_len, result_str, _TRUNCATE);

            Py_DECREF(globals);
            Py_DECREF(locals);
            return 1;
        }
        catch (...)
        {
            return 0;
        }
    }

} // extern "C"