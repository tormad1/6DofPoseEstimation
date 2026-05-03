#include "pch.h"
#include "GigaPoseBridge.h"

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <filesystem>
#include <string>

namespace
{
    bool g_runtime_initialized = false;

    std::wstring Utf8ToWide(const char* text)
    {
        if (text == nullptr || text[0] == '\0')
        {
            return L"";
        }

        const int wide_length = MultiByteToWideChar(CP_UTF8, 0, text, -1, nullptr, 0);
        if (wide_length <= 0)
        {
            return L"";
        }

        std::wstring wide_text(static_cast<size_t>(wide_length), L'\0');
        const int converted = MultiByteToWideChar(
            CP_UTF8,
            0,
            text,
            -1,
            wide_text.data(),
            wide_length
        );
        if (converted <= 0)
        {
            return L"";
        }
        wide_text.resize(static_cast<size_t>(converted - 1));
        return wide_text;
    }

    void WriteMessage(char* out_buf, int out_buf_len, const std::string& message)
    {
        if (out_buf == nullptr || out_buf_len <= 0)
        {
            return;
        }
        strncpy_s(out_buf, out_buf_len, message.c_str(), _TRUNCATE);
    }

    std::string PythonErrorToString()
    {
        if (!PyErr_Occurred())
        {
            return "unknown python error";
        }

        PyObject* exc_type = nullptr;
        PyObject* exc_value = nullptr;
        PyObject* exc_traceback = nullptr;
        PyErr_Fetch(&exc_type, &exc_value, &exc_traceback);
        PyErr_NormalizeException(&exc_type, &exc_value, &exc_traceback);

        std::string message = "python exception";
        PyObject* traceback_module = PyImport_ImportModule("traceback");
        if (traceback_module != nullptr)
        {
            PyObject* format_exception = PyObject_GetAttrString(traceback_module, "format_exception");
            if (format_exception != nullptr)
            {
                PyObject* formatted = PyObject_CallFunctionObjArgs(
                    format_exception,
                    exc_type ? exc_type : Py_None,
                    exc_value ? exc_value : Py_None,
                    exc_traceback ? exc_traceback : Py_None,
                    nullptr
                );
                if (formatted != nullptr)
                {
                    PyObject* empty = PyUnicode_FromString("");
                    if (empty != nullptr)
                    {
                        PyObject* joined = PyUnicode_Join(empty, formatted);
                        if (joined != nullptr)
                        {
                            const char* utf8 = PyUnicode_AsUTF8(joined);
                            if (utf8 != nullptr)
                            {
                                message = utf8;
                            }
                            Py_DECREF(joined);
                        }
                        Py_DECREF(empty);
                    }
                    Py_DECREF(formatted);
                }
                Py_DECREF(format_exception);
            }
            Py_DECREF(traceback_module);
        }

        Py_XDECREF(exc_type);
        Py_XDECREF(exc_value);
        Py_XDECREF(exc_traceback);
        return message;
    }

    bool AppendSysPath(const std::filesystem::path& path)
    {
        PyObject* sys_path = PySys_GetObject("path");
        if (sys_path == nullptr || !PyList_Check(sys_path))
        {
            return false;
        }

        const std::wstring wide_path = path.wstring();
        PyObject* python_path = PyUnicode_FromWideChar(
            wide_path.c_str(),
            static_cast<Py_ssize_t>(wide_path.size())
        );
        if (python_path == nullptr)
        {
            return false;
        }

        const int contains = PySequence_Contains(sys_path, python_path);
        if (contains == 0)
        {
            if (PyList_Insert(sys_path, 0, python_path) != 0)
            {
                Py_DECREF(python_path);
                return false;
            }
        }

        Py_DECREF(python_path);
        return contains >= 0;
    }

    PyObject* BuildMatrix3x3(const float* camera_k_3x3)
    {
        PyObject* matrix = PyList_New(3);
        if (matrix == nullptr)
        {
            return nullptr;
        }

        for (Py_ssize_t row = 0; row < 3; ++row)
        {
            PyObject* row_list = PyList_New(3);
            if (row_list == nullptr)
            {
                Py_DECREF(matrix);
                return nullptr;
            }

            for (Py_ssize_t column = 0; column < 3; ++column)
            {
                PyObject* value = PyFloat_FromDouble(
                    static_cast<double>(camera_k_3x3[row * 3 + column])
                );
                if (value == nullptr)
                {
                    Py_DECREF(row_list);
                    Py_DECREF(matrix);
                    return nullptr;
                }
                PyList_SET_ITEM(row_list, column, value);
            }
            PyList_SET_ITEM(matrix, row, row_list);
        }

        return matrix;
    }

    bool FillPoseFromResult(PyObject* result, int64_t timestamp_us, GigaPoseNativePose* out_pose)
    {
        if (!PyDict_Check(result) || out_pose == nullptr)
        {
            return false;
        }

        PyObject* translation = PyDict_GetItemString(result, "translation");
        PyObject* rotation = PyDict_GetItemString(result, "rotation");
        PyObject* score = PyDict_GetItemString(result, "score");

        if (
            translation == nullptr ||
            rotation == nullptr ||
            score == nullptr ||
            !PySequence_Check(translation) ||
            !PySequence_Check(rotation)
        )
        {
            return false;
        }

        if (PySequence_Size(translation) != 3 || PySequence_Size(rotation) != 4)
        {
            return false;
        }

        PyObject* tx = PySequence_GetItem(translation, 0);
        PyObject* ty = PySequence_GetItem(translation, 1);
        PyObject* tz = PySequence_GetItem(translation, 2);
        PyObject* qx = PySequence_GetItem(rotation, 0);
        PyObject* qy = PySequence_GetItem(rotation, 1);
        PyObject* qz = PySequence_GetItem(rotation, 2);
        PyObject* qw = PySequence_GetItem(rotation, 3);

        if (
            tx == nullptr || ty == nullptr || tz == nullptr ||
            qx == nullptr || qy == nullptr || qz == nullptr || qw == nullptr
        )
        {
            Py_XDECREF(tx);
            Py_XDECREF(ty);
            Py_XDECREF(tz);
            Py_XDECREF(qx);
            Py_XDECREF(qy);
            Py_XDECREF(qz);
            Py_XDECREF(qw);
            return false;
        }

        out_pose->px = static_cast<float>(PyFloat_AsDouble(tx));
        out_pose->py = static_cast<float>(PyFloat_AsDouble(ty));
        out_pose->pz = static_cast<float>(PyFloat_AsDouble(tz));
        out_pose->qx = static_cast<float>(PyFloat_AsDouble(qx));
        out_pose->qy = static_cast<float>(PyFloat_AsDouble(qy));
        out_pose->qz = static_cast<float>(PyFloat_AsDouble(qz));
        out_pose->qw = static_cast<float>(PyFloat_AsDouble(qw));
        out_pose->confidence = static_cast<float>(PyFloat_AsDouble(score));
        out_pose->timestamp_us = timestamp_us;

        Py_DECREF(tx);
        Py_DECREF(ty);
        Py_DECREF(tz);
        Py_DECREF(qx);
        Py_DECREF(qy);
        Py_DECREF(qz);
        Py_DECREF(qw);

        if (PyErr_Occurred())
        {
            return false;
        }
        return true;
    }
}


extern "C"
{
    int __cdecl InitPython(const char* python_home)
    {
        try
        {
            if (Py_IsInitialized())
            {
                return 1;
            }

            const std::wstring python_home_wide = Utf8ToWide(python_home);
            if (python_home_wide.empty())
            {
                return 0;
            }

            PyConfig config;
            PyConfig_InitPythonConfig(&config);

            PyStatus status = PyConfig_SetString(&config, &config.home, python_home_wide.c_str());
            if (PyStatus_Exception(status))
            {
                PyConfig_Clear(&config);
                return 0;
            }

            status = Py_InitializeFromConfig(&config);
            PyConfig_Clear(&config);
            if (PyStatus_Exception(status))
            {
                return 0;
            }

            PyEval_SaveThread();
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
            g_runtime_initialized = false;
            if (Py_IsInitialized())
            {
                PyGILState_Ensure();
                Py_Finalize();
            }
        }
        catch (...)
        {
        }
    }

    int __cdecl OpenImageTest(const char* image_path, char* out_buf, int out_buf_len)
    {
        if (!Py_IsInitialized() || image_path == nullptr || out_buf == nullptr || out_buf_len <= 0)
        {
            return 0;
        }

        PyGILState_STATE gil_state = PyGILState_Ensure();
        try
        {
            std::string script =
                "from PIL import Image\n"
                "try:\n"
                "    img = Image.open(r'" + std::string(image_path) + "')\n"
                "    _result = f'{img.width}x{img.height}'\n"
                "except Exception as exc:\n"
                "    _result = f'ERROR: {exc}'\n";

            PyObject* globals = PyDict_New();
            PyObject* locals = PyDict_New();
            PyRun_String(script.c_str(), Py_file_input, globals, locals);

            PyObject* result_obj = PyDict_GetItemString(locals, "_result");
            if (result_obj == nullptr)
            {
                WriteMessage(out_buf, out_buf_len, PythonErrorToString());
                Py_DECREF(globals);
                Py_DECREF(locals);
                PyGILState_Release(gil_state);
                return 0;
            }

            const char* result_str = PyUnicode_AsUTF8(result_obj);
            if (result_str == nullptr)
            {
                WriteMessage(out_buf, out_buf_len, PythonErrorToString());
                Py_DECREF(globals);
                Py_DECREF(locals);
                PyGILState_Release(gil_state);
                return 0;
            }

            WriteMessage(out_buf, out_buf_len, result_str);
            Py_DECREF(globals);
            Py_DECREF(locals);
            PyGILState_Release(gil_state);
            return 1;
        }
        catch (...)
        {
            PyGILState_Release(gil_state);
            return 0;
        }
    }

    int __cdecl InitGigaPoseRuntime(
        const char* repo_root,
        int cpu_threads,
        int warmup,
        char* out_buf,
        int out_buf_len
    )
    {
        if (!Py_IsInitialized() || repo_root == nullptr)
        {
            WriteMessage(out_buf, out_buf_len, "python not initialized");
            return 0;
        }

        PyGILState_STATE gil_state = PyGILState_Ensure();
        try
        {
            const std::filesystem::path repo_root_path = Utf8ToWide(repo_root);
            const std::filesystem::path gigapose_dir = repo_root_path / L"gigaposeFork";
            const std::filesystem::path site_packages_dir =
                gigapose_dir / L".venv" / L"Lib" / L"site-packages";

            if (!std::filesystem::exists(gigapose_dir))
            {
                WriteMessage(out_buf, out_buf_len, "gigaposeFork directory not found");
                PyGILState_Release(gil_state);
                return 0;
            }

            if (!AppendSysPath(gigapose_dir))
            {
                WriteMessage(out_buf, out_buf_len, "failed to append gigaposeFork to sys.path");
                PyGILState_Release(gil_state);
                return 0;
            }
            if (std::filesystem::exists(site_packages_dir) && !AppendSysPath(site_packages_dir))
            {
                WriteMessage(out_buf, out_buf_len, "failed to append site-packages to sys.path");
                PyGILState_Release(gil_state);
                return 0;
            }

            PyObject* module = PyImport_ImportModule("gigapose_bridge");
            if (module == nullptr)
            {
                WriteMessage(out_buf, out_buf_len, PythonErrorToString());
                PyGILState_Release(gil_state);
                return 0;
            }

            PyObject* init_runtime = PyObject_GetAttrString(module, "init_runtime");
            if (init_runtime == nullptr || !PyCallable_Check(init_runtime))
            {
                WriteMessage(out_buf, out_buf_len, "gigapose_bridge.init_runtime missing");
                Py_XDECREF(init_runtime);
                Py_DECREF(module);
                PyGILState_Release(gil_state);
                return 0;
            }

            const std::wstring repo_root_wide = repo_root_path.wstring();
            PyObject* args = PyTuple_New(1);
            PyObject* repo_root_obj = PyUnicode_FromWideChar(
                repo_root_wide.c_str(),
                static_cast<Py_ssize_t>(repo_root_wide.size())
            );
            if (args == nullptr || repo_root_obj == nullptr)
            {
                WriteMessage(out_buf, out_buf_len, "failed to build runtime init args");
                Py_XDECREF(repo_root_obj);
                Py_XDECREF(args);
                Py_DECREF(init_runtime);
                Py_DECREF(module);
                PyGILState_Release(gil_state);
                return 0;
            }
            PyTuple_SET_ITEM(args, 0, repo_root_obj);

            PyObject* kwargs = PyDict_New();
            PyObject* cpu_threads_obj = PyLong_FromLong(cpu_threads);
            if (
                kwargs == nullptr ||
                cpu_threads_obj == nullptr ||
                PyDict_SetItemString(kwargs, "cpu_threads", cpu_threads_obj) != 0 ||
                PyDict_SetItemString(kwargs, "warmup", warmup ? Py_True : Py_False) != 0
            )
            {
                WriteMessage(out_buf, out_buf_len, "failed to build runtime init kwargs");
                Py_XDECREF(cpu_threads_obj);
                Py_XDECREF(kwargs);
                Py_DECREF(args);
                Py_DECREF(init_runtime);
                Py_DECREF(module);
                PyGILState_Release(gil_state);
                return 0;
            }
            Py_DECREF(cpu_threads_obj);

            PyObject* result = PyObject_Call(init_runtime, args, kwargs);
            Py_DECREF(kwargs);
            Py_DECREF(args);
            Py_DECREF(init_runtime);
            Py_DECREF(module);

            if (result == nullptr)
            {
                WriteMessage(out_buf, out_buf_len, PythonErrorToString());
                PyGILState_Release(gil_state);
                return 0;
            }

            Py_DECREF(result);
            g_runtime_initialized = true;
            WriteMessage(out_buf, out_buf_len, "gigaPose runtime initialized");
            PyGILState_Release(gil_state);
            return 1;
        }
        catch (...)
        {
            WriteMessage(out_buf, out_buf_len, "native exception during runtime init");
            PyGILState_Release(gil_state);
            return 0;
        }
    }

    int __cdecl RunRoiPose(
        const uint8_t* rgba_data,
        int width,
        int height,
        int stride,
        const float* camera_k_3x3,
        float bbox_x,
        float bbox_y,
        float bbox_w,
        float bbox_h,
        int object_id,
        int64_t timestamp_us,
        GigaPoseNativePose* out_pose,
        char* out_buf,
        int out_buf_len
    )
    {
        if (!Py_IsInitialized() || !g_runtime_initialized)
        {
            WriteMessage(out_buf, out_buf_len, "gigaPose runtime not initialized");
            return -1;
        }
        if (
            rgba_data == nullptr ||
            camera_k_3x3 == nullptr ||
            out_pose == nullptr ||
            width <= 0 ||
            height <= 0
        )
        {
            WriteMessage(out_buf, out_buf_len, "invalid ROI input");
            return -1;
        }

        const int expected_stride = width * 4;
        if (stride <= 0)
        {
            stride = expected_stride;
        }
        if (stride < expected_stride)
        {
            WriteMessage(out_buf, out_buf_len, "stride smaller than width * 4");
            return -1;
        }

        PyGILState_STATE gil_state = PyGILState_Ensure();
        try
        {
            PyObject* module = PyImport_ImportModule("gigapose_bridge");
            if (module == nullptr)
            {
                WriteMessage(out_buf, out_buf_len, PythonErrorToString());
                PyGILState_Release(gil_state);
                return -1;
            }

            PyObject* run_roi = PyObject_GetAttrString(module, "run_roi_rgba");
            if (run_roi == nullptr || !PyCallable_Check(run_roi))
            {
                WriteMessage(out_buf, out_buf_len, "gigapose_bridge.run_roi_rgba missing");
                Py_XDECREF(run_roi);
                Py_DECREF(module);
                PyGILState_Release(gil_state);
                return -1;
            }

            PyObject* kwargs = PyDict_New();
            PyObject* roi_bytes = PyBytes_FromStringAndSize(
                reinterpret_cast<const char*>(rgba_data),
                static_cast<Py_ssize_t>(stride) * static_cast<Py_ssize_t>(height)
            );
            PyObject* camera_matrix = BuildMatrix3x3(camera_k_3x3);
            PyObject* bbox_xywh = Py_BuildValue("[ffff]", bbox_x, bbox_y, bbox_w, bbox_h);
            PyObject* width_obj = PyLong_FromLong(width);
            PyObject* height_obj = PyLong_FromLong(height);
            PyObject* stride_obj = PyLong_FromLong(stride);
            PyObject* object_id_obj = PyLong_FromLong(object_id);

            if (
                kwargs == nullptr ||
                roi_bytes == nullptr ||
                camera_matrix == nullptr ||
                bbox_xywh == nullptr ||
                width_obj == nullptr ||
                height_obj == nullptr ||
                stride_obj == nullptr ||
                object_id_obj == nullptr ||
                PyDict_SetItemString(kwargs, "roi_bytes", roi_bytes) != 0 ||
                PyDict_SetItemString(kwargs, "width", width_obj) != 0 ||
                PyDict_SetItemString(kwargs, "height", height_obj) != 0 ||
                PyDict_SetItemString(kwargs, "stride", stride_obj) != 0 ||
                PyDict_SetItemString(kwargs, "K", camera_matrix) != 0 ||
                PyDict_SetItemString(kwargs, "bbox_xywh", bbox_xywh) != 0 ||
                PyDict_SetItemString(kwargs, "object_id", object_id_obj) != 0
            )
            {
                WriteMessage(out_buf, out_buf_len, "failed to build ROI kwargs");
                Py_XDECREF(kwargs);
                Py_XDECREF(roi_bytes);
                Py_XDECREF(camera_matrix);
                Py_XDECREF(bbox_xywh);
                Py_XDECREF(width_obj);
                Py_XDECREF(height_obj);
                Py_XDECREF(stride_obj);
                Py_XDECREF(object_id_obj);
                Py_DECREF(run_roi);
                Py_DECREF(module);
                PyGILState_Release(gil_state);
                return -1;
            }

            Py_DECREF(roi_bytes);
            Py_DECREF(camera_matrix);
            Py_DECREF(bbox_xywh);
            Py_DECREF(width_obj);
            Py_DECREF(height_obj);
            Py_DECREF(stride_obj);
            Py_DECREF(object_id_obj);

            PyObject* empty_args = PyTuple_New(0);
            PyObject* result = PyObject_Call(run_roi, empty_args, kwargs);
            Py_DECREF(empty_args);
            Py_DECREF(kwargs);
            Py_DECREF(run_roi);
            Py_DECREF(module);

            if (result == nullptr)
            {
                WriteMessage(out_buf, out_buf_len, PythonErrorToString());
                PyGILState_Release(gil_state);
                return -1;
            }

            if (result == Py_None)
            {
                Py_DECREF(result);
                WriteMessage(out_buf, out_buf_len, "no pose");
                PyGILState_Release(gil_state);
                return 0;
            }

            if (!FillPoseFromResult(result, timestamp_us, out_pose))
            {
                Py_DECREF(result);
                WriteMessage(out_buf, out_buf_len, "invalid pose result payload");
                PyGILState_Release(gil_state);
                return -1;
            }

            Py_DECREF(result);
            WriteMessage(out_buf, out_buf_len, "ok");
            PyGILState_Release(gil_state);
            return 1;
        }
        catch (...)
        {
            WriteMessage(out_buf, out_buf_len, "native exception during ROI inference");
            PyGILState_Release(gil_state);
            return -1;
        }
    }
}
