#include <filesystem>
#include <iostream>
#include <string>

#include "GigaPoseBridge.h"

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: GigaPoseBridgeTest <repo_root>" << std::endl;
        return 1;
    }

    std::filesystem::path repo_root = argv[1];
    std::filesystem::path python_home =
        repo_root / "gigaposeFork" / ".python" / "python-3.11.9-embed-amd64";

    std::cout << "Initialising Python..." << std::endl;
    int init_python_result = InitPython(python_home.string().c_str());
    std::cout << "InitPython returned: " << init_python_result << std::endl;
    if (init_python_result != 1)
    {
        return 1;
    }

    char out_buf[512] = {};
    int init_runtime_result = InitGigaPoseRuntime(
        repo_root.string().c_str(),
        1,
        0,
        out_buf,
        sizeof(out_buf)
    );
    std::cout << "InitGigaPoseRuntime returned: "
              << init_runtime_result
              << " (" << out_buf << ")" << std::endl;

    ShutdownPython();
    return init_runtime_result == 1 ? 0 : 1;
}
