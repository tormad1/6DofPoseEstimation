#include <iostream>
#include <string>
#include "GigaPoseBridge.h"

int main()
{
    const char* python_home = "C:\\Users\\solar\\miniconda3";

    std::cout << "Initialising Python..." << std::endl;
    int init_result = InitPython(python_home);
    std::cout << "InitPython returned: " << init_result << std::endl;

    if (!init_result)
    {
        std::cout << "Python failed to initialise. Exiting." << std::endl;
        return 1;
    }

    // Create a small test image to open
    const char* test_image = "C:\\Users\\solar\\test_frame.png";

    char out_buf[256] = {};
    std::cout << "Calling OpenImageTest..." << std::endl;
    int result = OpenImageTest(test_image, out_buf, sizeof(out_buf));

    std::cout << "OpenImageTest returned: " << result << std::endl;
    std::cout << "Output: " << out_buf << std::endl;

    ShutdownPython();
    std::cout << "Python shut down cleanly." << std::endl;
    return 0;
}