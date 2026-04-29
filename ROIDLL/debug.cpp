#include <iostream>
#include "debug.hpp"

void print_DEBUG(const std::string& msg, bool DEBUG) {
	if (DEBUG) {
	std::cout << "DEBUG: " << msg << std::endl;
	}
	else {
		return;
	}
}