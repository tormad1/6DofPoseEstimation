#include "pch.h"
#include "debug.h"

#include <iostream>

void print_DEBUG(const std::string& msg, bool DEBUG) {
	if (DEBUG) {
	std::cout << "DEBUG: " << msg << std::endl;
	}
	else {
		return;
	}
}