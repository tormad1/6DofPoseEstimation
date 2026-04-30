#include "pch.h"
#include <iostream>
#include "debug.h"

void print_DEBUG(const std::string& msg, bool DEBUG) {
	if (DEBUG) {
	std::cout << "DEBUG: " << msg << std::endl;
	}
	else {
		return;
	}
}