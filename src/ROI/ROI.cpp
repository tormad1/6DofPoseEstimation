#include "crop.hpp"
#include "model.hpp"
#include "test.hpp"
#include <iostream>

int main() {

	// Init env.
	OrtContext ctx(L"resources/models/yolov8m-canned.onnx");


	//auto originalImage = loadImage("monster-2.jpg");
	//auto letterboxed = letterboxResizeImage(originalImage, 640, 640);
	//runInference(originalImage, letterboxed, ctx);

	test_allMonsters(ctx);


	return 0;
}