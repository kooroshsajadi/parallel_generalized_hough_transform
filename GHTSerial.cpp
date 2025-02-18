#include <iostream>
#include <vector>
#include <utility> // For std::pair

// Structure to represent a point (x, y)
typedef std::pair<int, int> Point;

// Model representation: A vector of points relative to the reference point
typedef std::vector<Point> Model;

// Image representation: A 2D vector of pixel values (e.g., grayscale)
typedef std::vector<std::vector<int>> Image;