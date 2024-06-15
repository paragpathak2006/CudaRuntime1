#pragma once
#include "Point.h"
#include "../Thrust_lib/unsigned_distance_function.h"
class Face
{
public:
	int v[3];
	Face() { v[0] = -1; v[1] = -1; v[2] = -1;}
	Face(int i,int j,int k) { v[0] = i; v[1] = j; v[2] = k; }
	Face(int _v[3]) { v[0] = _v[0]; v[1] = _v[1]; v[2] = _v[2]; }
	double dist(const Point& target, const Points& points) {
		return get_nearest_point_dist(points[v[0]], points[v[1]], points[v[2]], target);
	}
};

typedef vector<Face> Faces;

