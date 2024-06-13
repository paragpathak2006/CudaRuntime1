#pragma once
#include <iostream>
#include <string>
#include <vector>
using namespace std;

class Point {
public:
    double x = 0, y = 0, z = 0;
    Point(double X, double Y, double Z) : x(X), y(Y), z(Z) {}
    Point() { x = 0; y = 0; z = 0; }

    Point operator+(const Point& rhs) const { return Point(x + rhs.x, y + rhs.y, z + rhs.z); }
    Point operator-(const Point& rhs) const { 
        return Point(x - rhs.x, y - rhs.y, z - rhs.z); 
    }
    Point operator/(double d) { return Point(x / d, y / d, z / d); }
    Point operator*(double d) { return Point(x * d, y * d, z * d); }
    bool operator==(Point rhs) { return x == rhs.x && y == rhs.y && z == rhs.z; }
    Point operator*(Point rhs) {
        return Point(
            y * rhs.z - z * rhs.y,
            z * rhs.x - x * rhs.z,
            x * rhs.y - y * rhs.x
        );
    }

    void print() const { cout << "(" << x << "," << y << "," << z << ")" <<endl; }


    //Vector2D rotate_90(Vector2D v) { return Vector2D(-v.y, v.x); }
    //Point get_normal(Point P1, Point P3) {        return (*this * 2 - P1 - P3) * sense(P1, P3) / 2;    }

    //int sense(Point P1, Point P3) {
    //    auto v1 = Vector2D(P1);
    //    auto v2 = Vector2D(*this );
    //    auto v3 = Vector2D(P3);
    //    return v2.sense(v1, v2);
    //}

    void translate(Point& v) {
        x += v.x;
        y += v.y;
        z += v.z;
    }

    Point normalize() {
        double d = sqrt(x * x + y * y + z * z);
        x = x / d;
        y = y / d;
        z = z / d;
        return Point(x, y, z);
    }

    double length() const { 
        return x * x + y * y + z * z; 
    };
    double dist(const Point &p) const { 
        auto p0 = *this - p;
        return p0.length(); 
    };

    Point unity() {
        double d = sqrt(x * x + y * y + z * z);
        return Point(x / d, y / d, z / d);
    }

};

typedef vector<Point> Points;