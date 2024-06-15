// unsigned_distance_function.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#pragma once
// fStream - STD File I/O Library
#include <fstream>
#include <iostream>
#include <vector>
#include "../Geometry/Point.h"
#include "../Containers/Space_map2.h"
#include "../Input_output/Loader.h"
#include "unsigned_distance_function.h"


float unsigned_distance_brute_force(const Points& points, const Point& target, double beta, int& nearest_point) {
    nearest_point = -1;
    float min_dist = target.dist(points[0]);
    int i = 0;
    double beta2 = beta * beta;
    for (const Point& p : points)
    {
        float dist = target.dist(p);
        if (dist < min_dist)
        {
            min_dist = dist;
            nearest_point = i;
        }
        i++;
    }
    return (min_dist > beta2) ? beta2 : min_dist;
}

double unsigned_distance_space_map2(const Points& points, const Point& target, double beta, double map_size, int& nearest_point) {

    nearest_point = -1;
    Space_map2 space_map(/* with input points as */ points, /* map_size as */ map_size);
    double unsigned_dist = space_map.search_space_map(points, target, beta, nearest_point);
    space_map.make_empty();
    return unsigned_dist;
}

void print_output(float dist, int nearest_point, const Point& target, const Points& points) {
    cout << "Unsigned distance : " << sqrt(dist) << endl;
    cout << "Target point : "; target.print();
    cout << "Nearest point : ";
    if (nearest_point >= 0) points[nearest_point].print();
    else cout << "Point not found!" << endl;

    cout << endl << endl;
}

void test_local()
{
//    load_file();

    Point target = { 0,1,1.2 };
    double beta = 2;
    double map_size = 0.5;
    // Points points = {Point(0,0,0),Point(0,0,1),Point(0,1,1),Point(0,1,0)};
    Points points;

    objl::Mesh mesh;
    get_mesh("piston.obj", mesh);
    get_points(mesh, points);

    int nearest_point1;
    int nearest_point2;

    cout << "Beta : " << beta <<endl;    
    cout << "Map_size : " << map_size << endl << endl;
    cout << "Target point : "; target.print();
    cout << "Points : " << endl;

//    for(const Point& p : points) p.print();

    cout << "------------------------------------------------------" << endl;
    cout << endl;
    cout << "Unsigned_distance_space_map Debug log" << endl;
    float dist2 = unsigned_distance_space_map2(points, target, beta, map_size, nearest_point2);
    cout << endl << endl;

    cout << "******************************************************************" << endl << endl;
    cout << "Unsigned_distance_brute_force output" << endl;
    float dist1 = unsigned_distance_brute_force(points, target, beta, nearest_point1);
    print_output(dist1, nearest_point1, target, points);

    cout << "------------------------------------------------------" << endl << endl;
    cout << "Unsigned_distance_space_map output..." << endl;
    print_output(dist2, nearest_point2, target, points);

}

__device__ __host__
double get_nearest_point(const Point& P0, const Point& P1, const Point& P2, const Point& target) {
    double zero = 0.0;
    double one = 1.0;
    double two = 2.0;
    Point diff = P0 - target;
    Point edge0 = P1 - P0;
    Point edge1 = P2 - P0;
    double  a00 = _DOT_(edge0, edge0);
    double  a01 = _DOT_(edge0, edge1);
    double  a11 = _DOT_(edge1, edge1);
    double  b0 = _DOT_(diff, edge0);
    double  b1 = _DOT_(diff, edge1);
    double  det = std::max(a00 * a11 - a01 * a01, zero);
    double  s = a01 * b1 - a11 * b0;
    double  t = a01 * b0 - a00 * b1;

    if (s + t <= det)
    {
        if (s < zero)
        {
            if (t < zero)  // region 4
            {
                if (b0 < zero)
                {
                    t = zero;
                    if (-b0 >= a00)
                        s = one;
                    else
                        s = -b0 / a00;
                }
                else
                {
                    s = zero;
                    if (b1 >= zero)
                        t = zero;
                    else if (-b1 >= a11)
                        t = one;
                    else
                        t = -b1 / a11;
                }
            }
            else  // region 3
            {
                s = zero;
                if (b1 >= zero)
                    t = zero;
                else if (-b1 >= a11)
                    t = one;
                else
                    t = -b1 / a11;
            }
        }
        else if (t < zero)  // region 5
        {
            t = zero;
            if (b0 >= zero)
                s = zero;
            else if (-b0 >= a00)
                s = one;
            else
                s = -b0 / a00;
        }
        else  // region 0
        {
            // minimum at interior point
            s /= det;
            t /= det;
        }
    }
    else
    {
        double tmp0{}, tmp1{}, numer{}, denom{};

        if (s < zero)  // region 2
        {
            tmp0 = a01 + b0;
            tmp1 = a11 + b1;
            if (tmp1 > tmp0)
            {
                numer = tmp1 - tmp0;
                denom = a00 - two * a01 + a11;
                if (numer >= denom)
                {
                    s = one;
                    t = zero;
                }
                else
                {
                    s = numer / denom;
                    t = one - s;
                }
            }
            else
            {
                s = zero;
                if (tmp1 <= zero)
                    t = one;
                else if (b1 >= zero)
                    t = zero;
                else
                    t = -b1 / a11;
            }
        }
        else if (t < zero)  // region 6
        {
            tmp0 = a01 + b1;
            tmp1 = a00 + b0;
            if (tmp1 > tmp0)
            {
                numer = tmp1 - tmp0;
                denom = a00 - two * a01 + a11;
                if (numer >= denom)
                {
                    t = one;
                    s = zero;
                }
                else
                {
                    t = numer / denom;
                    s = one - t;
                }
            }
            else
            {
                t = zero;
                if (tmp1 <= zero)
                    s = one;
                else if (b0 >= zero)
                    s = zero;
                else
                    s = -b0 / a00;
            }
        }
        else  // region 1
        {
            numer = a11 + b1 - a01 - b0;
            if (numer <= zero)
            {
                s = zero;
                t = one;
            }
            else
            {
                denom = a00 - two * a01 + a11;
                if (numer >= denom)
                {
                    s = one;
                    t = zero;
                }
                else
                {
                    s = numer / denom;
                    t = one - s;
                }
            }
        }
    }
    
    auto res = P0 + edge0 * s + edge1 * t;
    return _DISTANCE_(res,target);
}


// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
