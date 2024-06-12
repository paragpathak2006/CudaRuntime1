// unsigned_distance_function.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#pragma once
// fStream - STD File I/O Library
#include <fstream>
#include <iostream>
#include <vector>
#include "Point.h"
#include "Finder.h"
#include "Space_map2.h"
#include "Loader.h"
#include "unsigned_distance_function.h"



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

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
