#pragma once
#include <tuple>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "../Geometry/Point.h"
#include "../Geometry/Face.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class Point_index
{
public:
    int x, y, z, index = -1e5;
    Point_index(int X, int Y, int Z) : x(X), y(Y), z(Z) {}
    Point_index(int X, int Y, int Z, int _index) : x(X), y(Y), z(Z), index(_index) {}

    Point_index(const Point& P, const double& map_size)
    {
        x = round(P.x / map_size); 
        y = round(P.y / map_size);
        z = round(P.z / map_size);
    }

    Point_index(const Point& P, int i, const double& map_size)
    {
        x = round(P.x / map_size);
        y = round(P.y / map_size);
        z = round(P.z / map_size);
        index = i;
    }

    Point_index() { x = 0; y = 0; z = 0; index = -1e5; }
    bool operator==(const Point_index& rhs) const { return x == rhs.x && y == rhs.y && z == rhs.z; }
    void print() const { cout << "(" << x << "," << y << "," << z << ")" << endl; }
    void print2() const { cout << "(" << x << "," << y << "," << z << ")"; }
    size_t get_hash() const {return _HASH_(x, y, z);}
};

// Point Hash functor
struct Hash_of_Point_index {
    size_t operator() (const Point_index& P) const {
        return _HASH_(P.x, P.y, P.z);
    }
};

struct Bucket {
    int first, count;
    Bucket(int _first, int _count) : first(_first), count(_count) {}
};

typedef unordered_multimap< Point_index, int, Hash_of_Point_index> Point_Index_Map;
typedef vector<Bucket> Buckets;
typedef vector<Point_index> Point_indexes;

#define FOR_RANGE(it,range) for(auto& it = range.first; it != range.second; ++it)
#define _MAX2_(A,B) ((A > B) ? A : B) 
#define _MIN2_(A,B) ((A < B) ? A : B) 
#define _MAX3_(A,B,C) (A > B) ? _MAX2_(A,C) : _MAX2_(B,C)
#define _MIN3_(A,B,C) (A < B) ? _MIN2_(A,C) : _MIN2_(B,C)
class Space_map2
{
    double map_size;
public:
    Point_Index_Map point_map;
    Buckets buckets;
    Point_indexes point_indexes;

    Space_map2(const Points& points, const double& mapsize) {
        int num_points = points.size();
        map_size = mapsize;
        point_map.reserve(num_points * 2);

        for (size_t i = 0; i < num_points; i++)
            point_map.emplace(Point_index(points[i], i, map_size), i);
    }


    Space_map2(const Faces faces, const Points& points, const double& mapsize) {
        int num_faces = points.size();
        map_size = mapsize;
        point_map.reserve(num_faces * 5);

        for (size_t iFace = 0; iFace < num_faces; iFace++) {
            Point_index p0 = Point_index(points[faces[iFace].v[0]], iFace, map_size);
            Point_index p1 = Point_index(points[faces[iFace].v[1]], iFace, map_size);
            Point_index p2 = Point_index(points[faces[iFace].v[2]], iFace, map_size);

            int px_min = _MIN3_(p0.x, p1.x, p2.x);            int px_max = _MAX3_(p0.x, p1.x, p2.x);
            int py_min = _MIN3_(p0.y, p1.y, p2.y);            int py_max = _MAX3_(p0.y, p1.y, p2.y);
            int pz_min = _MIN3_(p0.z, p1.z, p2.z);            int pz_max = _MAX3_(p0.z, p1.z, p2.z);

            for (int i = px_min; i <= px_max; i++)
            for (int j = py_min; j <= py_max; j++)
            for (int k = pz_min; k <= pz_max; k++)
                point_map.emplace(Point_index(i,j,k, iFace), iFace);

        }
    }

    void generate_cuda_hashmap() {
        int first = 0, count = 0;
        auto bucket_count = point_map.bucket_count();

        for (unsigned i = 0; i < bucket_count; ++i) {
            count = 0;
            for (auto local_it = point_map.begin(i); local_it != point_map.end(i); ++local_it)
            {
                point_indexes.push_back(local_it->first);
                count++;
            }
            buckets.push_back(Bucket(first, count));
            first = first + count;
        }

    }

    void lookup_point_region(const Point_index& P, vector<int>& point_indexes) {
        auto count = point_map.count(P);

        if (count > 0) {
            auto range = point_map.equal_range(P);
            FOR_RANGE(point, range) {
                int point_index = point->second;
                point_indexes.push_back(point_index);
            }
        }
    }

    void get_nearby_candidate_points(const Point& target, const double& beta, vector<int>& point_indexes) {
        int max_index = round(beta / map_size);
        Point_index target_index(target,map_size);
        point_indexes.reserve(50);

        for (int i = target_index.x - max_index; i <= target_index.x + max_index; i++)
        for (int j = target_index.y - max_index; j <= target_index.y + max_index; j++)
        for (int k = target_index.z - max_index; k <= target_index.z + max_index; k++)
            lookup_point_region(Point_index( i, j, k), point_indexes);
    }

    double search_space_map(const Points& points, const Point& target, const double& beta, int& nearest_point ) {
        vector<int> point_indexes;
        double beta2 = beta * beta;
        get_nearby_candidate_points(target, beta, point_indexes);

        double min_dist = target.dist(points[0]);

        for (int i : point_indexes)
        {
            float dist = target.dist(points[i]);
            if (dist < min_dist && dist < beta2)
            {
                min_dist = dist;
                nearest_point = i;
            }
        }
        return (min_dist > beta2) ? beta2 : min_dist;
    }

    bool make_empty() {
        return point_map.empty();
    }


    //double search_space_map_parallel(const Points& points, const Point& target, const double& beta, int& nearest_point) {
    //    vector<int> point_indexes;
    //    vector<double> dists;
    //    double beta2 = beta * beta;
    //    get_nearby_candidate_points(target, beta, point_indexes);
    //    Points points_filtered;
    //    points_filtered.reserve(point_indexes.size());

    //    for (int i : point_indexes)
    //        points_filtered.push_back(points[i]);

    //    lookup_region_parallel(points_filtered, target, beta2, dists);
    //}

    //void lookup_region_parallel(const Points& points, const Point& target, const double& beta2, vector<double>& dist) {
    //    int i = threadIdx.x;
    //    double min_dist = target.dist(points[i]);
    //    dist[i] = (min_dist > beta2) ? beta2 : min_dist;
    //}

};
