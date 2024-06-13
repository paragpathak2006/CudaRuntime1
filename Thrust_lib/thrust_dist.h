#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../Geometry/Point.h"
#include "../Containers/Space_map2.h"

typedef thrust::host_vector<double> Hvec;   typedef thrust::host_vector<Point> HPoint;
typedef thrust::device_vector<double> Dvec; typedef thrust::device_vector<Point> DPoint;

#define _ITER_(Y) Y.begin(), Y.end()
#define _POINT_(X,Y,Z) thrust::make_zip_iterator(thrust::make_tuple(X.begin(), Y.begin(), Z.begin())) , thrust::make_zip_iterator(thrust::make_tuple(X.end(), Y.end(), Z.end()))
#define CALC_DIST_TO_(target) dist2_tuple(target.x, target.y, target.z)

struct min_dist
{
    __host__ __device__
        double operator()(const double& Z, const double& Y) const {
        return (Z < Y) ? Z : Y;
    }
};

#define _px_ thrust::get<0>(t)
#define _py_ thrust::get<1>(t)
#define _pz_ thrust::get<2>(t)
struct dist2_tuple
{
    const double x, y, z;
    dist2_tuple(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}

    template <typename Tuple>
    __host__ __device__
        void operator()(Tuple t) {
        _py_ = (_px_ - x) * (_px_ - x) + (_py_ - y) * (_py_ - y) + (_pz_ - z) * (_pz_ - z);
    }
};

struct dist2_point
{
    const Point target;
    dist2_point(Point _target) : target(_target) {}

    __host__ __device__
        double operator()(const Point& P) {
        return (P.x - target.x) * (P.x - target.x) 
             + (P.y - target.y) * (P.y - target.y) 
             + (P.z - target.z) * (P.z - target.z);
    }
};

struct dist_sqxy
{
    const double x, y;
    dist_sqxy(double _x, double _y) : x(_x), y(_y) {}
    __host__ __device__
        double operator()(const double& X, const double& Y) const {
        return (X - x) * (X - x) + (Y - y) * (Y - y);
    }
};

struct dist_sqz
{
    const double z;
    dist_sqz(double _z) : z(_z) {}
    __host__ __device__
        double operator()(const double& Z, const double& Y) const {
        return (Z - z) * (Z - z) + Y;
    }
};

double min_dist_calculation(const Hvec& Px, const Hvec& Py, const Hvec& Pz, const Point& target, const double& beta2) {
    Dvec X = Px, Y = Py, Z = Pz;
    // apply the transformation
    //thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), dist_sqxy(target.x,target.y));
    //thrust::transform(Z.begin(), Z.end(), Y.begin(), Y.begin(), dist_sqz(target.z));
    thrust::for_each(_POINT_(X, Y, Z), CALC_DIST_TO_(target));
    return thrust::reduce(_ITER_(Y), beta2, min_dist());
}

double min_dist_calculation2(const HPoint& Px, const Point& target, const double& beta2) {
    DPoint points = Px;
    Dvec distances(Px.size());
    // apply the transformation
    thrust::transform(_ITER_(points), distances.begin(), dist2_point(target));
    return thrust::reduce(_ITER_(distances), beta2, min_dist());
}

double unsigned_distance_space_map_cuda(const Points& points, const Point& target, double beta, double map_size, int& nearest_point) {

    //    cout << test_functor(target.z)(1.1, 1.2);
    nearest_point = -1;
    Space_map2 space_map(/* with input points as */ points, /* map_size as */ map_size);

    vector<int> point_indexes;
    int n = point_indexes.size();
    double beta2 = beta * beta;

    space_map.lookup_region(target, beta, point_indexes);

    HPoint P(n);
    for (const int& i : point_indexes)
        P.push_back(points[i]);

    return min_dist_calculation2(P, target, beta2);

    //Hvec X(n), Y(n), Z(n);
    //for (const int& i : point_indexes)
    //{
    //    X.push_back(points[i].x);
    //    Y.push_back(points[i].y);
    //    Z.push_back(points[i].z);
    //}
//    return min_dist_calculation(X, Y, Z, target, beta2);

}


// test code only. not for use
typedef vector<int> Index;
typedef vector<Index>   Indexes;

typedef thrust::host_vector<Index>   HIndexes;
typedef thrust::device_vector<Index> DIndexes;

void custom_undoreded_map_implementation1(const Points& points, Indexes &indexes, double map_size) {
    for (Index& index : indexes)
        index.reserve(5);

    for (size_t i = 0; i < points.size(); i++)
    {
        Point_index pi = Point_index(points[i], i, map_size);
        auto x = Hash_of_Point_index()(pi);
        auto &ind = indexes[x];
        ind.push_back(i);
    }
}

//struct Index_points
//{
//    const Point_Index_Map map;
//    Index_points(Point_Index_Map _map) : map(_map) {}
//
//    __host__ __device__
//        vector<int> operator()(const Point_index& point_index) {
//        vector<int> point_indexes(10);
//        auto range = map.equal_range(point_index);
//        FOR_RANGE(point, range) {
//            int point_index = point->second;
//            point_indexes.push_back(point_index);
//        }
//        return point_indexes;
//    }
//};

//double get_point_indexes(const thrust::host_vector<Point_indexes>& Px, const Point& target, const double& beta2,const Point_Index_Map &map) {
//    DPoint points = Px;
//    thrust::device_vector<vector<int>>  point_indexes(Px.size());

//    // apply the transformation
//    thrust::transform(_ITER_(points), point_indexes.begin(), Index_points(map));
//}
