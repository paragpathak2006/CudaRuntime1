#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../Geometry/Point.h"
#include "../Containers/Space_map2.h"

typedef thrust::host_vector<double> Host_Vector;   typedef thrust::host_vector<Point> Host_Points;
typedef thrust::device_vector<double> Device_Vector; typedef thrust::device_vector<Point> Device_Points;

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

struct dist2_point
{
    const Point target;
    dist2_point(Point _target) : target(_target) {}

    __host__ __device__
        double operator()(const Point& P) {
        return _DISTANCE_(P,target);
    }
};

double min_dist_calculation2(const Points& candidate_points, const Point& target, const double& beta2) {
    Device_Points device_candidate_points = candidate_points;
    Device_Vector candidate_min_distances(candidate_points.size());

    // apply the transformation
    thrust::transform(_ITER_(device_candidate_points), candidate_min_distances.begin(), dist2_point(target));
    return thrust::reduce(_ITER_(candidate_min_distances), beta2, min_dist());
}

double unsigned_distance_space_map_cuda(const Points& points, const Point& target, double beta, double map_size, int& nearest_point) {

    nearest_point = -1;
    Space_map2 space_map(/* with input points as */ points, /* map_size as */ map_size);

    vector<int> candidate_point_indexes;
    double beta2 = beta * beta;

    space_map.get_nearby_candidate_points(target, beta, candidate_point_indexes);

    int num_candidate_points = candidate_point_indexes.size();
    Points candidate_points(num_candidate_points);
    for (const int& i : candidate_point_indexes)
        candidate_points.push_back(points[i]);

    return min_dist_calculation2(candidate_points, target, beta2);
}

// test code only. not for use
//typedef vector<int> Index;
//typedef vector<Index>   Indexes;
//
//typedef thrust::host_vector<Index>   HIndexes;
//typedef thrust::device_vector<Index> DIndexes;

//double min_dist_calculation(const Host_Vector& Px, const Host_Vector& Py, const Host_Vector& Pz, const Point& target, const double& beta2) {
//    Device_Vector X = Px, Y = Py, Z = Pz;
//    // apply the transformation
//    thrust::for_each(_POINT_(X, Y, Z), CALC_DIST_TO_(target));
//    return thrust::reduce(_ITER_(Y), beta2, min_dist());
//}

    //Host_Vector X(n), Y(n), Z(n);
    //for (const int& i : candidate_point_indexes)
    //{
    //    X.push_back(points[i].x);
    //    Y.push_back(points[i].y);
    //    Z.push_back(points[i].z);
    //}
//    return min_dist_calculation(X, Y, Z, target, beta2);

//#define _px_ thrust::get<0>(t)
//#define _py_ thrust::get<1>(t)
//#define _pz_ thrust::get<2>(t)
//#define _TGET_(i) thrust::get<i>
//#define _SQ_DIFF_(P,Q) (P - Q)*(P - Q)
//struct dist2_tuple
//{
//    const double x, y, z;
//    dist2_tuple(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}
//
//    template <typename Tuple>
//    __host__ __device__
//        void operator()(Tuple t) {
//        _py_ = _SQ_DIFF_(_px_, x) + _SQ_DIFF_(_py_, y) + _SQ_DIFF_(_pz_, z);
//    }
//};

//struct dist_sqxy
//{
//    const double x, y;
//    dist_sqxy(double _x, double _y) : x(_x), y(_y) {}
//    __host__ __device__
//        double operator()(const double& X, const double& Y) const {
//        return (X - x) * (X - x) + (Y - y) * (Y - y);
//    }
//};
//
//struct dist_sqz
//{
//    const double z;
//    dist_sqz(double _z) : z(_z) {}
//    __host__ __device__
//        double operator()(const double& Z, const double& Y) const {
//        return (Z - z) * (Z - z) + Y;
//    }
//};
