
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

#include <fstream>
#include <iostream>
#include <vector>
#include "Point.h"
#include "Finder.h"
#include "Space_map2.h"
#include "Loader.h"


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

struct mim_dist
{
    __host__ __device__
        double operator()(const double& Z, const double& Y) const {
        return (Z < Y) ? Z : Y;
    }
};

typedef thrust::host_vector<double> Hvec;
typedef thrust::device_vector<double> Dvec;
double min_dist_calculation(const Hvec& Px, const Hvec& Py, const Hvec& Pz, const Point& target, const double& beta2);

double unsigned_distance_space_map_cuda(const Points& points, const Point& target, double beta, double map_size, int& nearest_point) {

//    cout << test_functor(target.z)(1.1, 1.2);
    nearest_point = -1;
    Space_map2 space_map(/* with input points as */ points, /* map_size as */ map_size);

    vector<int> point_indexes;
    int n = point_indexes.size();
    double beta2 = beta * beta;

    space_map.lookup_region(target, beta, point_indexes);

    Hvec X(n), Y(n), Z(n);
    for (const int& i : point_indexes)
    {
        X.push_back(points[i].x);
        Y.push_back(points[i].y);
        Z.push_back(points[i].z);
    }

    return min_dist_calculation(X, Y, Z, target, beta2);
}

double min_dist_calculation(const Hvec& Px, const Hvec& Py, const Hvec& Pz, const Point& target, const double& beta2) {
    Dvec X = Px, Y = Py, Z = Pz;

    thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), dist_sqxy(target.x,target.y));
    thrust::transform(Z.begin(), Z.end(), Y.begin(), Y.begin(), dist_sqz(target.z));

    return thrust::reduce(Y.begin(), Y.end(), beta2, mim_dist());
}

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
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

double unsigned_distance_space_map(const Points& points, const Point& target, double beta, double map_size, int& nearest_point) {

    nearest_point = -1;
    Space_map::initialize_space_map(/* with input points as */ points,/* map_size as */ map_size, /*  and beta as */ beta);
    double unsigned_dist = Space_map::search_space_map(points, target, nearest_point);
    Space_map::make_empty();

    return unsigned_dist;
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


int main()
{

    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n\naddWithCuda failed!\n\n");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n\ncudaDeviceReset failed!\n\n");
        return 1;
    }


    cout << "******************************************************************" << endl << endl;
    cout << "******************************************************************" << endl << endl;
    cout << "******************************************************************" << endl << endl;



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
    int nearest_point3;

    cout << "Beta : " << beta << endl;
    cout << "Map_size : " << map_size << endl << endl;
    cout << "Target point : "; target.print();
    cout << "Points : " << endl;

    //    for(const Point& p : points) p.print();

    cout << "------------------------------------------------------" << endl;
    cout << endl;
    cout << "Unsigned_distance_space_map Debug log" << endl;
    float dist3 = unsigned_distance_space_map_cuda(points, target, beta, map_size, nearest_point3);
    cout << endl << endl;

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

    cout << "------------------------------------------------------" << endl << endl;
    cout << "Unsigned_distance_space_map cuda output..." << endl;
    cout << "Cuda Distance is : " << dist3;

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n\ncudaSetDevice failed!  \n\nDo you have a CUDA-capable GPU installed?\n\n");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n\ncudaMalloc failed!\n\n");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n\ncudaMalloc failed!\n\n");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n\ncudaMalloc failed!\n\n");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n\ncudaMalloc failed!\n\n");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n\ncudaMemcpy failed!\n\n");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n\naddKernel launch failed: %s\n\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n\ncudaDeviceSynchronize returned error code %d after launching addKernel!\n\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n\ncudaMemcpy failed!\n\n");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
