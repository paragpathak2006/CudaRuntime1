
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

#include "Geometry/Point.h"
#include "Containers/Space_map2.h"
#include "Input_output/Loader.h"
#include "Geometry/Point.h"
#include "Thrust_lib/thrust_dist.h"
#include "Thrust_lib/unsigned_distance_function.h"

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ 
void addKernel(int *c, const int *a, const int *b){
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

#define _MINIMUM_(A,B) A < B ? A : B
#define _LINEAR_INDEX_(i,j,k,dim) i + j * dim + k * dim * dim
__global__
void calculate_min_dist(
    const Bucket* buckets, const Point_index* indexes, const Point* points, double* min_distances,
    Point target, double beta2, int bucket_count, int dim,
    int i0, int j0, int k0)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i >= dim || j >= dim || k >= dim)
        return;

    int bucket_index = _HASH_(i0 + i, j0 + j, k0 + k) % bucket_count;

    int first = buckets[bucket_index].first;
    int count = buckets[bucket_index].count;
    double min_distance = beta2, dist;

    for (size_t iter = first; iter < count; iter++)
    {
        const Point_index& p = indexes[iter];
        if (p.x == i && p.y == j && p.z == k)
            dist = _DISTANCE_(points[p.index], target);

        min_distance = _MINIMUM_(min_distance, dist);
    }
    min_distances[_LINEAR_INDEX_(i, j, k, dim)] = min_distance;
}


#define _DIM3_(x) x,x,x
#define _CALC_BLOCK_DIM_(n,t) (n+t-1)/t
#define _MAP_INDEX_(x,y) round(x/y)
#define _CAST_(P) thrust::raw_pointer_cast(P.data())
#define _RAW_CAST_(P,Q,R,S) _CAST_(P) ,_CAST_(Q) , _CAST_(R) , _CAST_(S)
double cuda_hash_map_implementation(const Points& points, const Point& target, double map_size,double beta)
{
    Space_map2 space_map(/* with input points as */ points, /* map_size as */ map_size);
    space_map.generate_cuda_hashmap();

    Point_index target_index(target, map_size);
    auto beta2 = beta * beta;
    auto bucket_count = space_map.buckets.size();
    int max_index = _MAP_INDEX_(beta,map_size); max_index = max_index + max_index % 2;
    int num_threads = 2 * max_index;
    int threads_dim = 4;

    int blocks_dim = _CALC_BLOCK_DIM_(num_threads,threads_dim);
    dim3 threads_per_block(_DIM3_(threads_dim));
    dim3 blocks_per_grid(_DIM3_(blocks_dim));

    thrust::device_vector<Bucket>       buckets = space_map.buckets;
    thrust::device_vector<Point_index>  point_indexes = space_map.point_indexes;
    thrust::device_vector<Point>        Dpoints = points;
    thrust::device_vector<double>       min_distances(num_threads * num_threads * num_threads);

    calculate_min_dist << <blocks_per_grid, threads_per_block >> > (
        _RAW_CAST_(buckets, point_indexes, Dpoints, min_distances),
        target, beta2, bucket_count, 2 * max_index,
        target_index.x - max_index, target_index.y - max_index, target_index.z - max_index
        );

    return thrust::reduce(_ITER_(min_distances), beta2, min_dist());

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
    get_mesh("3DObjects/piston.obj", mesh);
    get_points(mesh, points);

    int nearest_point0 = -1;
    int nearest_point1 = -1;
    int nearest_point2 = -1;
    int nearest_point3 = -1;

    cout << "Beta : " << beta << endl;
    cout << "Map_size : " << map_size << endl << endl;
    cout << "Target point : "; target.print();
    cout << "Points : " << endl;

    //    for(const Point& p : points) p.print();

    cout << "------------------------------------------------------" << endl;
    cout << "Unsigned_distance_cuda_hash_table Debug log" << endl;
    float dist0 = cuda_hash_map_implementation(points, target, map_size, beta);
    print_output(dist0, nearest_point0, target, points);
    cout << endl << endl;

    cout << "------------------------------------------------------" << endl;
    cout << "Unsigned_distance_space_map Debug log" << endl;
    float dist1 = unsigned_distance_space_map_cuda(points, target, beta, map_size, nearest_point1);
    print_output(dist1, nearest_point1, target, points);
    cout << endl << endl;

    cout << "------------------------------------------------------" << endl;
    cout << "Unsigned_distance_space_map Debug log" << endl;
    float dist2 = unsigned_distance_space_map2(points, target, beta, map_size, nearest_point2);
    print_output(dist2, nearest_point2, target, points);
    cout << endl << endl;

    cout << "------------------------------------------------------" << endl;
    cout << "Unsigned_distance_brute_force output" << endl;
    float dist3 = unsigned_distance_brute_force(points, target, beta, nearest_point3);
    print_output(dist3, nearest_point3, target, points);

    return 0;
}

#define _SETUP_FAILED_ if (cudaStatus != cudaSuccess) {fprintf(stderr, "\n\ncudaSetDevice failed!  \n\nDo you have a CUDA-capable GPU installed?\n\n");goto Error;}
#define _MALLOC_FAILED_ if (cudaStatus != cudaSuccess) {fprintf(stderr, "\n\ncudaMalloc failed!\n\n");goto Error;}
#define _SYNC_FAILED_ if (cudaStatus != cudaSuccess) {fprintf(stderr, "\n\ncudaDeviceSynchronize returned error code %d after launching addKernel!\n\n");goto Error;}
#define _KERNAL_FAILED_ if (cudaStatus != cudaSuccess) {    fprintf(stderr, "\n\naddKernel launch failed: %s\n\n", cudaGetErrorString(cudaStatus));    goto Error;}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0); _SETUP_FAILED_

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int)); _MALLOC_FAILED_
    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int)); _MALLOC_FAILED_
    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int)); _MALLOC_FAILED_

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice); _MALLOC_FAILED_
    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice); _MALLOC_FAILED_

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError(); _KERNAL_FAILED_
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize(); _SYNC_FAILED_

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost); _MALLOC_FAILED_

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}



