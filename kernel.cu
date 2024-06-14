
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
#define _CAST_(P) thrust::raw_pointer_cast(P.data())
#define _RAW_CAST_(P,Q,R,S) _CAST_(P) ,_CAST_(Q) , _CAST_(R) , _CAST_(S)  

__global__
void calculate_min_dist(
    Bucket* buckets, Point_index* indexes, Point* points, double* min_distances,
    Point target, double beta2, int bucket_count, int imax, 
    int i0, int j0, int k0);

double custom_hash_map_implementation(const Points& points, const Point& target, double map_size,double beta) 
{
    Space_map2 space_map(/* with input points as */ points, /* map_size as */ map_size);
    space_map.generate_cuda_hashmap();
    auto pi = Point_index(target, map_size);
    auto beta2 = beta * beta;
    auto bucket_count = space_map.buckets.size();
    int max_index = round(beta / map_size);
    max_index = max_index + max_index % 2;
    //int n = 8 * max_index * max_index * max_index;
    //int threads_per_block = 256;
    //int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    int n = 2 * max_index;
    int threads_per_block_x = 4;
    int blocks_per_grid_x = (n + threads_per_block_x - 1) / threads_per_block_x;

    dim3 threads_per_block(threads_per_block_x, threads_per_block_x, threads_per_block_x);
    dim3 blocks_per_grid(blocks_per_grid_x, blocks_per_grid_x, blocks_per_grid_x);

    thrust::device_vector<Bucket> buckets(space_map.buckets);
    thrust::device_vector<Point_index> point_indexes(space_map.point_indexes);
    thrust::device_vector<Point> Dpoints(points);
    thrust::device_vector<double> min_distances(n*n*n);

    calculate_min_dist << <blocks_per_grid, threads_per_block >> > (
        _RAW_CAST_(buckets, point_indexes, Dpoints, min_distances),
        target, beta2, bucket_count, 2 * max_index,
        pi.x - max_index, pi.y - max_index, pi.z - max_index
        );

    return thrust::reduce(_ITER_(min_distances), beta2, min_dist());

}

__device__
size_t hash_of_point_index(int i, int j, int k) {
    return (i * 18397) + (j * 20483) + (k * 29303);
}
#define _DIS_(P,Q) (P.x-Q.x)*(P.x-Q.x) + (P.y-Q.y)*(P.y-Q.y) + (P.z-Q.z)*(P.z-Q.z) 
__global__ 
void calculate_min_dist(
    Bucket* buckets, Point_index* indexes, Point* points, double* min_distances,
    Point target, double beta2, int bucket_count, int n,
    int i0,int j0,int k0)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i >= n || j >= n || k >= n)
        return;

    //int temp = threadIdx.x / imax;
    //int i = threadIdx.x % imax + i0;
    //int j = temp % jmax + j0;
    //int k = temp / jmax + k0;

    int bucket_i = hash_of_point_index(i0 + i, j0 + j, k0 + k) % bucket_count;

    int first = buckets[bucket_i].first;
    int count = buckets[bucket_i].count;
     
    double min_distance, dist;
    min_distance = beta2;
    for (size_t i = first; i < count; i++)
    {
        if (indexes[i].x == i && indexes[i].y == j && indexes[i].z == k)
            dist = _DIS_(points[indexes[i].index], target);
            min_distance = (min_distance < dist) ? min_distance : dist;
    };

    min_distances[i + j*n + k*n*n] = min_distance;
}

int main()
{
    int n = 5;
    int p = 5 / 3;
    cout << 5.0 / 3.0<<endl;
    cout << p << endl;
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



