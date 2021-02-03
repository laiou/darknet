int gpu_index = 0;

#ifdef GPU

#include "cuda.h"
#include "utils.h"
#include "blas.h"
#include <assert.h>
#include <stdlib.h>
#include <time.h>
//设置cuda设备
void cuda_set_device(int n)
{
    gpu_index = n;
    //cudaSetDevice()必须要在使用 __global__ 的函数或者Runtime的其他的API调用之前才能生效
    //没有使用则默认使用device 0作为默认设备
    //用来设置代码在哪个设备上运行，__host__表示这个函数在主机上运行，仅可通过主机调用
    //__host__ ​cudaError_t cudaSetDevice ( int  device )，实际上是一个主机代码
    //cudaError_t是一个表示cuda错误类型的类。。。判断调用的返回值确实是否设置成功
    cudaError_t status = cudaSetDevice(n);
    //判断是否设置成功，具体实现参考src/cuda.c
    check_error(status);
}
//获得当前设备的编号
int cuda_get_device()
{
    int n = 0;
    //cudaGetDevice获得当前正在使用的设备的编号
    //__host__ ​ __device__ ​cudaError_t cudaGetDevice ( int* device )
    cudaError_t status = cudaGetDevice(&n);
    check_error(status);
    return n;
}
//判断cudaError_t中的返回值
void check_error(cudaError_t status)
{
    //cudaDeviceSynchronize();
    //__host__ ​ __device__ ​cudaError_t cudaGetLastError ( void )
    //返回运行时调用中的最后一个错误
    cudaError_t status2 = cudaGetLastError();
    //如果失败的话。。。。
    if (status != cudaSuccess)
    {   
        //__host__ ​ __device__ ​const char* cudaGetErrorString ( cudaError_t error )
        //返回错误码的描述字符串，也就是上面的cudaError_t中返回的实际上是相应的错误代码或者说是id..
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error: %s", s);
        error(buffer);
    } 
    if (status2 != cudaSuccess)
    {   
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error Prev: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error Prev: %s", s);
        error(buffer);
    } 
}

dim3 cuda_gridsize(size_t n){
    size_t k = (n-1) / BLOCK + 1;
    size_t x = k;
    size_t y = 1;
    if(x > 65535){
        x = ceil(sqrt(k));
        y = (n-1)/(x*BLOCK) + 1;
    }
    dim3 d = {x, y, 1};
    //printf("%ld %ld %ld %ld\n", n, x, y, x*y*BLOCK);
    return d;
}

#ifdef CUDNN
cudnnHandle_t cudnn_handle()
{
    static int init[16] = {0};
    static cudnnHandle_t handle[16];
    int i = cuda_get_device();
    if(!init[i]) {
        cudnnCreate(&handle[i]);
        init[i] = 1;
    }
    return handle[i];
}
#endif

//初始化cudaBLAS的句柄
cublasHandle_t blas_handle()
{
    static int init[16] = {0};
    static cublasHandle_t handle[16];
    //cuda_get_device的具体实现参考src/cuda.c
    int i = cuda_get_device();
    if(!init[i]) {
        //cublasCreate用来初始化cuBLAS库的上下文句柄，初始化的句柄会传递给后续库函数使用
        //使用完毕调用cublasDestroy()销毁句柄
        cublasCreate(&handle[i]);
        init[i] = 1;
    }
    return handle[i];
}
//在gpu上分配内存
float *cuda_make_array(float *x, size_t n)
{
    //声明一个指针
    float *x_gpu;
    //相应分配的内存数量
    size_t size = sizeof(float)*n;
    //__host__ ​ __device__ ​cudaError_t cudaMalloc ( void** devPtr, size_t size )
    //cudaMalloc在设备上分配内存，devPtr指向已分配设备内存的指针，size表示请求分配的内存大小
    cudaError_t status = cudaMalloc((void **)&x_gpu, size);
    //检查内存分配是否正确
    check_error(status);
    //如果x本身存在数值的话
    if(x){
        //将相应的数值从主机复制到gpu中
        status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
        check_error(status);
    } else {
        //如果x本身没有值，或者说是0的话
        //将x_gpu指向的位置用0初始化
        fill_gpu(n, 0, x_gpu, 1);
    }
    if(!x_gpu) error("Cuda malloc failed\n");
    //返回分配的gpu内存指针
    return x_gpu;
}
//cuda_random(layer.rand_gpu, layer.batch*8);
void cuda_random(float *x_gpu, size_t n)
{
    //一般使用curandGenerator_t生成随机数指针
    static curandGenerator_t gen[16];
    static int init[16] = {0};
    //获得当前运行的设备编号
    int i = cuda_get_device();
    if(!init[i]){
        //curandCreateGenerator用来创建新的随机数生成器
        //curandStatus_t CURANDAPI curandCreateGenerator ( curandGenerator_t* generator, curandRngType_t rng_type )
        //generator是指针生成器，curandRngType_t是要创建的生成器类型
        //具体参考https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1g56ff2b3cf7e28849f73a1e22022bcbfd
        //CURAND_RNG_PSEUDO_DEFAULT表示随机数的offset和seed都是0
        curandCreateGenerator(&gen[i], CURAND_RNG_PSEUDO_DEFAULT);
        //curandSetPseudoRandomGeneratorSeed用来设置上面产生的生成器的相关选项
        //这里是将对应的gen[i]的生成器的seed设置成当前系统时间
        curandSetPseudoRandomGeneratorSeed(gen[i], time(0));
        init[i] = 1;
    }
    //curandGenerateUniform用来生成0.0到1.0之间均匀分布的浮点值，其中0.0被排除，1.0被包含
    curandGenerateUniform(gen[i], x_gpu, n);
    check_error(cudaPeekAtLastError());
}

float cuda_compare(float *x_gpu, float *x, size_t n, char *s)
{
    float *tmp = calloc(n, sizeof(float));
    cuda_pull_array(x_gpu, tmp, n);
    //int i;
    //for(i = 0; i < n; ++i) printf("%f %f\n", tmp[i], x[i]);
    axpy_cpu(n, -1, x, 1, tmp, 1);
    float err = dot_cpu(n, tmp, 1, tmp, 1);
    printf("Error %s: %f\n", s, sqrt(err/n));
    free(tmp);
    return err;
}

int *cuda_make_int_array(int *x, size_t n)
{
    int *x_gpu;
    size_t size = sizeof(int)*n;
    cudaError_t status = cudaMalloc((void **)&x_gpu, size);
    check_error(status);
    if(x){
        status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
        check_error(status);
    }
    if(!x_gpu) error("Cuda malloc failed\n");
    return x_gpu;
}

void cuda_free(float *x_gpu)
{
    cudaError_t status = cudaFree(x_gpu);
    check_error(status);
}
//cuda_push_array(l.weights_gpu, l.weights, l.nweights);
//将一个权重矩阵推送到gpu设备上
void cuda_push_array(float *x_gpu, float *x, size_t n)
{
    //通过sizeof来计算n个浮点数占用的内存空间
    size_t size = sizeof(float)*n;
    //通过cudaMemcpy将相应数值从主机复制到设备上
    //__host__ ​cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind )
    //dst表示目的内存地址，src表示源内存地址，count复制的字节数
    cudaError_t status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
    //检查是否复制成功，check_error的具体实现参考src/cuda.c
    check_error(status);
}

//cuda_pull_array(l.output_gpu, l.output, l.outputs*l.batch);
//将一个矩阵从gpu设备上拉取到本地主机
void cuda_pull_array(float *x_gpu, float *x, size_t n)
{
    size_t size = sizeof(float)*n;
    //注意一下这里的细微差别就行了。。和上面的push刚好是相反的过程
    cudaError_t status = cudaMemcpy(x, x_gpu, size, cudaMemcpyDeviceToHost);
    check_error(status);
}

float cuda_mag_array(float *x_gpu, size_t n)
{
    float *temp = calloc(n, sizeof(float));
    cuda_pull_array(x_gpu, temp, n);
    float m = mag_array(temp, n);
    free(temp);
    return m;
}
#else
void cuda_set_device(int n){}

#endif
