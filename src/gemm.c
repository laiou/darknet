#include "gemm.h"
#include "utils.h"
#include "cuda.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

void gemm_bin(int M, int N, int K, float ALPHA, 
        char  *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            char A_PART = A[i*lda+k];
            if(A_PART){
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] += B[k*ldb+j];
                }
            } else {
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] -= B[k*ldb+j];
                }
            }
        }
    }
}

float *random_matrix(int rows, int cols)
{
    int i;
    float *m = calloc(rows*cols, sizeof(float));
    for(i = 0; i < rows*cols; ++i){
        m[i] = (float)rand()/RAND_MAX;
    }
    return m;
}

void time_random_matrix(int TA, int TB, int m, int k, int n)
{
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i<10; ++i){
        gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf ms\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}

//gemm(0,0,m,n,k,1,a,k,b,n,1,c,n)
//m表示卷积核的个数，n表示输出特征图的w*h,k表示一次卷积的参数量，也可以看成一个卷积核的参数量
//a指向当前卷积权重的其实位置，b指向存储展开图片数据的起始位置
//c指向存储输出特征图的起始位置
void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    //gemm_cpu具体实现参考src/gemm.c
    gemm_cpu( TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);
}
//gemm_nn(m,n,k,1,a,k,b,n,c,n)
//m表示卷积核的个数，n表示输出特征图的w*h,k表示一次卷积的参数量，也可以看成一个卷积核的参数量
//a指向当前卷积权重的其实位置，b指向存储展开图片数据的起始位置
//c指向存储输出特征图的起始位置
//实现具体的矩阵相乘
void gemm_nn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    //#pragma omp parallel for是OpenMP中的一个指令，表示接下来的for循环将被多线程执行，另外每次循环之间不能有关系。
    #pragma omp parallel for
    //计算两个矩阵的点积，并将结果写入对应位置
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            //register关键字请求编译器尽可能的将变量存在CPU内部寄存器中，而不是通过内存寻址访问，以提高效率
            //取出相应的权重
            register float A_PART = ALPHA*A[i*lda+k];
            //取出相应的图片值进行乘积，加到原来的数据上
            //这里的逻辑并不是理论上卷积计算一次计算完一个点的结果，而是根据对应的权重值，一次完成一个权重值在相应特征图上的全部计算
            //可以理解成一次完成了整个output特征图上的全部点的一次更新，只不过这些更新都还不是最终结果，整个大循环的最后一轮更新才是
            //所有点一次完成更新
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}
// gemm_nt(m,n,k,1,a,k,b,k,c,n)
//这里的m表示当前层卷积核的数量
//n代表l.size*l.size*l.c,k表示out_w*out_h,a表示当前层的l.delta(与当前处理的图片对应)，b表示工作空间，里面存储了输入特征图展开后的矩阵
//c表示weights_update的值
void gemm_nt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    //#pragma omp parallel for是OpenMP中的一个指令，表示接下来的for循环将被多线程执行，另外每次循环之间不能有关系。
    #pragma omp parallel for
    //计算两个矩阵的点积，将结果写入相应位置
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
             //register关键字请求编译器尽可能的将变量存在CPU内部寄存器中，而不是通过内存寻址访问，以提高效率
            register float sum = 0;
            //实现的是l.delta和b中的输入的特征图展开矩阵的转置的乘积
            //具体逻辑参考gemm_nn的逻辑
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            //注意这里也是累加到了原来的数据上，并未重写，从而完成一个batch_size数据weights_update的累积
            C[i*ldc+j] += sum;
        }
    }
}
//gemm(1,0,n,k,m,1,a,n,b,k,0,c,k)
//gemm(1,0,m,n,k,1,a,m,b,n,1,c,n);m--outputs，n--inputs，k--batch,a--l.delta,b---input
//gemm_cpu(m,n,k,1,a,m,b,n,c,n);
void gemm_tn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    //M表示所有卷积核在一个channel上的参数量，lda的值是M，ldb的值是N
    for(i = 0; i < M; ++i){
        //一张图的不同通道
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[k*lda+i];
            //一张图的某一个通道上的值
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_tt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}
// gemm(0,1,m,n,k,1,a,k,b,k,1,c,n)
//gemm_cpu(0,0,m,n,k,1,a,k,b,n,1,c,n)
//gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);
//m表示卷积核的个数，n表示输出特征图的w*h,k表示一次卷积的参数量，也可以看成一个卷积核的参数量
//a指向当前卷积权重的其实位置，b指向存储展开图片数据的起始位置
//c指向存储输出特征图的起始位置
//gemm_cpu(1,0,m,n,k,1,a,m,b,n,0,c,n);
void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    //printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
    int i, j;
    //对output存储的部分进行一个缩放。。。实际上这里的BETA是1...
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }
    //然后这里就是具体的矩阵乘积操作了
    if(!TA && !TB)
        //gemm_nn的具体实现参考src/gemm.c
        //gemm_nn(m,n,k,1,a,k,b,n,c,n)
        gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(TA && !TB)
        //gemm(1,0,n,k,m,1,a,n,b,k,0,c,k)
        //这里需要注意一下，就是这里的BETA是0，在上面先将c中的数据清0了
        gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        //gemm_nt的具体实现参考src/gemm.c
        // gemm(m,n,k,1,a,k,b,k,c,n)
        gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
}

#ifdef GPU

#include <math.h>
//gemm_gpu(0,1,m,n,k,1,a,k,b,k,1,c,n);
//gemm函数的gpu版本
void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A_gpu, int lda, 
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc)
{
    //cublasHandle_t指向包含cuBLAS库上下文的不透明结构的指针类型
    //具体参考https://docs.nvidia.com/cuda/cublas/index.html
    //blas_handle的具体实现参考src/cuda.c
    //这里的操作是进行cuBLAS上下文的初始化，返回相应的上下文句柄
    cublasHandle_t handle = blas_handle();
    //cublasSgemm是一个cuda运行时api，主要实现的是矩阵计算功能
    //具体参考https://docs.nvidia.com/cuda/cublas/index.html#cublasxt_gemm
    //以及http://www.netlib.org/blas/sgemm.f和https://blog.csdn.net/yutianzuijin/article/details/90411622
    cudaError_t status = cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N), 
            (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc);
    check_error(status);
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void time_gpu_random_matrix(int TA, int TB, int m, int k, int n)
{
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i<32; ++i){
        gemm_gpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}

void time_gpu(int TA, int TB, int m, int k, int n)
{
    int iter = 10;
    float *a = random_matrix(m,k);
    float *b = random_matrix(k,n);

    int lda = (!TA)?k:m;
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);

    float *a_cl = cuda_make_array(a, m*k);
    float *b_cl = cuda_make_array(b, k*n);
    float *c_cl = cuda_make_array(c, m*n);

    int i;
    clock_t start = clock(), end;
    for(i = 0; i<iter; ++i){
        gemm_gpu(TA,TB,m,n,k,1,a_cl,lda,b_cl,ldb,1,c_cl,n);
        cudaThreadSynchronize();
    }
    double flop = ((double)m)*n*(2.*k + 2.)*iter;
    double gflop = flop/pow(10., 9);
    end = clock();
    double seconds = sec(end-start);
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s, %lf GFLOPS\n",m,k,k,n, TA, TB, seconds, gflop/seconds);
    cuda_free(a_cl);
    cuda_free(b_cl);
    cuda_free(c_cl);
    free(a);
    free(b);
    free(c);
}


void test_gpu_accuracy(int TA, int TB, int m, int k, int n)
{
    srand(0);
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    float *c_gpu = random_matrix(m,n);
    memset(c, 0, m*n*sizeof(float));
    memset(c_gpu, 0, m*n*sizeof(float));
    int i;
    //pm(m,k,b);
    gemm_gpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c_gpu,n);
    //printf("GPU\n");
    //pm(m, n, c_gpu);

    gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    //printf("\n\nCPU\n");
    //pm(m, n, c);
    double sse = 0;
    for(i = 0; i < m*n; ++i) {
        //printf("%f %f\n", c[i], c_gpu[i]);
        sse += pow(c[i]-c_gpu[i], 2);
    }
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %g SSE\n",m,k,k,n, TA, TB, sse/(m*n));
    free(a);
    free(b);
    free(c);
    free(c_gpu);
}

int test_gpu_blas()
{
    /*
       test_gpu_accuracy(0,0,10,576,75); 

       test_gpu_accuracy(0,0,17,10,10); 
       test_gpu_accuracy(1,0,17,10,10); 
       test_gpu_accuracy(0,1,17,10,10); 
       test_gpu_accuracy(1,1,17,10,10); 

       test_gpu_accuracy(0,0,1000,10,100); 
       test_gpu_accuracy(1,0,1000,10,100); 
       test_gpu_accuracy(0,1,1000,10,100); 
       test_gpu_accuracy(1,1,1000,10,100); 

       test_gpu_accuracy(0,0,10,10,10); 

       time_gpu(0,0,64,2916,363); 
       time_gpu(0,0,64,2916,363); 
       time_gpu(0,0,64,2916,363); 
       time_gpu(0,0,192,729,1600); 
       time_gpu(0,0,384,196,1728); 
       time_gpu(0,0,256,196,3456); 
       time_gpu(0,0,256,196,2304); 
       time_gpu(0,0,128,4096,12544); 
       time_gpu(0,0,128,4096,4096); 
     */
    time_gpu(0,0,64,75,12544); 
    time_gpu(0,0,64,75,12544); 
    time_gpu(0,0,64,75,12544); 
    time_gpu(0,0,64,576,12544); 
    time_gpu(0,0,256,2304,784); 
    time_gpu(1,1,2304,256,784); 
    time_gpu(0,0,512,4608,196); 
    time_gpu(1,1,4608,512,196); 

    return 0;
}
#endif

