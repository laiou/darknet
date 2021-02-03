#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include <assert.h>

extern "C" {
#include "blas.h"
#include "cuda.h"
#include "utils.h"
}
//scale_bias_kernel<<<dimGrid, dimBlock>>>(output, biases, n, size);
//实现的是将output中的值和biases中的值相乘，结果写入output中
__global__ void scale_bias_kernel(float *output, float *biases, int n, int size)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int filter = blockIdx.y;
    int batch = blockIdx.z;

    if(offset < size) output[(batch*n+filter)*size + offset] *= biases[filter];
}

//scale_bias_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);
void scale_bias_gpu(float *output, float *biases, int batch, int n, int size)
{
    //用两个dim3类型的变量声明grid的维度和block的维度
    //注意这里的(size-1)/BLOCK+1,是为了保证取到一个整数，同时保证整个grid中的线程数量大于或者等于一个batch中要处理的数据总量
    dim3 dimGrid((size-1)/BLOCK + 1, n, batch);
    dim3 dimBlock(BLOCK, 1, 1);

    //scale_bias_kernel的具体实现参考src/blas_kernels.cu
    //将output中的值和biases中的值相乘，结果写入output中
    scale_bias_kernel<<<dimGrid, dimBlock>>>(output, biases, n, size);
    check_error(cudaPeekAtLastError());
}

//backward_scale_kernel<<<n, BLOCK>>>(x_norm, delta, batch, n, size, scale_updates);
//计算相应的scale的更新值
__global__ void backward_scale_kernel(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates)
{
    //声明一个共享变量
    __shared__ float part[BLOCK];
    int i,b;
    int filter = blockIdx.x;
    int p = threadIdx.x;
    float sum = 0;
    //遍历batch中每张图片的处理数据
    for(b = 0; b < batch; ++b){
    //遍历某个通道下全部位置上的每个数据
        for(i = 0; i < size; i += BLOCK){
        //计算相应的索引
            int index = p + i + size*(filter + n*b);
            //将相应位置上的delta和x_norm中的值相乘，结果累加到sum
            sum += (p+i < size) ? delta[index]*x_norm[index] : 0;
        }
    }
    //这里是完成了整个batch中某一个通道上全部位置上相应乘加操作的累计
    part[p] = sum;
    //线程的同步
    __syncthreads();
    //id==0的线程完成数据更新
    if (p == 0) {
    //将上面part中的数据累加到scale_update里面。。作为scale的更新值
        for(i = 0; i < BLOCK; ++i) scale_updates[filter] += part[i];
    }
}

//backward_scale_gpu(l.x_norm_gpu, l.delta_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.scale_updates_gpu);
//计算相应的scale的更新值
void backward_scale_gpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates)
{
    //backward_scale_kernel的具体实现参考src/blas_kernels.cu
    backward_scale_kernel<<<n, BLOCK>>>(x_norm, delta, batch, n, size, scale_updates);
    check_error(cudaPeekAtLastError());
}

//add_bias_kernel<<<cuda_gridsize(num), BLOCK>>>(output, biases, batch, n, size);
//实现的是将bias中的值加到output中去
__global__ void add_bias_kernel(float *output, float *biases, int batch, int n, int size)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= n*size*batch) return;
    int i = index % size;
    index /= size;
    int j = index % n;
    index /= n;
    int k = index;

    output[(k*n+j)*size + i] += biases[j];
}

//add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.out_c, l.out_w*l.out_h);
//实现的是将l.biases_gpu中的值加到l.output_gpu中去
void add_bias_gpu(float *output, float *biases, int batch, int n, int size)
{
    //计算参与计算的数据总量
    int num = n*size*batch;
    //add_bias_kernel的具体实现参考src/blas_kernels.cu
    add_bias_kernel<<<cuda_gridsize(num), BLOCK>>>(output, biases, batch, n, size);
    check_error(cudaPeekAtLastError());
}

__global__ void backward_bias_conn_kernel(float *bias_updates, float *delta, int batch, int n)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= n) return;
    int b;
    float sum = 0;
    for(b = 0; b < batch; ++b){
        int i = b*n + index;
        sum += delta[i];
    }
    bias_updates[index] += sum;
}


// backward_bias_kernel<<<n, BLOCK>>>(bias_updates, delta, batch, n, size);
//这里的n表示输出通道数out-c
//计算相应的偏置更新值，bias_update
__global__ void backward_bias_kernel(float *bias_updates, float *delta, int batch, int n, int size)
{
    //声明一个共享变量
    __shared__ float part[BLOCK];
    int i,b;
    int filter = blockIdx.x;
    int p = threadIdx.x;
    float sum = 0;
    //遍历batch中每一个图片产生的数据
    for(b = 0; b < batch; ++b){
    //遍历单个通道上每个位置的数据
        for(i = 0; i < size; i += BLOCK){
        //计算相应的数据索引
            int index = p + i + size*(filter + n*b);
            //将相应位置上的l.delta_gpu累加到sun中去
            sum += (p+i < size) ? delta[index] : 0;
        }
    }
    //整个batch上某个通道下的delta全部累加完毕。。赋值到part中
    part[p] = sum;
    //线程同步。。到这里的时候。。part中存储了一个batch中全部图片数据在每个通道上的梯度的累加值
    //比如说part[0]里面就是batch中全部图片在第0个通道上的梯度的累加
    __syncthreads();
    //用id==0的线程完成最后的提取操作
    //将上述part中的delta累加到bias_updates中，作为对相应偏置的更新值
    if (p == 0) {
        for(i = 0; i < BLOCK; ++i) bias_updates[filter] += part[i];
    }
}

//backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.out_c, l.out_w*l.out_h);
//计算偏置更新值，bias_update
void backward_bias_gpu(float *bias_updates, float *delta, int batch, int n, int size)
{
    if(size == 1){
    //如果当前层每个通道上只有一个数据
        backward_bias_conn_kernel<<<cuda_gridsize(n), BLOCK>>>(bias_updates, delta, batch, n);
    }else{
    //backward_bias_kernel的具体实现参考src/blas_kernels.cu
    //计算偏置更新
        backward_bias_kernel<<<n, BLOCK>>>(bias_updates, delta, batch, n, size);
    }
    check_error(cudaPeekAtLastError());
}

/*
__global__ void dot_kernel(float *output, float scale, int batch, int n, int size, float *delta)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    int f1 = index / n;
    int f2 = index % n;
    if (f2 <= f1) return;
    
    float sum = 0;
    float norm1 = 0;
    float norm2 = 0;
    int b, i;
    for(b = 0; b <  batch; ++b){
        for(i = 0; i < size; ++i){
            int i1 = b * size * n + f1 * size + i;
            int i2 = b * size * n + f2 * size + i;
            sum += output[i1] * output[i2];
            norm1 += output[i1] * output[i1];
            norm2 += output[i2] * output[i2];
        }
    }
    norm1 = sqrt(norm1);
    norm2 = sqrt(norm2);
    float norm = norm1 * norm2;
    sum = sum / norm;
    for(b = 0; b <  batch; ++b){
        for(i = 0; i < size; ++i){
            int i1 = b * size * n + f1 * size + i;
            int i2 = b * size * n + f2 * size + i;
            delta[i1] += - scale * sum * output[i2] / norm;
            delta[i2] += - scale * sum * output[i1] / norm;
        }
    }
}

void dot_error_gpu(layer l)
{
    dot_kernel<<<cuda_gridsize(l.n*l.n), BLOCK>>>(l.output_gpu, l.dot, l.batch, l.n, l.out_w * l.out_h, l.delta_gpu);
    check_error(cudaPeekAtLastError());
}
*/


__global__ void adam_kernel(int N, float *x, float *m, float *v, float B1, float B2, float rate, float eps, int t)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N) return;

    float mhat = m[index] / (1.f - powf(B1, t));
    float vhat = v[index] / (1.f - powf(B2, t));
    
    x[index] = x[index] + rate * mhat / (sqrtf(vhat) + eps);
}

extern "C" void adam_gpu(int n, float *x, float *m, float *v, float B1, float B2, float rate, float eps, int t)
{
    adam_kernel<<<cuda_gridsize(n), BLOCK>>>(n, x, m, v, B1, B2, rate, eps, t);
    check_error(cudaPeekAtLastError());
}

//根据adam算法进行参数更新
extern "C" void adam_update_gpu(float *w, float *d, float *m, float *v, float B1, float B2, float eps, float decay, float rate, int n, int batch, int t)
{
    scal_gpu(n, B1, m, 1);
    scal_gpu(n, B2, v, 1);
    axpy_gpu(n, -decay*batch, w, 1, d, 1);

    axpy_gpu(n, (1-B1), d, 1, m, 1);
    mul_gpu(n, d, 1, d, 1);
    axpy_gpu(n, (1-B2), d, 1, v, 1);

    adam_gpu(n, w, m, v, B1, B2, rate, eps, t);
    fill_gpu(n, 0, d, 1);
}

//normalize_kernel<<<cuda_gridsize(N), BLOCK>>>(N, x, mean, variance, batch, filters, spatial);
//根据相应的均值和方差做归一化
__global__ void normalize_kernel(int N, float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    //计算相应的线程id
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N) return;
    //计算相应的均值和方差的索引，表示的是当前计算的的是哪一个通道上的数据的归一化
    //具体可以结合相应的调用中的grid的划分的来推理
    int f = (index/spatial)%filters;
    //根据均值和方差对相应数据做归一化
    x[index] = (x[index] - mean[f])/(sqrtf(variance[f] + .00001f));
}

//normalize_delta_kernel<<<cuda_gridsize(N), BLOCK>>>(N, x, mean, variance, mean_delta, variance_delta, batch, filters, spatial, delta);
__global__ void normalize_delta_kernel(int N, float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta)
{   
    //计算相应的线程id
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N) return;
    int f = (index/spatial)%filters;
    //计算相应的dleta。。。结合BN反向传播的公式来看。。
    delta[index] = delta[index] * 1.f/(sqrtf(variance[f] + .00001f)) + variance_delta[f] * 2.f * (x[index] - mean[f]) / (spatial * batch) + mean_delta[f]/(spatial*batch);
}

// normalize_delta_gpu(l.x_gpu, l.mean_gpu, l.variance_gpu, l.mean_delta_gpu, l.variance_delta_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.delta_gpu);
//完成当前的BN层的l.x_gpu的delta的计算，结果写入l.delta_gpu
extern "C" void normalize_delta_gpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta)
{
    size_t N = batch*filters*spatial;
    //normalize_delta_kernel的具体实现参考src/blas_kernels.cu
    normalize_delta_kernel<<<cuda_gridsize(N), BLOCK>>>(N, x, mean, variance, mean_delta, variance_delta, batch, filters, spatial, delta);
    check_error(cudaPeekAtLastError());
}

__global__ void  variance_delta_kernel(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= filters) return;
    int j,k;
    variance_delta[i] = 0;
    for(j = 0; j < batch; ++j){
        for(k = 0; k < spatial; ++k){
            int index = j*filters*spatial + i*spatial + k;
            variance_delta[i] += delta[index]*(x[index] - mean[i]);
        }
    }
    variance_delta[i] *= -.5f * powf(variance[i] + .00001f, (float)(-3.f/2.f));
}

__global__ void accumulate_kernel(float *x, int n, int groups, float *sum)
{
    int k;
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= groups) return;
    sum[i] = 0;
    for(k = 0; k < n; ++k){
        sum[i] += x[k*groups + i];
    }
}

//fast_mean_delta_kernel<<<filters, BLOCK>>>(delta, variance, batch, filters, spatial, mean_delta);
//计算当前BN层均值的delta
__global__ void fast_mean_delta_kernel(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
{
    const int threads = BLOCK;
    //声明一个共享变量
    __shared__ float local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int filter = blockIdx.x;

    int i, j;
    //遍历batch中每个图像产生的相应数据
    for(j = 0; j < batch; ++j){
    //遍历某个通道上的全部位置上的数据
        for(i = 0; i < spatial; i += threads){
        //计算相应的索引
            int index = j*spatial*filters + filter*spatial + i + id;
            //将相应位置上的l.delta_gpu值累加到local中
            local[id] += (i+id < spatial) ? delta[index] : 0;
        }
    }

    //线程间的同步操作
    __syncthreads();

    //id==0的线程独自完成后续计算
    if(id == 0){
        mean_delta[filter] = 0;
        for(i = 0; i < threads; ++i){
        //将local中的数据累加到mean_delta上，实际上就是累加到l.mean_delta_gpu上
            mean_delta[filter] += local[i];
        }
        //然后计算当前层次均值的梯度，具体参考BN反向传播的公式
        mean_delta[filter] *= (-1.f/sqrtf(variance[filter] + .00001f));
    }
}

//fast_variance_delta_kernel<<<filters, BLOCK>>>(x, delta, mean, variance, batch, filters, spatial, variance_delta);
//计算当前BN层方差的delta
__global__ void  fast_variance_delta_kernel(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta)
{
    const int threads = BLOCK;
    //声明一个共享变量
    __shared__ float local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int filter = blockIdx.x;

    int i, j;
    //遍历batch中每张图片产生的数据
    for(j = 0; j < batch; ++j){
    //遍历某个通道上全部位置的数据
        for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + filter*spatial + i + id;
            //累加计算相应的值
            //结合后面的实现，参考BN层反向传播的推导
            local[id] += (i+id < spatial) ? delta[index]*(x[index] - mean[filter]) : 0;
        }
    }
    //线程同步
    __syncthreads();
    //接着就是完成从共享变量中的数值提取
    //和方差梯度计算的最后一步了
    if(id == 0){
        variance_delta[filter] = 0;
        for(i = 0; i < threads; ++i){
            variance_delta[filter] += local[i];
        }
        variance_delta[filter] *= -.5f * powf(variance[filter] + .00001f, (float)(-3.f/2.f));
    }
}


__global__ void mean_delta_kernel(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= filters) return;
    int j,k;
    mean_delta[i] = 0;
    for (j = 0; j < batch; ++j) {
        for (k = 0; k < spatial; ++k) {
            int index = j*filters*spatial + i*spatial + k;
            mean_delta[i] += delta[index];
        }
    }
    mean_delta[i] *= (-1.f/sqrtf(variance[i] + .00001f));
}

extern "C" void mean_delta_gpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
{
    mean_delta_kernel<<<cuda_gridsize(filters), BLOCK>>>(delta, variance, batch, filters, spatial, mean_delta);
    check_error(cudaPeekAtLastError());
}

//fast_mean_delta_gpu(l.delta_gpu, l.variance_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.mean_delta_gpu);
//计算当前层均值的delta
extern "C" void fast_mean_delta_gpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
{
    //fast_mean_delta_kernel的具体实现参考src/blas_kernels.cu
    
    fast_mean_delta_kernel<<<filters, BLOCK>>>(delta, variance, batch, filters, spatial, mean_delta);
    check_error(cudaPeekAtLastError());
}

//fast_variance_delta_gpu(l.x_gpu, l.delta_gpu, l.mean_gpu, l.variance_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.variance_delta_gpu);
//计算当前BN层方差的delta
extern "C" void fast_variance_delta_gpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta)
{   
    //fast_variance_delta_kernel的具体实现参考src/blas_kernels.cu
    fast_variance_delta_kernel<<<filters, BLOCK>>>(x, delta, mean, variance, batch, filters, spatial, variance_delta);
    check_error(cudaPeekAtLastError());
}

__global__ void  mean_kernel(float *x, int batch, int filters, int spatial, float *mean)
{
    float scale = 1.f/(batch * spatial);
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= filters) return;
    int j,k;
    mean[i] = 0;
    for(j = 0; j < batch; ++j){
        for(k = 0; k < spatial; ++k){
            int index = j*filters*spatial + i*spatial + k;
            mean[i] += x[index];
        }
    }
    mean[i] *= scale;
}

__global__ void variance_kernel(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    float scale = 1.f/(batch * spatial - 1);
    int j,k;
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= filters) return;
    variance[i] = 0;
    for(j = 0; j < batch; ++j){
        for(k = 0; k < spatial; ++k){
            int index = j*filters*spatial + i*spatial + k;
            variance[i] += powf((x[index] - mean[i]), 2);
        }
    }
    variance[i] *= scale;
}

//reorg_kernel<<<cuda_gridsize(size), BLOCK>>>(size, x, w, h, c, batch, stride, forward, out);
//实现reorg操作的核函数
__global__ void reorg_kernel(int N, float *x, int w, int h, int c, int batch, int stride, int forward, float *out)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= N) return;
    int in_index = i;
    int in_w = i%w;
    i = i/w;
    int in_h = i%h;
    i = i/h;
    int in_c = i%c;
    i = i/c;
    int b = i%batch;

    int out_c = c/(stride*stride);

    int c2 = in_c % out_c;
    int offset = in_c / out_c;
    int w2 = in_w*stride + offset % stride;
    int h2 = in_h*stride + offset / stride;
    //printf("%d\n", offset);
    int out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));

   // printf("%d %d %d\n", w2, h2, c2);
    //printf("%d %d\n", in_index, out_index);
    //if(out_index >= N || out_index < 0) printf("bad bad bad \n");

    if(forward) out[out_index] = x[in_index];
    else out[in_index] = x[out_index];
    //if(forward) out[1] = x[1];
    //else out[0] = x[0];
}

__global__ void axpy_kernel(int N, float ALPHA, float *X, int OFFX, int INCX,  float *Y, int OFFY, int INCY)
{
    //计算线程id
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    //将X中的值乘上ALPHA然后在加到Y中去
    if(i < N) Y[OFFY+i*INCY] += ALPHA*X[OFFX+i*INCX];
}

__global__ void pow_kernel(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) Y[i*INCY] = pow(X[i*INCX], ALPHA);
}

__global__ void const_kernel(int N, float ALPHA, float *X, int INCX)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) X[i*INCX] = ALPHA;
}

//constrain_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
//将X修正到-ALPHA和ALPHA之间
__global__ void constrain_kernel(int N, float ALPHA, float *X, int INCX)
{
    //计算相应的线程索引
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    //fminf返回两个浮点数中的较小值，fmaxf返回两个浮点数中的较大值
    //这里实现的是将x修正到-ALPHA和ALPHA之间
    if(i < N) X[i*INCX] = fminf(ALPHA, fmaxf(-ALPHA, X[i*INCX]));
}

//supp_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
//如果X中值的平方小于ALPHA的平方，将其置为0
__global__ void supp_kernel(int N, float ALPHA, float *X, int INCX)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) {
        if((X[i*INCX] * X[i*INCX]) < (ALPHA * ALPHA)) X[i*INCX] = 0;
    }
}

//add_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
__global__ void add_kernel(int N, float ALPHA, float *X, int INCX)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    //将ALPHA加到x中对应的值上
    if(i < N) X[i*INCX] += ALPHA;
}

__global__ void scal_kernel(int N, float ALPHA, float *X, int INCX)
{
    //计算相应的线程id
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    //完成相应的赋值操作
    if(i < N) X[i*INCX] *= ALPHA;
}

//fill_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
//fill_gpu(l.outputs * l.batch, 0, l.delta_gpu, 1)
//__golbal__将这个函数声明成内核，该函数仅在设备上执行，仅可通过主机调用
__global__ void fill_kernel(int N, float ALPHA, float *X, int INCX)
{
    //计算相应的线程索引，计算索引的过程。。参考cuda编程中grid和block以及thread之间的联系。。
    //映射到矩阵上其实很容易理解，具体可以参考https://blog.csdn.net/JackZhang_123/article/details/78020444
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    //将相应的值进行初始化。。从上面传入的参数来看，如果ALPHA是0的话，就是初始化成0
    if(i < N) X[i*INCX] = ALPHA;
}

//copy_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X, OFFX, INCX, Y, OFFY, INCY);
//这里传入的参数是(N, X, 0, INCX, Y, 0, INCY)，INCX和INCY都是1
__global__ void copy_kernel(int N,  float *X, int OFFX, int INCX, float *Y, int OFFY, int INCY)
{
    //计算相应的线程索引
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    //每个线程完成一个赋值的操作。。。
    if(i < N) Y[i*INCY + OFFY] = X[i*INCX + OFFX];
}

//mul_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X, INCX, Y, INCY);
//实现的是将x中对应的值乘到y中对应的数值上去，结果存入y中
__global__ void mul_kernel(int N, float *X, int INCX, float *Y, int INCY)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) Y[i*INCY] *= X[i*INCX];
}

//normalize_gpu(l.output_gpu, l.mean_gpu, l.variance_gpu, l.batch, l.out_c, l.out_h*l.out_w);
//根据相应的均值和方差对相应的数据做归一化
extern "C" void normalize_gpu(float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    //计算grid划分的个数是batch和输出通道数以及输出某一个通道上的参数数量的乘积
    size_t N = batch*filters*spatial;
    //normalize_kernel的具体实现参考src/blas_kernels.cu
    normalize_kernel<<<cuda_gridsize(N), BLOCK>>>(N, x, mean, variance, batch, filters, spatial);
    check_error(cudaPeekAtLastError());
}

//l2norm_kernel<<<cuda_gridsize(N), BLOCK>>>(N, x, dx, batch, filters, spatial);
//l2正则化的gpu版本实现
__global__ void l2norm_kernel(int N, float *x, float *dx, int batch, int filters, int spatial)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N) return;
    int b = index / spatial;
    int i = index % spatial;
    int f;
    float sum = 0;
    for(f = 0; f < filters; ++f){
        int index = b*filters*spatial + f*spatial + i;
        sum += powf(x[index], 2);
    }
    sum = sqrtf(sum);
    if(sum == 0) sum = 1;
    //printf("%f\n", sum);
    for(f = 0; f < filters; ++f){
        int index = b*filters*spatial + f*spatial + i;
        x[index] /= sum;
        dx[index] = (1 - x[index]) / sum;
    }
}

// l2normalize_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_w*l.out_h);
//l2正则化的gpu实现
extern "C" void l2normalize_gpu(float *x, float *dx, int batch, int filters, int spatial)
{
    size_t N = batch*spatial;
    //l2norm_kernel的具体实现参考src/blas_kernels.cu
    l2norm_kernel<<<cuda_gridsize(N), BLOCK>>>(N, x, dx, batch, filters, spatial);
    check_error(cudaPeekAtLastError());
}

//fast_mean_gpu(l.output_gpu, l.batch, l.out_c, l.out_h*l.out_w, l.mean_gpu);
//这里实现的是计算每张图片对应通道上的均值
__global__ void  fast_mean_kernel(float *x, int batch, int filters, int spatial, float *mean)
{
    //这里的BLOCK的值是512。。表示一个block中有512个线程，在example/darknet.h中有声明
    const int threads = BLOCK;
    //__shared__表示所声明的变量：位于线程块的共享存储器空间中，与块具有相同的生命周期，尽可能通过块内所有线程访问
    //使用share memory可以有效减少程序从global memory中读取数据的次数，加速程序执行
    //声明一个local的浮点数组到共享内存
    __shared__ float local[threads];
    //计算线程id
    int id = threadIdx.x;
    //初始化local中的值为0
    local[id] = 0;
    //filter是block在x方向上的索引，实际数值也就是输出的通道数。。
    //从这里能看出实际上的线程配置应该是grid中有out_c个Block，每个block中512个线程
    int filter = blockIdx.x;

    int i, j;
    //这里关于通道的遍历由不同的线程实现，也就是通过线程id实现了
    //遍历batch中每张图片的数据
    for(j = 0; j < batch; ++j){
    //遍历相应通道上每个位置的数值
        for(i = 0; i < spatial; i += threads){
        //计算相应的索引
            int index = j*spatial*filters + filter*spatial + i + id;
            //将相应位置上的值累加到local的对应位置里，这里还有一个补0的操作是因为block中的线程数量是512个。。但是实际上，每一个线程处理的数据可能不足512
            //所以有些线程只能做一个补0的操作
            local[id] += (i+id < spatial) ? x[index] : 0;
        }
    }
    // __syncthreads();进行cuda线程同步操作。。。也就是所有的相关线程将上面的代码都执行完毕才能进入下面的操作,防止某些计算没有完成就进入了下个阶段
    //导致有些计算没有完成而产生错误
    //这里需要等到所有值都累加完毕再去计算均值
    __syncthreads();
    //用id是0的线程完成最后的求均值的操作
    if(id == 0){
        mean[filter] = 0;
        //从共享变量local中将值取到mean中
        for(i = 0; i < threads; ++i){
            mean[filter] += local[i];
        }
        //求均值
        mean[filter] /= spatial * batch;
    }
}
//fast_variance_gpu(l.output_gpu, l.mean_gpu, l.batch, l.out_c, l.out_h*l.out_w, l.variance_gpu)
//跟上面求均值对应起来。。计算完均值之后在计算相应的方差
__global__ void  fast_variance_kernel(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{   
    //还是先定义一个block中的线程数量
    const int threads = BLOCK;
    //同样是声明一个共享变量
    __shared__ float local[threads];
    //计算相应的线程索引
    int id = threadIdx.x;
    //初始化local中的值
    local[id] = 0;
    //计算blocl在x方向上的索引
    int filter = blockIdx.x;

    int i, j;
    //遍历batch中每张图的处理数据
    for(j = 0; j < batch; ++j){
    //遍历某个通道上每一个位置上的值
        for(i = 0; i < spatial; i += threads){
            //计算相应的索引
            int index = j*spatial*filters + filter*spatial + i + id;
            //计算对应的数值减去均值的平方，powf(x,y):返回x的y次方
            local[id] += (i+id < spatial) ? powf((x[index] - mean[filter]), 2) : 0;
        }
    }
    //线程间的同步操作
    __syncthreads();

    if(id == 0){
        variance[filter] = 0;
        for(i = 0; i < threads; ++i){
        //从共享内存中取出相应的值
            variance[filter] += local[i];
        }
        //完成最后的方差计算
        variance[filter] /= (spatial * batch - 1);
    }
}

//计算batch中每张图在相应通道上的均值
extern "C" void fast_mean_gpu(float *x, int batch, int filters, int spatial, float *mean)
{
    //fast_mean_kernel的是实现参考src/blas_kernels.cu
    fast_mean_kernel<<<filters, BLOCK>>>(x, batch, filters, spatial, mean);
    check_error(cudaPeekAtLastError());
}
//和上面的均值计算对应，这里是结合上面的均值计算相应的方差
extern "C" void fast_variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    //fast_variance_kernel的具体实现参考src/blas_kernels.cu
    fast_variance_kernel<<<filters, BLOCK>>>(x, mean, batch, filters, spatial, variance);
    check_error(cudaPeekAtLastError());
}


extern "C" void mean_gpu(float *x, int batch, int filters, int spatial, float *mean)
{
    mean_kernel<<<cuda_gridsize(filters), BLOCK>>>(x, batch, filters, spatial, mean);
    check_error(cudaPeekAtLastError());
}

extern "C" void variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    variance_kernel<<<cuda_gridsize(filters), BLOCK>>>(x, mean, batch, filters, spatial, variance);
    check_error(cudaPeekAtLastError());
}

//axpy_gpu(l.out_c, .01, l.mean_gpu, 1, l.rolling_mean_gpu, 1);
//这里实现的是将l.mean_gpu中的值乘上ALPHA然后在加到l.rolling_mean_gpu中去
extern "C" void axpy_gpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY)
{
    //axpy_gpu_offset的具体实现参考src/blas_kernels.cu
    axpy_gpu_offset(N, ALPHA, X, 0, INCX, Y, 0, INCY);
}

extern "C" void pow_gpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY)
{
    pow_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX, Y, INCY);
    check_error(cudaPeekAtLastError());
}

//axpy_gpu_offset(N, ALPHA, X, 0, INCX, Y, 0, INCY);
extern "C" void axpy_gpu_offset(int N, float ALPHA, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY)
{
    //axpy_kernel的具体实现参考src/blas_kernels.cu
    //实现的是将X中的值乘上ALPHA在加到Y中去
    axpy_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, OFFX, INCX, Y, OFFY, INCY);
    check_error(cudaPeekAtLastError());
}

//copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
extern "C" void copy_gpu(int N, float * X, int INCX, float * Y, int INCY)
{
    //copy_gpu_offset的具体实现参考src/blas_kernels.cu
    //实现的是一个赋值操作，将net.input_gpu中的值赋值到l.output_gpu中去
    copy_gpu_offset(N, X, 0, INCX, Y, 0, INCY);
}

//mul_gpu(l.outputs*l.batch, l.r_gpu, 1, l.forgot_state_gpu, 1);
//这里实现的是将l.r_gpu中的值乘到l.forgot_state_gpu上去，结果写入l.forgot_state_gpu上
extern "C" void mul_gpu(int N, float * X, int INCX, float * Y, int INCY)
{
    //mul_kernel的具体实现参考src/blas_kernels.cu
    //将x中的数值乘到y中去，结果存入y中
    mul_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X, INCX, Y, INCY);
    check_error(cudaPeekAtLastError());
}

//copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
//copy_gpu_offset(N, X, 0, INCX, Y, 0, INCY);
extern "C" void copy_gpu_offset(int N, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY)
{
    //将这个复制的操作声明成了一个核函数。。。copy_kernel的具体实现参考src/blas_kernels.cu
    //完成一个复制的操作，将X中相应位置的值赋值到Y中相应的位置上
    copy_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X, OFFX, INCX, Y, OFFY, INCY);
    check_error(cudaPeekAtLastError());
}

//实现flatten操作的核函数
__global__ void flatten_kernel(int N, float *x, int spatial, int layers, int batch, int forward, float *out)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= N) return;
    int in_s = i%spatial;
    i = i/spatial;
    int in_c = i%layers;
    i = i/layers;
    int b = i;

    int i1 = b*layers*spatial + in_c*spatial + in_s;
    int i2 = b*layers*spatial + in_s*layers +  in_c;

    if (forward) out[i2] = x[i1];
    else out[i1] = x[i2];
}

//flatten_gpu(net.input_gpu, l.w*l.h, l.c, l.batch, 0, l.output_gpu);
//flatten的gpu版本实现
extern "C" void flatten_gpu(float *x, int spatial, int layers, int batch, int forward, float *out)
{
    int size = spatial*batch*layers;
    //flatten_kernel的具体实现参考src/blas_kernels.cu
    flatten_kernel<<<cuda_gridsize(size), BLOCK>>>(size, x, spatial, layers, batch, forward, out);
    check_error(cudaPeekAtLastError());
}

//reorg层的gpu实现
extern "C" void reorg_gpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out)
{
    int size = w*h*c*batch;
    //reorg_kernel的具体实现参考src/blas_kernels.cu
    reorg_kernel<<<cuda_gridsize(size), BLOCK>>>(size, x, w, h, c, batch, stride, forward, out);
    check_error(cudaPeekAtLastError());
}

//mask_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X, mask_num, mask, val);
__global__ void mask_kernel(int n,  float *x, float mask_num, float *mask, float val)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n && mask[i] == mask_num) x[i] = val;
}

//mask_gpu(l.batch*l.inputs, net.delta_gpu, SECRET_NUM, net.truth_gpu, 0);
extern "C" void mask_gpu(int N, float * X, float mask_num, float * mask, float val)
{
    //mask_kernel的具体实现参考src/blas_kernels.cu
    mask_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X, mask_num, mask, val);
    check_error(cudaPeekAtLastError());
}

//scale_mask_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X, mask_num, mask, scale);
__global__ void scale_mask_kernel(int n,  float *x, float mask_num, float *mask, float scale)
{
    //计算相应的线程id
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    //将x上对应的值乘上scale
    if(i < n && mask[i] == mask_num) x[i] *= scale;
}

//scale_mask_gpu(l.batch*l.inputs, l.delta_gpu, 0, net.truth_gpu, l.noobject_scale);
//将net.truth_gpu中是0的值对应位置上的l.delta_gpu乘上scale
extern "C" void scale_mask_gpu(int N, float * X, float mask_num, float * mask, float scale)
{
    //scale_mask_kernel的具体实现参考drc/blas_kernels.cu
    scale_mask_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X, mask_num, mask, scale);
    check_error(cudaPeekAtLastError());
}

extern "C" void const_gpu(int N, float ALPHA, float * X, int INCX)
{
    const_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
    check_error(cudaPeekAtLastError());
}

//constrain_gpu(l.outputs*l.batch, 1, l.delta_gpu, 1);
//将l.delta_gpu中的值修正到-1和1之间
extern "C" void constrain_gpu(int N, float ALPHA, float * X, int INCX)
{
    //constrain_kernel的具体实现参考src/blas_kernels.cu
    constrain_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
    check_error(cudaPeekAtLastError());
}

//add_gpu(l.batch*l.inputs, l.smooth * 1./l.inputs, net.truth_gpu, 1);
//将l.smooth*1./l.inputs加到net_truth_gpu上
extern "C" void add_gpu(int N, float ALPHA, float * X, int INCX)
{
    //add_kernel的具体实现参考src/blas_kernels.cu
    add_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
    check_error(cudaPeekAtLastError());
}

//scal_gpu(l.out_c, .99, l.rolling_mean_gpu, 1);
//对l.rolling_mean_gpu中每一个值都乘上0.99
extern "C" void scal_gpu(int N, float ALPHA, float * X, int INCX)
{
    //scal_kernel的具体实现参考src/blas_kernels.cu
    //实现的是将X中的每一个值都乘上ALPHA，结果写入X
    scal_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
    check_error(cudaPeekAtLastError());
}

//supp_gpu(l.batch*l.inputs, thresh, l.delta_gpu, 1);
//将l.delta_gpu相应值的平方小于thresh平方的值置为0
extern "C" void supp_gpu(int N, float ALPHA, float * X, int INCX)
{
    //supp_kernel的具体实现参考src/blas_kernels.cu
    supp_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
    check_error(cudaPeekAtLastError());
}

// fill_gpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
//这个函数实现的是将相应内存的值进行初始化
extern "C" void fill_gpu(int N, float ALPHA, float * X, int INCX)
{
    //fill_kernel的具体实现参考src/blas_kernels.cu
    //<<< >>>中表示的是执行内核函数的配置，表示执行某一个指定内核的线程数量
    //这里的<<<cuda_gridsize(N), BLOCK>>>，表示将整个内核的grid是一维，有N个block。。每一个block是一维，有BLOCK个线程,BLOCK的值是512,这个在example/darknet.h中有声明
    //(N, ALPHA, X, INCX)表示传入内核函数中的参数。。和正常的函数调用的参数一致
    fill_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
    //检查操作是否正确执行
    //__host__ ​ __device__ ​cudaError_t cudaPeekAtLastError ( void )
    //cudaPeekAtLastError返回运行时调用中的最后一个错误
    check_error(cudaPeekAtLastError());
}

//shortcut_kernel<<<cuda_gridsize(size), BLOCK>>>(size, minw, minh, minc, stride, sample, batch, w1, h1, c1, add, w2, h2, c2, s1, s2, out);
//实现shortcpu的核函数
__global__ void shortcut_kernel(int size, int minw, int minh, int minc, int stride, int sample, int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= size) return;
    int i = id % minw;
    id /= minw;
    int j = id % minh;
    id /= minh;
    int k = id % minc;
    id /= minc;
    int b = id % batch;

    int out_index = i*sample + w2*(j*sample + h2*(k + c2*b));
    int add_index = i*stride + w1*(j*stride + h1*(k + c1*b));
    out[out_index] = s1*out[out_index] + s2*add[add_index];
    //out[out_index] += add[add_index];
}

//shortcut_gpu(l.batch, l.w, l.h, l.c, net.layers[l.index].output_gpu, l.out_w, l.out_h, l.out_c, l.alpha, l.beta, l.output_gpu);
//实现shortcut层的gpu版本
extern "C" void shortcut_gpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out)
{
    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;

    int stride = w1/w2;
    int sample = w2/w1;
    assert(stride == h1/h2);
    assert(sample == h2/h1);
    if(stride < 1) stride = 1;
    if(sample < 1) sample = 1;

    int size = batch * minw * minh * minc;
    //shortcut_kernel的具体实现参考src/blas_kernels.cu
    shortcut_kernel<<<cuda_gridsize(size), BLOCK>>>(size, minw, minh, minc, stride, sample, batch, w1, h1, c1, add, w2, h2, c2, s1, s2, out);
    check_error(cudaPeekAtLastError());
}

//smooth_l1_kernel<<<cuda_gridsize(n), BLOCK>>>(n, pred, truth, delta, error);
__global__ void smooth_l1_kernel(int n, float *pred, float *truth, float *delta, float *error)
{
    //计算相应的线程id
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    //计算smoothl1 loss并更新相应的梯度
    if(i < n){
        float diff = truth[i] - pred[i];
        float abs_val = fabsf(diff);
        if(abs_val < 1) {
            error[i] = diff * diff;
            delta[i] = diff;
        }
        else {
            error[i] = 2*abs_val - 1;
            delta[i] = (diff > 0) ? 1 : -1;
        }
    }
}

//smooth_l1_gpu(l.batch*l.inputs, net.input_gpu, net.truth_gpu, l.delta_gpu, l.output_gpu);
//计算smooth l1 loss并更新相应的梯度
extern "C" void smooth_l1_gpu(int n, float *pred, float *truth, float *delta, float *error)
{
    //smooth_l1_kernel的具体实现参考src/blas_kernels.cu
    smooth_l1_kernel<<<cuda_gridsize(n), BLOCK>>>(n, pred, truth, delta, error);
    check_error(cudaPeekAtLastError());
}

//softmax_x_ent_kernel<<<cuda_gridsize(n), BLOCK>>>(n, pred, truth, delta, error);
//具体细节可以参考cpu版本的实现
__global__ void softmax_x_ent_kernel(int n, float *pred, float *truth, float *delta, float *error)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n){
        float t = truth[i];
        float p = pred[i];
        error[i] = (t) ? -log(p) : 0;
        delta[i] = t-p;
    }
}

// softmax_x_ent_gpu(l.batch*l.inputs, l.output_gpu, net.truth_gpu, l.delta_gpu, l.loss_gpu);
extern "C" void softmax_x_ent_gpu(int n, float *pred, float *truth, float *delta, float *error)
{
    //softmax_x_ent_kernel的具体实现参考src/blas_kernels.cu
    softmax_x_ent_kernel<<<cuda_gridsize(n), BLOCK>>>(n, pred, truth, delta, error);
    check_error(cudaPeekAtLastError());
}


//logistic_x_ent_kernel<<<cuda_gridsize(n), BLOCK>>>(n, pred, truth, delta, error);
//计算相应的logistic损失并且更新相关梯度
__global__ void logistic_x_ent_kernel(int n, float *pred, float *truth, float *delta, float *error)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n){
        float t = truth[i];
        float p = pred[i];
        error[i] = -t*log(p+.0000001) - (1-t)*log(1-p+.0000001);
        delta[i] = t-p;
    }
}

//logistic_x_ent_gpu(l.batch*l.inputs, l.output_gpu, net.truth_gpu, l.delta_gpu, l.loss_gpu);
//logistic的相关实现
extern "C" void logistic_x_ent_gpu(int n, float *pred, float *truth, float *delta, float *error)
{
    //logistic_x_ent_kernel的具体实现参考src/blas_kernels.cu
    logistic_x_ent_kernel<<<cuda_gridsize(n), BLOCK>>>(n, pred, truth, delta, error);
    check_error(cudaPeekAtLastError());
}

//l2_kernel<<<cuda_gridsize(n), BLOCK>>>(n, pred, truth, delta, error);
//计算l2loss，并更新梯度
__global__ void l2_kernel(int n, float *pred, float *truth, float *delta, float *error)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n){
        float diff = truth[i] - pred[i];
        error[i] = diff * diff; //I know this is technically wrong, deal with it.
        delta[i] = diff;
    }
}

//l2_gpu(l.batch*l.inputs, net.input_gpu, net.truth_gpu, l.delta_gpu, l.output_gpu);
extern "C" void l2_gpu(int n, float *pred, float *truth, float *delta, float *error)
{
    //l2_kernel的具体实现参考src/blas_kernels.cu
    l2_kernel<<<cuda_gridsize(n), BLOCK>>>(n, pred, truth, delta, error);
    check_error(cudaPeekAtLastError());
}

//l1_kernel<<<cuda_gridsize(n), BLOCK>>>(n, pred, truth, delta, error);

__global__ void l1_kernel(int n, float *pred, float *truth, float *delta, float *error)
{
    //计算l1 loss并更新相应的梯度
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n){
        float diff = truth[i] - pred[i];
        error[i] = abs(diff);
        delta[i] = (diff > 0) ? 1 : -1;
    }
}

//l1_gpu(l.batch*l.inputs, net.input_gpu, net.truth_gpu, l.delta_gpu, l.output_gpu);
//计算l1 loss，并更新相应的梯度
extern "C" void l1_gpu(int n, float *pred, float *truth, float *delta, float *error)
{
    //l1_kernel的具体实现参考src/blas_kernels.cu
    l1_kernel<<<cuda_gridsize(n), BLOCK>>>(n, pred, truth, delta, error);
    check_error(cudaPeekAtLastError());
}

// wgan_kernel<<<cuda_gridsize(n), BLOCK>>>(n, pred, truth, delta, error);
__global__ void wgan_kernel(int n, float *pred, float *truth, float *delta, float *error)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n){
        error[i] = truth[i] ? -pred[i] : pred[i];
        delta[i] = (truth[i] > 0) ? 1 : -1;
    }
}

//wgan_gpu(l.batch*l.inputs, net.input_gpu, net.truth_gpu, l.delta_gpu, l.output_gpu);
//计算相应的损失并更新梯度
extern "C" void wgan_gpu(int n, float *pred, float *truth, float *delta, float *error)
{
    //wgan_kernel的具体实现参考src/blas_kernels.cu
    wgan_kernel<<<cuda_gridsize(n), BLOCK>>>(n, pred, truth, delta, error);
    check_error(cudaPeekAtLastError());
}




//实现的是gru推导公式中的相关内容。。具体参考gru推导
__global__ void weighted_sum_kernel(int n, float *a, float *b, float *s, float *c)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n){
        c[i] = s[i]*a[i] + (1-s[i])*(b ? b[i] : 0);
    }
}

__global__ void deinter_kernel(int NX, float *X, int NY, float *Y, int B, float *OUT)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < (NX+NY)*B){
        int b = i / (NX+NY);
        int j = i % (NX+NY);
        if (j < NX){
            if(X) X[b*NX + j] += OUT[i];
        } else {
            if(Y) Y[b*NY + j - NX] += OUT[i];
        }
    }
}

extern "C" void deinter_gpu(int NX, float *X, int NY, float *Y, int B, float *OUT)
{
    deinter_kernel<<<cuda_gridsize((NX+NY)*B), BLOCK>>>(NX, X, NY, Y, B, OUT);
    check_error(cudaPeekAtLastError());
}

__global__ void inter_kernel(int NX, float *X, int NY, float *Y, int B, float *OUT)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < (NX+NY)*B){
        int b = i / (NX+NY);
        int j = i % (NX+NY);
        if (j < NX){
            OUT[i] = X[b*NX + j];
        } else {
            OUT[i] = Y[b*NY + j - NX];
        }
    }
}

extern "C" void inter_gpu(int NX, float *X, int NY, float *Y, int B, float *OUT)
{
    inter_kernel<<<cuda_gridsize((NX+NY)*B), BLOCK>>>(NX, X, NY, Y, B, OUT);
    check_error(cudaPeekAtLastError());
}

//weighted_sum_gpu(l.state_gpu, l.h_gpu, l.z_gpu, l.outputs*l.batch, l.output_gpu);
//实现的是gru推导过程中的公式，具体参考gru前向推导
extern "C" void weighted_sum_gpu(float *a, float *b, float *s, int num, float *c)
{
    //weighted_sum_kernel具体实现参考src/blas_kernels.cu
    weighted_sum_kernel<<<cuda_gridsize(num), BLOCK>>>(num, a, b, s, c);
    check_error(cudaPeekAtLastError());
}

//实现的是gru层的反向传播的相关内容，具体参考gru反向传播推导
__global__ void weighted_delta_kernel(int n, float *a, float *b, float *s, float *da, float *db, float *ds, float *dc)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n){
        if(da) da[i] += dc[i] * s[i];
        if(db) db[i] += dc[i] * (1-s[i]);
        ds[i] += dc[i] * (a[i] - b[i]);
    }
}
//weighted_delta_gpu(l.state_gpu, l.h_gpu, l.z_gpu, prev_delta_gpu, uh.delta_gpu, uz.delta_gpu, l.outputs*l.batch, l.delta_gpu);
//具体实现的是gru反向传播的相关内容。。参考gru反向传播推导
extern "C" void weighted_delta_gpu(float *a, float *b, float *s, float *da, float *db, float *ds, int num, float *dc)
{
    //weighted_delta_kernel的具体实现参考src/blas_kernels.cu
    weighted_delta_kernel<<<cuda_gridsize(num), BLOCK>>>(num, a, b, s, da, db, ds, dc);
    check_error(cudaPeekAtLastError());
}

////实现的是将a和b中的值相乘以后加到c上去
__global__ void mult_add_into_kernel(int n, float *a, float *b, float *c)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n){
        c[i] += a[i]*b[i];
    }
}

//mult_add_into_gpu(l.outputs*l.batch, l.forgot_delta_gpu, l.r_gpu, prev_delta_gpu);
//实现的是将l.forgot_delta_gpu和l.r_gpu中的值相乘之后加到prev_delta_gpu上去
extern "C" void mult_add_into_gpu(int num, float *a, float *b, float *c)
{
    mult_add_into_kernel<<<cuda_gridsize(num), BLOCK>>>(num, a, b, c);
    check_error(cudaPeekAtLastError());
}

//softmax_device(input + b*batch_offset + g*group_offset, n, temp, stride, output + b*batch_offset + g*group_offset)
//实现softmax函数中的相关计算
__device__ void softmax_device(float *input, int n, float temp, int stride, float *output)
{
    int i;
    float sum = 0;
    float largest = -INFINITY;
    for(i = 0; i < n; ++i){
        int val = input[i*stride];
        largest = (val>largest) ? val : largest;
    }
    for(i = 0; i < n; ++i){
        float e = expf(input[i*stride]/temp - largest/temp);
        sum += e;
        output[i*stride] = e;
    }
    for(i = 0; i < n; ++i){
        output[i*stride] /= sum;
    }
}


__global__ void softmax_tree_kernel(float *input, int spatial, int batch, int stride, float temp, float *output, int groups, int *group_size, int *group_offset)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= spatial*batch*groups) return;
    int s = id % spatial;
    id = id / spatial;
    int g = id % groups;
    int b = id / groups;
    int goff = group_offset[g]*spatial;
    int boff = b*stride;
    softmax_device(input + goff + boff + s, group_size[g], temp, spatial, output + goff + boff + s);
}

extern "C" void softmax_tree(float *input, int spatial, int batch, int stride, float temp, float *output, tree hier)
{
    int *tree_groups_size = cuda_make_int_array(hier.group_size, hier.groups);
    int *tree_groups_offset = cuda_make_int_array(hier.group_offset, hier.groups);
    /*
       static int *tree_groups_size = 0;
       static int *tree_groups_offset = 0;
       if(!tree_groups_size){
       tree_groups_size = cuda_make_int_array(hier.group_size, hier.groups);
       tree_groups_offset = cuda_make_int_array(hier.group_offset, hier.groups);
       }
     */
    int num = spatial*batch*hier.groups;
    softmax_tree_kernel<<<cuda_gridsize(num), BLOCK>>>(input, spatial, batch, stride, temp, output, hier.groups, tree_groups_size, tree_groups_offset);
    check_error(cudaPeekAtLastError());
    cuda_free((float *)tree_groups_size);
    cuda_free((float *)tree_groups_offset);
}

//实现softmax的核函数
__global__ void softmax_kernel(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= batch*groups) return;
    int b = id / groups;
    int g = id % groups;
    //softma_device的具体实现参考src/blas_kernels.cu
    softmax_device(input + b*batch_offset + g*group_offset, n, temp, stride, output + b*batch_offset + g*group_offset);
}

//softmax_gpu(net.input_gpu, l.c, l.batch*l.c, l.inputs/l.c, l.w*l.h, 1, l.w*l.h, 1, l.output_gpu);
//softmax的gpu实现
extern "C" void softmax_gpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
    //实现softmax的核函数，具体实现参考src/blas_kernels.cu
    softmax_kernel<<<cuda_gridsize(batch*groups), BLOCK>>>(input, n, batch, batch_offset, groups, group_offset, stride, temp, output);
    check_error(cudaPeekAtLastError());
}

//upsample_kernel<<<cuda_gridsize(size), BLOCK>>>(size, in, w, h, c, batch, stride, forward, scale, out);
//实现upsample的核函数
__global__ void upsample_kernel(size_t N, float *x, int w, int h, int c, int batch, int stride, int forward, float scale, float *out)
{
    size_t i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= N) return;
    int out_index = i;
    int out_w = i%(w*stride);
    i = i/(w*stride);
    int out_h = i%(h*stride);
    i = i/(h*stride);
    int out_c = i%c;
    i = i/c;
    int b = i%batch;

    int in_w = out_w / stride;
    int in_h = out_h / stride;
    int in_c = out_c;

    int in_index = b*w*h*c + in_c*w*h + in_h*w + in_w;


    if(forward) out[out_index] += scale * x[in_index];
    //atomicAdd的实现参考https://blog.csdn.net/shungry/article/details/90521592
    //实现的是将scale * out[out_index]加到x+in_index的地址上去
    else atomicAdd(x+in_index, scale * out[out_index]);
}

//upsample_gpu(l.output_gpu, l.out_w, l.out_h, l.c, l.batch, l.stride, 0, l.scale, net.input_gpu);
//upsample层的gpu实现
extern "C" void upsample_gpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out)
{
    size_t size = w*h*c*batch*stride*stride;
    //upsample_kernel的具体实现参考src/blas_kernels.cu
    upsample_kernel<<<cuda_gridsize(size), BLOCK>>>(size, in, w, h, c, batch, stride, forward, scale, out);
    check_error(cudaPeekAtLastError());
}
