#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "avgpool_layer.h"
#include "cuda.h"
}
//实现avgpool层的前向传播
__global__ void forward_avgpool_layer_kernel(int n, int w, int h, int c, float *input, float *output)
{
    //计算相应的线程id，这里面是一个线程完成一个通道上求平均的计算
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;
    //通过k来定位当前的通道是某张图数据的第几个通道
    int k = id % c;
    id /= c;
    //通过b来定位当前处理的是第几张图片
    int b = id;

    int i;
    int out_index = (k + c*b);
    output[out_index] = 0;
    for(i = 0; i < w*h; ++i){
        int in_index = i + h*w*(k + b*c);
        //将相应通道上上的值累加到输出的相应位置上
        output[out_index] += input[in_index];
    }
    //最后累加完毕计算均值
    output[out_index] /= w*h;
}
//avgpool的反向传播gpu上的计算
__global__ void backward_avgpool_layer_kernel(int n, int w, int h, int c, float *in_delta, float *out_delta)
{
    //还是先计算线程id
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;
    //k定位当前是一张图片数据的第几个通道
    int k = id % c;
    id /= c;
    //b定位当前处理的是第几张图
    int b = id;

    int i;
    int out_index = (k + c*b);
    //一个循环计算相应的delta
    for(i = 0; i < w*h; ++i){
    //计算相应的索引
        int in_index = i + h*w*(k + b*c);
        //将相应的delta计算完写入net.delta完成delta的传递
        //+=还是为了让batch中多张图的结果累加到一起
        in_delta[in_index] += out_delta[out_index] / (w*h);
    }
}

//实现gpu上avgpool层的前向传播
extern "C" void forward_avgpool_layer_gpu(avgpool_layer layer, network net)
{   
    //计算当前层通道和batch的乘积，作为后续cuda线程grid的划分依据
    //因为当前实现的这个avgpool层是将一个通道上的值最终做一个平均，也就是输入是w,h,c但是输出是1,1,c
    //这里划分grid也是通过输出值的数量
    size_t n = layer.c*layer.batch;
    //forward_avgpool_layer_kernel的具体实现参考src/avgpool_layer_kernels.cu
    //实现avgpool的核函数，注意一下的是一个线程完成一个通道上的求和平均
    forward_avgpool_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, layer.w, layer.h, layer.c, net.input_gpu, layer.output_gpu);
    check_error(cudaPeekAtLastError());
}
//实现gpu上avgpool层的反向传播
extern "C" void backward_avgpool_layer_gpu(avgpool_layer layer, network net)
{
    size_t n = layer.c*layer.batch;
    //实现avgpool的反向传播的核函数，具体实现参考src/avgpool_layer_kernels.cu
    //同样是一个线程完成一个通道上的delta的计算
    backward_avgpool_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, layer.w, layer.h, layer.c, net.delta_gpu, layer.delta_gpu);
    check_error(cudaPeekAtLastError());
}

