#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "dropout_layer.h"
#include "cuda.h"
#include "utils.h"
}

//dropout的相关计算
__global__ void yoloswag420blazeit360noscope(float *input, int size, float *rand, float prob, float scale)
{
    //计算相应的线程id
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    //根据阈值选择是否剪除相应的连接，并完成对应的计算
    if(id < size) input[id] = (rand[id] < prob) ? 0 : input[id]*scale;
}
//gpu版本的dropout层的前向传播
void forward_dropout_layer_gpu(dropout_layer layer, network net)
{
    //不是训练过程。。直接返回。。。
    if (!net.train) return;
    //size表示的是输入的参数量
    int size = layer.inputs*layer.batch;
    //利用cuda_random产生相应的随机数
    //具体实现参考src/cuda.c
    cuda_random(layer.rand_gpu, size);
    /*
    int i;
    for(i = 0; i < size; ++i){
        layer.rand[i] = rand_uniform();
    }
    cuda_push_array(layer.rand_gpu, layer.rand, size);
    */
    //这里实现的是drop层相应的操作
    //具体实现参考src/dropout_layer_kernels.cu
    yoloswag420blazeit360noscope<<<cuda_gridsize(size), BLOCK>>>(net.input_gpu, size, layer.rand_gpu, layer.probability, layer.scale);
    check_error(cudaPeekAtLastError());
}
//gpu版本的dropout层的反向传播
void backward_dropout_layer_gpu(dropout_layer layer, network net)
{
    //判断是否存在net.delta_gpu
    if(!net.delta_gpu) return;
    int size = layer.inputs*layer.batch;
    //具体实现参考src/dropout_layer_kernels.cu
    yoloswag420blazeit360noscope<<<cuda_gridsize(size), BLOCK>>>(net.delta_gpu, size, layer.rand_gpu, layer.probability, layer.scale);
    check_error(cudaPeekAtLastError());
}
