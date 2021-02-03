#include "l2norm_layer.h"
#include "activations.h"
#include "blas.h"
#include "cuda.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
//创建一个L2正则化层
layer make_l2norm_layer(int batch, int inputs)
{
    fprintf(stderr, "l2norm                                         %4d\n",  inputs);
    layer l = {0};
    l.type = L2NORM;
    l.batch = batch;
    l.inputs = inputs;
    l.outputs = inputs;
    l.output = calloc(inputs*batch, sizeof(float));
    l.scales = calloc(inputs*batch, sizeof(float));
    l.delta = calloc(inputs*batch, sizeof(float));
    //l2正则化的前向传播，具体参考src/l2norm_layer.c
    l.forward = forward_l2norm_layer;
    //L2正则化的反向传播，具体参考src/l2norm_layer.c
    l.backward = backward_l2norm_layer;
    #ifdef GPU
    //gpu版本的l2正则化前向传播，具体实现参考src/l2norm_layer.c
    l.forward_gpu = forward_l2norm_layer_gpu;
    //gpu版本的l2正则化的反向传播，具体实现参考src/l2norm_layer.c
    l.backward_gpu = backward_l2norm_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs*batch); 
    l.scales_gpu = cuda_make_array(l.output, inputs*batch); 
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch); 
    #endif
    return l;
}
//l2的前向传播
void forward_l2norm_layer(const layer l, network net)
{   
    //这里是实现的是将net.input中的值赋值到l.output中去
    copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
    //l2normalize_cpu的具体实现参考src/blas.c
    //实现通道维度上的正则化
    l2normalize_cpu(l.output, l.scales, l.batch, l.out_c, l.out_w*l.out_h);
}
//l2正则化的反向传播
void backward_l2norm_layer(const layer l, network net)
{
    //axpy_cpu(l.inputs*l.batch, 1, l.scales, 1, l.delta, 1);
    //将l.delta的值赋值到net.delta中去
    axpy_cpu(l.inputs*l.batch, 1, l.delta, 1, net.delta, 1);
}

#ifdef GPU
//gpu版本的l2正则化的前向传播
void forward_l2norm_layer_gpu(const layer l, network net)
{
    copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
    //l2normalize_gpu的具体实现参考src/blas_kernels.cu
    l2normalize_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_w*l.out_h);
}
//gpu版本的l2正则化的反向传播
void backward_l2norm_layer_gpu(const layer l, network net)
{
    axpy_gpu(l.batch*l.inputs, 1, l.scales_gpu, 1, l.delta_gpu, 1);
    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}

#endif
