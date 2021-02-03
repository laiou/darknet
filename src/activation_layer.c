#include "activation_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//创建一个激活层，给相关参数赋值，分配相应的内存
layer make_activation_layer(int batch, int inputs, ACTIVATION activation)
{
    //给相关参数分配内存和赋值
    layer l = {0};
    l.type = ACTIVE;

    l.inputs = inputs;
    l.outputs = inputs;
    l.batch=batch;

    l.output = calloc(batch*inputs, sizeof(float*));
    l.delta = calloc(batch*inputs, sizeof(float*));
    //激活层的前向传播
    //具体实现参考src/activation_layer.c
    l.forward = forward_activation_layer;
    //激活层的反向传播
    //具体实现参考src/activation_layer.c
    l.backward = backward_activation_layer;
    //gpu版本的激活层
#ifdef GPU
    //gpu版本的激活层前向传播,具体实现参考src/activation_layer.c
    l.forward_gpu = forward_activation_layer_gpu;
    //gpu版本的激活层反向传播,具体实现参考src/activation_layer.c
    l.backward_gpu = backward_activation_layer_gpu;
    //给gpu的输出分配相应的内存
    //cuda_make_array的具体实现参考src/cuda.c
    l.output_gpu = cuda_make_array(l.output, inputs*batch);
    //给gpu的delta分配内存
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch);
#endif
    l.activation = activation;
    fprintf(stderr, "Activation Layer: %d inputs\n", inputs);
    return l;
}
//激活层的前向传播
void forward_activation_layer(layer l, network net)
{
    //copy_cpu的具体实现参考src/blas.c
    //这里实现的是将net.input中的值赋值给l.output
    //net.input里面存储了当前层次的输入特征图
    copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
    //activate_array的具体实现参考src/activations.c
    //这里实现的是对应激活函数的操作，将l.output中的值通过对应激活函数得到激活后的值
    //写入l.output
    activate_array(l.output, l.outputs*l.batch, l.activation);
}
//实现激活层的反向传播
void backward_activation_layer(layer l, network net)
{
    //根据相应的激活函数，计算激活函数相对于当前l.output的导数值，跟原来存储到l.delta中的值相乘。结果存入l.delta
    //具体细节参考src/activations.c
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);
    //将l.delta中的数值赋值给net.delta，将l.delta向上一层传递
    copy_cpu(l.outputs*l.batch, l.delta, 1, net.delta, 1);
}

#ifdef GPU
//gpu版本的激活层前向传播
void forward_activation_layer_gpu(layer l, network net)
{   
    //copy_gpu的具体实现参考src/blas_kernels.cu
    //实现的是将net.input_gpu中的值赋值到net.output_gpu中去
    copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
    //activate_array_gpu的具体实现参考src/activation_kernels.cu
    //在gpu上完成相应的激活运算
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
}
//gpu版本的激活层的反向传播
void backward_activation_layer_gpu(layer l, network net)
{   
    //gradient_array_gpu的具体实现参考src/activation_kernel.cu
    //跟据不同的激活函数计算相应的delta
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    //将l.delta_gpu中的值赋值到net.delta_gpu中，完成l.delta_gpu的传递
    copy_gpu(l.outputs*l.batch, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif
