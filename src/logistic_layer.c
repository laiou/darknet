#include "logistic_layer.h"
#include "activations.h"
#include "blas.h"
#include "cuda.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
//创建一个logistic层,其实也是一个loss层
layer make_logistic_layer(int batch, int inputs)
{
    fprintf(stderr, "logistic x entropy                             %4d\n",  inputs);
    layer l = {0};
    l.type = LOGXENT;
    l.batch = batch;
    l.inputs = inputs;
    l.outputs = inputs;
    l.loss = calloc(inputs*batch, sizeof(float));
    l.output = calloc(inputs*batch, sizeof(float));
    l.delta = calloc(inputs*batch, sizeof(float));
    l.cost = calloc(1, sizeof(float));
    //logistic的前向传播,具体细节参考src/logistic_layer.c
    l.forward = forward_logistic_layer;
    //logistic的反向传播，具体细节参考src/logistic_layer.c
    l.backward = backward_logistic_layer;
    #ifdef GPU
    //gpu版本的前向传播，具体参考src/logistic_layer.c
    l.forward_gpu = forward_logistic_layer_gpu;
    //gpu版本的反向传播，具体参考src/logistic_layer.c
    l.backward_gpu = backward_logistic_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs*batch); 
    l.loss_gpu = cuda_make_array(l.loss, inputs*batch); 
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch); 
    #endif
    return l;
}
//logistic的前向传播
void forward_logistic_layer(const layer l, network net)
{
    //这里实现的是将net.input赋值到l.output中作为输出。。loss层输入和输出一致
    copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
    //计算激活值
    activate_array(l.output, l.outputs*l.batch, LOGISTIC);
    //如果存在真值的情况
    if(net.truth){
        //logistic_x_ent_cpu的具体实现参考src/blas.c
        //更新了l.delta，同时将误差写道l.loss中，l.loss记录的是输入数据在每一个数据位置上的误差损失
        logistic_x_ent_cpu(l.batch*l.inputs, l.output, net.truth, l.delta, l.loss);
        //统计总的损失，将l.loss中全部的值累积起来写入l.cost[0]
        l.cost[0] = sum_array(l.loss, l.batch*l.inputs);
    }
}
//logistic的反向传播
void backward_logistic_layer(const layer l, network net)
{
    //这里实现的是将l.delta中的值赋值到net.delta中，完成l.delta的传递
    axpy_cpu(l.inputs*l.batch, 1, l.delta, 1, net.delta, 1);
}

#ifdef GPU
//gpu版本的logistic层的前向传播
void forward_logistic_layer_gpu(const layer l, network net)
{
    copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, LOGISTIC);
    if(net.truth){
        //logistic_x_ent_gpu的具体实现参考src/blas_kernels.cu
        //计算相应的logistic的损失并且更新相应的梯度
        logistic_x_ent_gpu(l.batch*l.inputs, l.output_gpu, net.truth_gpu, l.delta_gpu, l.loss_gpu);
        cuda_pull_array(l.loss_gpu, l.loss, l.batch*l.inputs);
        l.cost[0] = sum_array(l.loss, l.batch*l.inputs);
    }
}
//gpu版本的logistic层的反向传播
void backward_logistic_layer_gpu(const layer l, network net)
{
    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}

#endif
