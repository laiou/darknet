#include "gru_layer.h"
#include "connected_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void increment_layer(layer *l, int steps)
{
    int num = l->outputs*l->batch*steps;
    l->output += num;
    l->delta += num;
    l->x += num;
    l->x_norm += num;

#ifdef GPU
    l->output_gpu += num;
    l->delta_gpu += num;
    l->x_gpu += num;
    l->x_norm_gpu += num;
#endif
}
//创建一个GRU层
//算法理论推导参考https://www.cnblogs.com/jiangxinyang/p/9376021.html
layer make_gru_layer(int batch, int inputs, int outputs, int steps, int batch_normalize, int adam)
{
    fprintf(stderr, "GRU Layer: %d inputs, %d outputs\n", inputs, outputs);
    //RNN中将一个batch的数据根据时序划分成多个steps的数据
    batch = batch / steps;
    layer l = {0};
    l.batch = batch;
    l.type = GRU;
    l.steps = steps;
    l.inputs = inputs;
    //创建一个全连接层 这里采用线性激活函数，根据定义，可以看成是不激活。。。
    //这里的LINEAR实际上就是没有激活。。
    //这里用6个全连接层来实现gru的相关操作。。具体参考gru的相关原理和前向传播的参数拆分。。
    //注意这里没有实现算法层面上的那个输出层output层，l.output是进入那个算法层面上output层的输入。。。
    //其中wr,和ur配合实现算法层面上的更新门
    //uz和wz配合实现算法层面上的重置门
    //uh和wh配合完成当前层状态值h_t的计算
    l.uz = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.uz) = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize, adam);
    //重置l.uz->batch的值为一个steps数据的大小
    l.uz->batch = batch;

    l.wz = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.wz) = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize, adam);
    l.wz->batch = batch;

    l.ur = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.ur) = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize, adam);
    l.ur->batch = batch;

    l.wr = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.wr) = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize, adam);
    l.wr->batch = batch;



    l.uh = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.uh) = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize, adam);
    l.uh->batch = batch;

    l.wh = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.wh) = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize, adam);
    l.wh->batch = batch;

    l.batch_normalize = batch_normalize;


    l.outputs = outputs;
    l.output = calloc(outputs*batch*steps, sizeof(float));
    l.delta = calloc(outputs*batch*steps, sizeof(float));
    l.state = calloc(outputs*batch, sizeof(float));
    l.prev_state = calloc(outputs*batch, sizeof(float));
    l.forgot_state = calloc(outputs*batch, sizeof(float));
    l.forgot_delta = calloc(outputs*batch, sizeof(float));

    l.r_cpu = calloc(outputs*batch, sizeof(float));
    l.z_cpu = calloc(outputs*batch, sizeof(float));
    l.h_cpu = calloc(outputs*batch, sizeof(float));
    //gru的前向传播,实现细节参考src/forward_gru_layer
    l.forward = forward_gru_layer;
    //gru的反向出传播，直接参考gpu版本的反向传播***********
    l.backward = backward_gru_layer;
    //gru的参数更新，具体参考src/gru_layer.c
    l.update = update_gru_layer;

#ifdef GPU
    //gpu版本的gru层的前向传播，具体实现参考src/gru_layer.c
    l.forward_gpu = forward_gru_layer_gpu;
    //gpu版本的gru层的反向传播,具体实现参考src/gru_layer.c
    l.backward_gpu = backward_gru_layer_gpu;
    //gpu版本的gru层的反向传播，具体实现参考src/gru_layer.c
    l.update_gpu = update_gru_layer_gpu;

    l.forgot_state_gpu = cuda_make_array(0, batch*outputs);
    l.forgot_delta_gpu = cuda_make_array(0, batch*outputs);
    l.prev_state_gpu = cuda_make_array(0, batch*outputs);
    l.state_gpu = cuda_make_array(0, batch*outputs);
    l.output_gpu = cuda_make_array(0, batch*outputs*steps);
    l.delta_gpu = cuda_make_array(0, batch*outputs*steps);
    l.r_gpu = cuda_make_array(0, batch*outputs);
    l.z_gpu = cuda_make_array(0, batch*outputs);
    l.h_gpu = cuda_make_array(0, batch*outputs);

#ifdef CUDNN
    cudnnSetTensor4dDescriptor(l.uz->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.uz->out_c, l.uz->out_h, l.uz->out_w); 
    cudnnSetTensor4dDescriptor(l.uh->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.uh->out_c, l.uh->out_h, l.uh->out_w); 
    cudnnSetTensor4dDescriptor(l.ur->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.ur->out_c, l.ur->out_h, l.ur->out_w); 
    cudnnSetTensor4dDescriptor(l.wz->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.wz->out_c, l.wz->out_h, l.wz->out_w); 
    cudnnSetTensor4dDescriptor(l.wh->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.wh->out_c, l.wh->out_h, l.wh->out_w); 
    cudnnSetTensor4dDescriptor(l.wr->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.wr->out_c, l.wr->out_h, l.wr->out_w); 
#endif
#endif

    return l;
}
//更新gru层的参数
void update_gru_layer(layer l, update_args a)
{   //具体细节实现参考src/connected_layer.c
    update_connected_layer(*(l.ur), a);
    update_connected_layer(*(l.uz), a);
    update_connected_layer(*(l.uh), a);
    update_connected_layer(*(l.wr), a);
    update_connected_layer(*(l.wz), a);
    update_connected_layer(*(l.wh), a);
}
//gru层的前向传播
void forward_gru_layer(layer l, network net)
{   
    network s = net;
    s.train = net.train;
    int i;
    //定位其中相关子层次的位置
    layer uz = *(l.uz);
    layer ur = *(l.ur);
    layer uh = *(l.uh);

    layer wz = *(l.wz);
    layer wr = *(l.wr);
    layer wh = *(l.wh);
    //实现的是将相应的存储delta的位置置0
    fill_cpu(l.outputs * l.batch * l.steps, 0, uz.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, ur.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, uh.delta, 1);

    fill_cpu(l.outputs * l.batch * l.steps, 0, wz.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, wr.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, wh.delta, 1);
    //如果是在训练的话
    if(net.train) {
        //将l.delta置0
        fill_cpu(l.outputs * l.batch * l.steps, 0, l.delta, 1);
        //将l.state中的值赋值到l.prev_state中，也就是将当前状态值保存作为上一个状态
        copy_cpu(l.outputs*l.batch, l.state, 1, l.prev_state, 1);
    }
    //循环进行每一个steps
    for (i = 0; i < l.steps; ++i) {
        //将s.input指向当前的l.state
        //即算法层面上的h_t-1
        s.input = l.state;
        //这里将上一时刻的状态值输入wz和wr进行前向传播
        //全连接层的前向传播的具体实现参考src/connected_layer.c
        //需要注意一下的是这里的全连接层的输入是一个向量，输出也是一个向量
        forward_connected_layer(wz, s);
        forward_connected_layer(wr, s);
        //将s.input换成net.input也就是上一个层次的输出
        //从算法层面上来说，这里就是将s.input指向了当前时刻的输入x_t
        s.input = net.input;
        //将当前时刻的输入x_t输入uz,ur,uh进行前向传播
        forward_connected_layer(uz, s);
        forward_connected_layer(ur, s);
        forward_connected_layer(uh, s);

        //copy_cpu的具体实现参考src/blas.c
        //将uz.output中的值赋值到l.z_cpu中去
        copy_cpu(l.outputs*l.batch, uz.output, 1, l.z_cpu, 1);
        //axpy_cpu的具体实现参考src/blas.c
        //这实现的是将l.z_cpu中的值加上wz.output中的值然后把结果写入l.z_cpu
        axpy_cpu(l.outputs*l.batch, 1, wz.output, 1, l.z_cpu, 1);
        //这里将ur.output中的值赋值到l.r_cpu中
        copy_cpu(l.outputs*l.batch, ur.output, 1, l.r_cpu, 1);
        //将wr.output中的值加到l.r_cpu上，将结果写入l.r_cpu
        axpy_cpu(l.outputs*l.batch, 1, wr.output, 1, l.r_cpu, 1);
        //对l.z_cpu中的值进行激活操作
        //具体实现细节参考src/activations.c
        //将激活后的值写入l.z_cpu
        activate_array(l.z_cpu, l.outputs*l.batch, LOGISTIC);
        //对l.r_cpu中的值进行激活操作
        //将激活后的值写入l.r_cpu
        activate_array(l.r_cpu, l.outputs*l.batch, LOGISTIC);
        //将l.state中的值赋值到l.fotgot_state中
        //也就是将算法层面上的h_t-1写入l.forgot_state
        copy_cpu(l.outputs*l.batch, l.state, 1, l.forgot_state, 1);
        //mul_cpu的具体实现参考src/blas.c
        //这里实现的是将forgot_state中的值与l.r_cpu中的值相乘，结果写入l.forgot_state中
        mul_cpu(l.outputs*l.batch, l.r_cpu, 1, l.forgot_state, 1);
        //这里将输入换成l.forgot_state
        s.input = l.forgot_state;
        //进行wh的前向传播
        forward_connected_layer(wh, s);
        //将uh.output中的值赋值到l.h_cpu中
        copy_cpu(l.outputs*l.batch, uh.output, 1, l.h_cpu, 1);
        //将wh.output中的值加到l.h_cpu上，结果写入l.h_cpu中
        axpy_cpu(l.outputs*l.batch, 1, wh.output, 1, l.h_cpu, 1);
        //是否采用tanh激活
        if(l.tanh){
            //进行tanh激活
            activate_array(l.h_cpu, l.outputs*l.batch, TANH);
        } else {
            //否则进行logistic激活
            activate_array(l.h_cpu, l.outputs*l.batch, LOGISTIC);
        }
        //weighted_sun_cou的实现参考src/blas.c
        //这里实现的是计算当前时刻的状态值，将结果写入l.output中同时作为传入output层的输入
        //但是这个是实现中没有包括算法层面上的那个output层
        //这里计算当前层状态和理论上算法推导有细微差别，实际上是一样的。。只是加权值不同。。总的加权和还是1
        weighted_sum_cpu(l.state, l.h_cpu, l.z_cpu, l.outputs*l.batch, l.output);
        //将l.output中的值赋值到l.state作为当前时刻的状态值
        copy_cpu(l.outputs*l.batch, l.output, 1, l.state, 1);

        net.input += l.inputs*l.batch;
        l.output += l.outputs*l.batch;
        //将每个层次对应的索引向后推一个step位置
        //increament_layer具体实现参考src/gru_layer.c
        increment_layer(&uz, 1);
        increment_layer(&ur, 1);
        increment_layer(&uh, 1);

        increment_layer(&wz, 1);
        increment_layer(&wr, 1);
        increment_layer(&wh, 1);
    }
}

void backward_gru_layer(layer l, network net)
{
}

#ifdef GPU

void pull_gru_layer(layer l)
{
}

void push_gru_layer(layer l)
{
}

void update_gru_layer_gpu(layer l, update_args a)
{
    update_connected_layer_gpu(*(l.ur), a);
    update_connected_layer_gpu(*(l.uz), a);
    update_connected_layer_gpu(*(l.uh), a);
    update_connected_layer_gpu(*(l.wr), a);
    update_connected_layer_gpu(*(l.wz), a);
    update_connected_layer_gpu(*(l.wh), a);
}
//gru层前向传播的gpu版本
void forward_gru_layer_gpu(layer l, network net)
{
    network s = {0};
    s.train = net.train;
    int i;
    //定位相应子层次的位置
    layer uz = *(l.uz);
    layer ur = *(l.ur);
    layer uh = *(l.uh);

    layer wz = *(l.wz);
    layer wr = *(l.wr);
    layer wh = *(l.wh);
    //将相应的delta置0
    fill_gpu(l.outputs * l.batch * l.steps, 0, uz.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, ur.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, uh.delta_gpu, 1);

    fill_gpu(l.outputs * l.batch * l.steps, 0, wz.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, wr.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, wh.delta_gpu, 1);
    //如果是在训练的化
    if(net.train) {
        //将l.delta_gpu置0
        fill_gpu(l.outputs * l.batch * l.steps, 0, l.delta_gpu, 1);
        //将l.state_gpu中的值赋值到l.prev_state_gpu里面
        //也就是将当前的step状态保存成上一个step的状态
        copy_gpu(l.outputs*l.batch, l.state_gpu, 1, l.prev_state_gpu, 1);
    }
    //循环遍历进行每一个steps
    for (i = 0; i < l.steps; ++i) {
        //将s.input_gpu换成l.state_gpu
        s.input_gpu = l.state_gpu;
        //进行wz和wr层的前向传播
        //接下来的过程可以结合cpu版本的前向传播和gru层的公式推导来看
        forward_connected_layer_gpu(wz, s);
        forward_connected_layer_gpu(wr, s);

        s.input_gpu = net.input_gpu;
        forward_connected_layer_gpu(uz, s);
        forward_connected_layer_gpu(ur, s);
        forward_connected_layer_gpu(uh, s);

        copy_gpu(l.outputs*l.batch, uz.output_gpu, 1, l.z_gpu, 1);
        axpy_gpu(l.outputs*l.batch, 1, wz.output_gpu, 1, l.z_gpu, 1);

        copy_gpu(l.outputs*l.batch, ur.output_gpu, 1, l.r_gpu, 1);
        axpy_gpu(l.outputs*l.batch, 1, wr.output_gpu, 1, l.r_gpu, 1);

        activate_array_gpu(l.z_gpu, l.outputs*l.batch, LOGISTIC);
        activate_array_gpu(l.r_gpu, l.outputs*l.batch, LOGISTIC);

        copy_gpu(l.outputs*l.batch, l.state_gpu, 1, l.forgot_state_gpu, 1);
        //mul_gpu的具体实现参考src/blas_kernels.cu
        //这里实现的是将l.r_gpu上的值乘到l.forgot_state_gpu上，结果写入l.forgot_state_gpu里面
        mul_gpu(l.outputs*l.batch, l.r_gpu, 1, l.forgot_state_gpu, 1);

        s.input_gpu = l.forgot_state_gpu;
        forward_connected_layer_gpu(wh, s);

        copy_gpu(l.outputs*l.batch, uh.output_gpu, 1, l.h_gpu, 1);
        axpy_gpu(l.outputs*l.batch, 1, wh.output_gpu, 1, l.h_gpu, 1);

        if(l.tanh){
            activate_array_gpu(l.h_gpu, l.outputs*l.batch, TANH);
        } else {
            activate_array_gpu(l.h_gpu, l.outputs*l.batch, LOGISTIC);
        }
        //weighted_sum_gpu的具体实现参考src/blas_kernels.cu
        //具体实现的内容参考gru前向传播公式推导和相应的代码实现
        weighted_sum_gpu(l.state_gpu, l.h_gpu, l.z_gpu, l.outputs*l.batch, l.output_gpu);
        copy_gpu(l.outputs*l.batch, l.output_gpu, 1, l.state_gpu, 1);

        net.input_gpu += l.inputs*l.batch;
        l.output_gpu += l.outputs*l.batch;
        increment_layer(&uz, 1);
        increment_layer(&ur, 1);
        increment_layer(&uh, 1);

        increment_layer(&wz, 1);
        increment_layer(&wr, 1);
        increment_layer(&wh, 1);
    }
}
//gpu版本的gru层反向传播
//具体细节参考cpu版本的实现和gru的反向传播推导
void backward_gru_layer_gpu(layer l, network net)
{
    network s = {0};
    s.train = net.train;
    int i;
    layer uz = *(l.uz);
    layer ur = *(l.ur);
    layer uh = *(l.uh);

    layer wz = *(l.wz);
    layer wr = *(l.wr);
    layer wh = *(l.wh);

    increment_layer(&uz, l.steps - 1);
    increment_layer(&ur, l.steps - 1);
    increment_layer(&uh, l.steps - 1);

    increment_layer(&wz, l.steps - 1);
    increment_layer(&wr, l.steps - 1);
    increment_layer(&wh, l.steps - 1);

    net.input_gpu += l.inputs*l.batch*(l.steps-1);
    if(net.delta_gpu) net.delta_gpu += l.inputs*l.batch*(l.steps-1);
    l.output_gpu += l.outputs*l.batch*(l.steps-1);
    l.delta_gpu += l.outputs*l.batch*(l.steps-1);
    float *end_state = l.output_gpu;
    for (i = l.steps-1; i >= 0; --i) {
        if(i != 0) copy_gpu(l.outputs*l.batch, l.output_gpu - l.outputs*l.batch, 1, l.state_gpu, 1);
        else copy_gpu(l.outputs*l.batch, l.prev_state_gpu, 1, l.state_gpu, 1);
        float *prev_delta_gpu = (i == 0) ? 0 : l.delta_gpu - l.outputs*l.batch;

        copy_gpu(l.outputs*l.batch, uz.output_gpu, 1, l.z_gpu, 1);
        axpy_gpu(l.outputs*l.batch, 1, wz.output_gpu, 1, l.z_gpu, 1);

        copy_gpu(l.outputs*l.batch, ur.output_gpu, 1, l.r_gpu, 1);
        axpy_gpu(l.outputs*l.batch, 1, wr.output_gpu, 1, l.r_gpu, 1);

        activate_array_gpu(l.z_gpu, l.outputs*l.batch, LOGISTIC);
        activate_array_gpu(l.r_gpu, l.outputs*l.batch, LOGISTIC);

        copy_gpu(l.outputs*l.batch, uh.output_gpu, 1, l.h_gpu, 1);
        axpy_gpu(l.outputs*l.batch, 1, wh.output_gpu, 1, l.h_gpu, 1);

        if(l.tanh){
            activate_array_gpu(l.h_gpu, l.outputs*l.batch, TANH);
        } else {
            activate_array_gpu(l.h_gpu, l.outputs*l.batch, LOGISTIC);
        }
        //weighted_delta_gpu的具体实现参考src/blas_kernels.cu
        //实现的是gru反向推导的相关过程
        weighted_delta_gpu(l.state_gpu, l.h_gpu, l.z_gpu, prev_delta_gpu, uh.delta_gpu, uz.delta_gpu, l.outputs*l.batch, l.delta_gpu);

        if(l.tanh){
            gradient_array_gpu(l.h_gpu, l.outputs*l.batch, TANH, uh.delta_gpu);
        } else {
            gradient_array_gpu(l.h_gpu, l.outputs*l.batch, LOGISTIC, uh.delta_gpu);
        }

        copy_gpu(l.outputs*l.batch, uh.delta_gpu, 1, wh.delta_gpu, 1);

        copy_gpu(l.outputs*l.batch, l.state_gpu, 1, l.forgot_state_gpu, 1);
        mul_gpu(l.outputs*l.batch, l.r_gpu, 1, l.forgot_state_gpu, 1);
        fill_gpu(l.outputs*l.batch, 0, l.forgot_delta_gpu, 1);

        s.input_gpu = l.forgot_state_gpu;
        s.delta_gpu = l.forgot_delta_gpu;

        backward_connected_layer_gpu(wh, s);
        //mul_add_into_gpu的具体实现参考src/blas_kernels.cu
        //实现的是将l.forgot_delta_gpu和l.r_gpu中的值相乘之后加到prev_delta_gpu上去
        if(prev_delta_gpu) mult_add_into_gpu(l.outputs*l.batch, l.forgot_delta_gpu, l.r_gpu, prev_delta_gpu);
        mult_add_into_gpu(l.outputs*l.batch, l.forgot_delta_gpu, l.state_gpu, ur.delta_gpu);

        gradient_array_gpu(l.r_gpu, l.outputs*l.batch, LOGISTIC, ur.delta_gpu);
        copy_gpu(l.outputs*l.batch, ur.delta_gpu, 1, wr.delta_gpu, 1);

        gradient_array_gpu(l.z_gpu, l.outputs*l.batch, LOGISTIC, uz.delta_gpu);
        copy_gpu(l.outputs*l.batch, uz.delta_gpu, 1, wz.delta_gpu, 1);

        s.input_gpu = l.state_gpu;
        s.delta_gpu = prev_delta_gpu;

        backward_connected_layer_gpu(wr, s);
        backward_connected_layer_gpu(wz, s);

        s.input_gpu = net.input_gpu;
        s.delta_gpu = net.delta_gpu;

        backward_connected_layer_gpu(uh, s);
        backward_connected_layer_gpu(ur, s);
        backward_connected_layer_gpu(uz, s);


        net.input_gpu -= l.inputs*l.batch;
        if(net.delta_gpu) net.delta_gpu -= l.inputs*l.batch;
        l.output_gpu -= l.outputs*l.batch;
        l.delta_gpu -= l.outputs*l.batch;
        increment_layer(&uz, -1);
        increment_layer(&ur, -1);
        increment_layer(&uh, -1);

        increment_layer(&wz, -1);
        increment_layer(&wr, -1);
        increment_layer(&wh, -1);
    }
    copy_gpu(l.outputs*l.batch, end_state, 1, l.state_gpu, 1);
}
#endif
