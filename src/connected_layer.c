#include "connected_layer.h"
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//创建一个全连接层，赋值相关参数，分配内存
layer make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam)
{
    int i;
    layer l = {0};
    l.learning_rate_scale = 1;
    l.type = CONNECTED;

    l.inputs = inputs;
    l.outputs = outputs;
    l.batch=batch;
    l.batch_normalize = batch_normalize;
    //从这里能看到全连接层的输入1,1,c
    l.h = 1;
    l.w = 1;
    l.c = inputs;
    //全连接层的输出1,1,n
    l.out_h = 1;
    l.out_w = 1;
    l.out_c = outputs;

    l.output = calloc(batch*outputs, sizeof(float));
    l.delta = calloc(batch*outputs, sizeof(float));

    l.weight_updates = calloc(inputs*outputs, sizeof(float));
    l.bias_updates = calloc(outputs, sizeof(float));

    l.weights = calloc(outputs*inputs, sizeof(float));
    l.biases = calloc(outputs, sizeof(float));
    //全连接层的前向传播
    //实现细节参考src/connected_layer.c
    l.forward = forward_connected_layer;
    //全连接层的反向传播
    //实现细节参考src/connected_layer.c
    l.backward = backward_connected_layer;
    //全连接层的参数更新
    //具体细节参考src/connected_layer.c
    l.update = update_connected_layer;

    //float scale = 1./sqrt(inputs);
    float scale = sqrt(2./inputs);
    //初始化
    for(i = 0; i < outputs*inputs; ++i){
        //rand_uniform具体参考src/utils.c
        l.weights[i] = scale*rand_uniform(-1, 1);
    }

    for(i = 0; i < outputs; ++i){
        l.biases[i] = 0;
    }

    if(adam){
        l.m = calloc(l.inputs*l.outputs, sizeof(float));
        l.v = calloc(l.inputs*l.outputs, sizeof(float));
        l.bias_m = calloc(l.outputs, sizeof(float));
        l.scale_m = calloc(l.outputs, sizeof(float));
        l.bias_v = calloc(l.outputs, sizeof(float));
        l.scale_v = calloc(l.outputs, sizeof(float));
    }
    if(batch_normalize){
        l.scales = calloc(outputs, sizeof(float));
        l.scale_updates = calloc(outputs, sizeof(float));
        for(i = 0; i < outputs; ++i){
            l.scales[i] = 1;
        }

        l.mean = calloc(outputs, sizeof(float));
        l.mean_delta = calloc(outputs, sizeof(float));
        l.variance = calloc(outputs, sizeof(float));
        l.variance_delta = calloc(outputs, sizeof(float));

        l.rolling_mean = calloc(outputs, sizeof(float));
        l.rolling_variance = calloc(outputs, sizeof(float));

        l.x = calloc(batch*outputs, sizeof(float));
        l.x_norm = calloc(batch*outputs, sizeof(float));
    }

#ifdef GPU
    //GPU版本的全连接层
    //全连接层的gpu前向传播,具体实现参考src/connected_layer.c
    l.forward_gpu = forward_connected_layer_gpu;
    //全连接层的gpu反向传播,具体实现参考src/connected_layer.c
    l.backward_gpu = backward_connected_layer_gpu;
    //全连接层的参数更新，具体是西安参考src/connected_layer.c
    l.update_gpu = update_connected_layer_gpu;

    //给相应参数在gpu上分配内存。。
    l.weights_gpu = cuda_make_array(l.weights, outputs*inputs);
    l.biases_gpu = cuda_make_array(l.biases, outputs);

    l.weight_updates_gpu = cuda_make_array(l.weight_updates, outputs*inputs);
    l.bias_updates_gpu = cuda_make_array(l.bias_updates, outputs);

    l.output_gpu = cuda_make_array(l.output, outputs*batch);
    l.delta_gpu = cuda_make_array(l.delta, outputs*batch);
    if (adam) {
        l.m_gpu =       cuda_make_array(0, inputs*outputs);
        l.v_gpu =       cuda_make_array(0, inputs*outputs);
        l.bias_m_gpu =  cuda_make_array(0, outputs);
        l.bias_v_gpu =  cuda_make_array(0, outputs);
        l.scale_m_gpu = cuda_make_array(0, outputs);
        l.scale_v_gpu = cuda_make_array(0, outputs);
    }

    if(batch_normalize){
        l.mean_gpu = cuda_make_array(l.mean, outputs);
        l.variance_gpu = cuda_make_array(l.variance, outputs);

        l.rolling_mean_gpu = cuda_make_array(l.mean, outputs);
        l.rolling_variance_gpu = cuda_make_array(l.variance, outputs);

        l.mean_delta_gpu = cuda_make_array(l.mean, outputs);
        l.variance_delta_gpu = cuda_make_array(l.variance, outputs);

        l.scales_gpu = cuda_make_array(l.scales, outputs);
        l.scale_updates_gpu = cuda_make_array(l.scale_updates, outputs);

        l.x_gpu = cuda_make_array(l.output, l.batch*outputs);
        l.x_norm_gpu = cuda_make_array(l.output, l.batch*outputs);
#ifdef CUDNN
        cudnnCreateTensorDescriptor(&l.normTensorDesc);
        cudnnCreateTensorDescriptor(&l.dstTensorDesc);
        cudnnSetTensor4dDescriptor(l.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.batch, l.out_c, l.out_h, l.out_w); 
        cudnnSetTensor4dDescriptor(l.normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l.out_c, 1, 1); 
#endif
    }
#endif
    l.activation = activation;
    fprintf(stderr, "connected                            %4d  ->  %4d\n", inputs, outputs);
    return l;
}
//更新全连接层的参数，基本和卷积层更新一致，也是动量梯度下降
void update_connected_layer(layer l, update_args a)
{
    //获取学习率，动量，衰减系数，batch_size的值
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;
    //更新偏置，同时积累偏置动量
    axpy_cpu(l.outputs, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.outputs, momentum, l.bias_updates, 1);
    //如果有batchnormalize
    if(l.batch_normalize){
        //更新l.scales，并积累scales的动量
        axpy_cpu(l.outputs, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.outputs, momentum, l.scale_updates, 1);
    }
    //计算权重衰减
    axpy_cpu(l.inputs*l.outputs, -decay*batch, l.weights, 1, l.weight_updates, 1);
    //更新权重，积累权重动量
    axpy_cpu(l.inputs*l.outputs, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(l.inputs*l.outputs, momentum, l.weight_updates, 1);
}
//全连接层的前向传播
void forward_connected_layer(layer l, network net)
{
    //fill_cpu的具体实现参考src/blas.c
    //这里实现的是将l.output中的值置0
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);
    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    //a定位当前层的输入
    float *a = net.input;
    //定位当前层的权重
    float *b = l.weights;
    //定位当前层的输出
    float *c = l.output;
    //调用gemm函数，具体实现参考src/gemm.c
    //最终调用的是gemm_nt函数，输入实际上就是一个行向量，权重展开矩阵是将每一个核展开成一行，从而转置之后每一个核的参数成为了矩阵的一列
    //完成了向量跟矩阵的计算。。得到最终结果的输出
    //这里跟卷积层调用gemm处不同，卷积层在gemm中一次处理一张图片，而这里因为FC层本身一张图的特征已经只有一个行向量了
    //所以直接将一个batch全部放到一起作为一个矩阵进入gemm，至于为什么要转置。。比较容易接受的理解是推导矩阵维度，比较方便看出来
    //实际上跟gemm的具体实现有关系，其实这里也可以和卷积层一致加一个循环，但是效率比较低，拼接成矩阵会更快
    //具体的逻辑画画图就能推导出来了
    gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);
    if(l.batch_normalize){
        //如果需要batchnorm操作
        //进行batchnorm的前向传播，具体参考src/batchnorm_layer.c
        forward_batchnorm_layer(l, net);
    } else {
        //添加偏置到l.output上
        //具体实现参考src/convolutional_layer.c
        add_bias(l.output, l.biases, l.batch, l.outputs, 1);
    }
    //进行激活操作
    //具体实现参考src/activations.c
    activate_array(l.output, l.outputs*l.batch, l.activation);
}
//全连接层的反向传播
void backward_connected_layer(layer l, network net)
{
    //计算激活函数相对于l.output的导数，并将结果跟l.delta中的参数相乘，新的结果写入l.delta
    //具体实现参考src/activations.c
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);
    //如果有batchnorm操作
    if(l.batch_normalize){
        //BN的返现传播，具体参考src/batchnorm_layer.c
        backward_batchnorm_layer(l, net);
    } else {
        //计算偏置的更新值,具体细节参考src/convolutional_layer.c
        backward_bias(l.bias_updates, l.delta, l.batch, l.outputs, 1);
    }

    int m = l.outputs;
    int k = l.batch;
    int n = l.inputs;
    //定位当前层的l.delta
    float *a = l.delta;
    //定位当前层的net.input
    float *b = net.input;
    //定位存储当前层权重更新值的位置
    float *c = l.weight_updates;
    //调用gemm完成l.weight_updates的计算和更新
    //具体细节参考src/gemm.c实际调用的是gemm_tn函数
    //实现l.delta的转置和et.input展开矩阵的乘积操作，然后完成对l.weight_updates的更新
    //跟前像传播中的原理一致，一整个batch拼接起来会更快的去计算
    //转置的原因根据FC反向传播的推导和整个计算过程不难推理出来，其实和前向传播中的转置逻辑上是一样的，于gemm实现关联
    gemm(1,0,m,n,k,1,a,m,b,n,1,c,n);

    //接下来就是传递l.delta到上一层的操作了
    m = l.batch;
    k = l.outputs;
    n = l.inputs;

    a = l.delta;
    b = l.weights;
    c = net.delta;
    //如果还存在上一层的话。。。
    //再次调用gemm_nn计算和传递l.delta
    //实际上调用的是gemm_nn，推导矩阵维度看起来会方便一点。。。
    if(c) gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
}


void denormalize_connected_layer(layer l)
{
    int i, j;
    for(i = 0; i < l.outputs; ++i){
        float scale = l.scales[i]/sqrt(l.rolling_variance[i] + .000001);
        for(j = 0; j < l.inputs; ++j){
            l.weights[i*l.inputs + j] *= scale;
        }
        l.biases[i] -= l.rolling_mean[i] * scale;
        l.scales[i] = 1;
        l.rolling_mean[i] = 0;
        l.rolling_variance[i] = 1;
    }
}


void statistics_connected_layer(layer l)
{
    if(l.batch_normalize){
        printf("Scales ");
        print_statistics(l.scales, l.outputs);
        /*
           printf("Rolling Mean ");
           print_statistics(l.rolling_mean, l.outputs);
           printf("Rolling Variance ");
           print_statistics(l.rolling_variance, l.outputs);
         */
    }
    printf("Biases ");
    print_statistics(l.biases, l.outputs);
    printf("Weights ");
    print_statistics(l.weights, l.outputs);
}

#ifdef GPU

void pull_connected_layer(layer l)
{
    cuda_pull_array(l.weights_gpu, l.weights, l.inputs*l.outputs);
    cuda_pull_array(l.biases_gpu, l.biases, l.outputs);
    cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.inputs*l.outputs);
    cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
    if (l.batch_normalize){
        cuda_pull_array(l.scales_gpu, l.scales, l.outputs);
        cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.outputs);
        cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.outputs);
    }
}

void push_connected_layer(layer l)
{
    cuda_push_array(l.weights_gpu, l.weights, l.inputs*l.outputs);
    cuda_push_array(l.biases_gpu, l.biases, l.outputs);
    cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.inputs*l.outputs);
    cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
    if (l.batch_normalize){
        cuda_push_array(l.scales_gpu, l.scales, l.outputs);
        cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.outputs);
        cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.outputs);
    }
}

void update_connected_layer_gpu(layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;
    if(a.adam){
        adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.inputs*l.outputs, batch, a.t);
        adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.outputs, batch, a.t);
        if(l.scales_gpu){
            adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.outputs, batch, a.t);
        }
    }else{
        axpy_gpu(l.outputs, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
        scal_gpu(l.outputs, momentum, l.bias_updates_gpu, 1);

        if(l.batch_normalize){
            axpy_gpu(l.outputs, learning_rate/batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
            scal_gpu(l.outputs, momentum, l.scale_updates_gpu, 1);
        }

        axpy_gpu(l.inputs*l.outputs, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
        axpy_gpu(l.inputs*l.outputs, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
        scal_gpu(l.inputs*l.outputs, momentum, l.weight_updates_gpu, 1);
    }
}
//全连接层前向传播的gpu版本
void forward_connected_layer_gpu(layer l, network net)
{
    //这里实现的是将l.output_gpu中的值用0初始化
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);

    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    //定位相应的参数的位置
    float * a = net.input_gpu;
    float * b = l.weights_gpu;
    float * c = l.output_gpu;

    //调用gemm_gpu函数
    //gemm_gpu的具体实现参考src/gemm.c
    //这里实现的是net.input_gpu和l.weights_gpu转置的乘积
    gemm_gpu(0,1,m,n,k,1,a,k,b,k,1,c,n);
    //如果需要进行BN操作
    if (l.batch_normalize) {
        //进行bn的前向传播，具体实现参考src/batchnorm_layer.c
        forward_batchnorm_layer_gpu(l, net);
    } else {
        //如果不需要bn操作
        //将l.biases_gpu中的值加到l.output_gpu中
        //具体实现参考src/blas_kernels.cu
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.outputs, 1);
    }
    //activate_array_gpu的具体实现参考src/activation_kernels.cu
    //这里实现的是计算相应的l.output_gpu的激活。。结果写入l.output_gpu
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
}
//全连接层的反向传播的gpu版本
void backward_connected_layer_gpu(layer l, network net)
{   
    //constrain_gpu的具体实现参考src/blas_kernels.cu
    //将l.delta_gpu中的值修正到-1和1之间
    constrain_gpu(l.outputs*l.batch, 1, l.delta_gpu, 1);
    //计算相应激活函数相对于l.output_gpu的导数值，将得到的导数值乘到原来的l.delta_gpu上
    //结果写入l.delta_gpu
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    //如果需要进行bn操作
    if(l.batch_normalize){
        //进行bn的前向传播，具体实现参考src/batchnorm_layer.c
        backward_batchnorm_layer_gpu(l, net);
    } else {
        //如果不需要bn操作的话
        //计算偏置的更新。。具体实现参考src/blas_kernels.cu
        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.outputs, 1);
    }

    //接下来就是计算权重的更新值了，具体计算方式参考全连接层的反向传播推导
    int m = l.outputs;
    int k = l.batch;
    int n = l.inputs;
    float * a = l.delta_gpu;
    float * b = net.input_gpu;
    float * c = l.weight_updates_gpu;
    //还是调用gemm_gpu函数，实现l.delta_gpu的转置和net.input_gpu的乘积
    //结果写入c中，具体参考src/gemm.c
    gemm_gpu(1,0,m,n,k,1,a,m,b,n,1,c,n);

    //接下来就是计算损失函数相对于当前层的input的导数了
    m = l.batch;
    k = l.outputs;
    n = l.inputs;

    a = l.delta_gpu;
    b = l.weights_gpu;
    c = net.delta_gpu;
    //如果还有上一层，需要网上传递l.delta的话
    //调用gemm_cpu，计算l.delta_gpu和l.weights_gpu的乘积，结果写入net.delta_gpu，完成delta的传递
    if(c) gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
}
#endif
