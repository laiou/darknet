#include "local_layer.h"
#include "utils.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>

int local_out_height(local_layer l)
{
    int h = l.h;
    if (!l.pad) h -= l.size;
    else h -= 1;
    return h/l.stride + 1;
}

int local_out_width(local_layer l)
{
    int w = l.w;
    if (!l.pad) w -= l.size;
    else w -= 1;
    return w/l.stride + 1;
}
//创建一个local_layer，局部连接层，工作原理和正常的卷积一样，但是不同的是卷积操作中卷积核在图片上滑动到不通区域上的时候卷积核的参数都是一样的
//但是在local中每个位置的卷积核都不一样，一个位置用一套参数
local_layer make_local_layer(int batch, int h, int w, int c, int n, int size, int stride, int pad, ACTIVATION activation)
{
    int i;
    local_layer l = {0};
    l.type = LOCAL;

    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.batch = batch;
    l.stride = stride;
    l.size = size;
    l.pad = pad;
    //计算local层输出的height
    int out_h = local_out_height(l);
    //计算local层输出的width
    int out_w = local_out_width(l);
    int locations = out_h*out_w;
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;
    //这里的weights就比正常卷积的参数多了locations倍
    l.weights = calloc(c*n*size*size*locations, sizeof(float));
    l.weight_updates = calloc(c*n*size*size*locations, sizeof(float));

    l.biases = calloc(l.outputs, sizeof(float));
    l.bias_updates = calloc(l.outputs, sizeof(float));

    // float scale = 1./sqrt(size*size*c);
    float scale = sqrt(2./(size*size*c));
    //权重初始化
    for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_uniform(-1,1);

    l.output = calloc(l.batch*out_h * out_w * n, sizeof(float));
    l.delta  = calloc(l.batch*out_h * out_w * n, sizeof(float));
    //工作空间
    l.workspace_size = out_h*out_w*size*size*c;
    //local的前向传播，具体细节参考src/local_layer.c
    l.forward = forward_local_layer;
    //local层的反向传播，具体实现参考src/local_layer.c
    l.backward = backward_local_layer;
    //local的参数更新，具体实现参考src/local_layer.c
    l.update = update_local_layer;

#ifdef GPU
    //gpu版本的local层的前向传播，具体实现参考src/local_layer.c
    l.forward_gpu = forward_local_layer_gpu;
    //gpu版本的local层的反向传播，具体实现参考src/local_layer.c
    l.backward_gpu = backward_local_layer_gpu;
    //gpu版本的local层的参数更新，具体实现参考src/local_layer.c
    l.update_gpu = update_local_layer_gpu;

    l.weights_gpu = cuda_make_array(l.weights, c*n*size*size*locations);
    l.weight_updates_gpu = cuda_make_array(l.weight_updates, c*n*size*size*locations);

    l.biases_gpu = cuda_make_array(l.biases, l.outputs);
    l.bias_updates_gpu = cuda_make_array(l.bias_updates, l.outputs);

    l.delta_gpu = cuda_make_array(l.delta, l.batch*out_h*out_w*n);
    l.output_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);

#endif
    l.activation = activation;

    fprintf(stderr, "Local Layer: %d x %d x %d image, %d filters -> %d x %d x %d image\n", h,w,c,n, out_h, out_w, n);

    return l;
}
//local层的前向传播
void forward_local_layer(const local_layer l, network net)
{   
    //计算local层的输出height和width
    //具体实现参考src/local_layer.c
    int out_h = local_out_height(l);
    int out_w = local_out_width(l);
    int i, j;
    //locations表示输出特征图的某一个通道上的数据量
    int locations = out_h * out_w;
    //遍历batch中每一张图的数据
    for(i = 0; i < l.batch; ++i){
        //这里实现的是将l.biases中的值赋值到l.output中相应的位置上
        copy_cpu(l.outputs, l.biases, 1, l.output + i*l.outputs, 1);
    }
    //遍历batch中每张图的处理数据
    for(i = 0; i < l.batch; ++i){
        //定位当前输入的位置
        float *input = net.input + i*l.w*l.h*l.c;
        //将输入特征图展开成一个矩阵，写入工作空间中
        //具体实现参考src/im2col.c
        im2col_cpu(input, l.c, l.h, l.w, 
                l.size, l.stride, l.pad, net.workspace);
        //定位output的位置
        float *output = l.output + i*l.outputs;
        //遍历输出特征图某一个通道上的全部数据
        for(j = 0; j < locations; ++j){
            //a指向权重的位置，因为卷积参数不共享，从而有了这个j*l.size*l.size*l.c*l.n
            float *a = l.weights + j*l.size*l.size*l.c*l.n;
            float *b = net.workspace + j;
            float *c = output + j;
            //m表示这一层的卷积核个数
            int m = l.n;
            int n = 1;
            //k表示一次卷积的参数量
            int k = l.size*l.size*l.c;
            //调用gemm函数实现矩阵a和b乘加操作，结果写入c中
            gemm(0,0,m,n,k,1,a,k,b,locations,1,c,locations);
        }
    }
    //将乘加的结果通过激活函数，结果写入l.output
    activate_array(l.output, l.outputs*l.batch, l.activation);
}

//local层的反向传播
void backward_local_layer(local_layer l, network net)
{
    int i, j;
    //locations表示输出特征图一个通道上的参数量
    int locations = l.out_w*l.out_h;
    //计算损失函数相对于l.output的导数
    //这里的逻辑和卷积操作比较类似
    //gradient_array的具体实现参考src/activations.c，结果写入l.delta
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);
    //遍历batch中每一张图片的处理数据
    for(i = 0; i < l.batch; ++i){
        //将l.delta中相应的值加到l.bias_updates上，结果写入l.bias_updates上
        axpy_cpu(l.outputs, 1, l.delta + i*l.outputs, 1, l.bias_updates, 1);
    }
    //遍历batch中每一张图的数据
    for(i = 0; i < l.batch; ++i){
        //当前层输入的相应位置
        float *input = net.input + i*l.w*l.h*l.c;
        //将当前层输入的特征图展开成矩阵，结果写入工作空间
        im2col_cpu(input, l.c, l.h, l.w, 
                l.size, l.stride, l.pad, net.workspace);
        //遍历输出特征图某个通道上的全部数据
        for(j = 0; j < locations; ++j){ 
            //a定位l.delta的位置
            float *a = l.delta + i*l.outputs + j;
            //b定位工作空间的位置
            float *b = net.workspace + j;
            //c定位权重更新的相应位置
            float *c = l.weight_updates + j*l.size*l.size*l.c*l.n;
            int m = l.n;
            int n = l.size*l.size*l.c;
            int k = 1;
            //调用gemm函数计算损失相对于权重的导数，结果写入c，也就是写入l.weight_updates
            gemm(0,1,m,n,k,1,a,locations,b,locations,1,c,n);
        }
        //接着就是计算传到上一层的l.delta的值，完成l.delta的传递
        if(net.delta){
            //遍历locations，也就是输出特征图上某一个通道下的全部数据
            for(j = 0; j < locations; ++j){
            //a定位权重的相应位置 
                float *a = l.weights + j*l.size*l.size*l.c*l.n;
                //b定位l.delta的位置
                float *b = l.delta + i*l.outputs + j;
                //c是工作空间
                float *c = net.workspace + j;
                //这里的反向传播和正常卷积的逻辑类似
                int m = l.size*l.size*l.c;
                int n = 1;
                int k = l.n;
                //调用gemm函数完成a的转置和b的乘加操作，结果写入c
                gemm(1,0,m,n,k,1,a,m,b,locations,0,c,locations);
            }
            //利用col2im_cpu将c中得到的矩阵还原成特征图的格式，结果写入net.delta
            col2im_cpu(net.workspace, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, net.delta+i*l.c*l.h*l.w);
        }
    }
}
//局部连接层的参数更新
void update_local_layer(local_layer l, update_args a)
{   
    //获得学习率
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    //提取动量值
    float momentum = a.momentum;
    //得到权重的衰减比例
    float decay = a.decay;
    int batch = a.batch;

    int locations = l.out_w*l.out_h;
    int size = l.size*l.size*l.c*l.n*locations;
    //这两步是更新偏置并且积累偏置动量
    axpy_cpu(l.outputs, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.outputs, momentum, l.bias_updates, 1);
    //这里接着计算权重的衰减值，然后更新权重，计算权重动量
    axpy_cpu(size, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(size, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(size, momentum, l.weight_updates, 1);
}

#ifdef GPU
//local层gpu版本的前向传播
void forward_local_layer_gpu(const local_layer l, network net)
{
    //local_out_height计算相应的输出的out_h
    //local_out_height的具体实现参考src/local_layer.c
    int out_h = local_out_height(l);
    //local_out_width计算相应的输出的out_w
    //local_out_width的具体实现参考src/local_layer.c
    int out_w = local_out_width(l);
    int i, j;
    int locations = out_h * out_w;

    for(i = 0; i < l.batch; ++i){
        copy_gpu(l.outputs, l.biases_gpu, 1, l.output_gpu + i*l.outputs, 1);
    }
    //local局部连接层的具体计算。和正常卷积类似，调用im2col_gpu和gemm_gpu相互配合完成
    for(i = 0; i < l.batch; ++i){
        float *input = net.input_gpu + i*l.w*l.h*l.c;
        im2col_gpu(input, l.c, l.h, l.w, 
                l.size, l.stride, l.pad, net.workspace);
        float *output = l.output_gpu + i*l.outputs;
        for(j = 0; j < locations; ++j){
            float *a = l.weights_gpu + j*l.size*l.size*l.c*l.n;
            float *b = net.workspace + j;
            float *c = output + j;

            int m = l.n;
            int n = 1;
            int k = l.size*l.size*l.c;

            gemm_gpu(0,0,m,n,k,1,a,k,b,locations,1,c,locations);
        }
    }
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
}

//gpu版本的local层的反向传播
void backward_local_layer_gpu(local_layer l, network net)
{
    int i, j;
    int locations = l.out_w*l.out_h;

    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    for(i = 0; i < l.batch; ++i){
        axpy_gpu(l.outputs, 1, l.delta_gpu + i*l.outputs, 1, l.bias_updates_gpu, 1);
    }

    for(i = 0; i < l.batch; ++i){
        float *input = net.input_gpu + i*l.w*l.h*l.c;
        im2col_gpu(input, l.c, l.h, l.w, 
                l.size, l.stride, l.pad, net.workspace);

        for(j = 0; j < locations; ++j){ 
            float *a = l.delta_gpu + i*l.outputs + j;
            float *b = net.workspace + j;
            float *c = l.weight_updates_gpu + j*l.size*l.size*l.c*l.n;
            int m = l.n;
            int n = l.size*l.size*l.c;
            int k = 1;

            gemm_gpu(0,1,m,n,k,1,a,locations,b,locations,1,c,n);
        }

        if(net.delta_gpu){
            for(j = 0; j < locations; ++j){ 
                float *a = l.weights_gpu + j*l.size*l.size*l.c*l.n;
                float *b = l.delta_gpu + i*l.outputs + j;
                float *c = net.workspace + j;

                int m = l.size*l.size*l.c;
                int n = 1;
                int k = l.n;

                gemm_gpu(1,0,m,n,k,1,a,m,b,locations,0,c,locations);
            }

            col2im_gpu(net.workspace, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, net.delta_gpu+i*l.c*l.h*l.w);
        }
    }
}
//local层的参数更新的gpu实现
void update_local_layer_gpu(local_layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    int locations = l.out_w*l.out_h;
    int size = l.size*l.size*l.c*l.n*locations;
    axpy_gpu(l.outputs, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
    scal_gpu(l.outputs, momentum, l.bias_updates_gpu, 1);

    axpy_gpu(size, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
    axpy_gpu(size, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
    scal_gpu(size, momentum, l.weight_updates_gpu, 1);
}

void pull_local_layer(local_layer l)
{
    int locations = l.out_w*l.out_h;
    int size = l.size*l.size*l.c*l.n*locations;
    cuda_pull_array(l.weights_gpu, l.weights, size);
    cuda_pull_array(l.biases_gpu, l.biases, l.outputs);
}

void push_local_layer(local_layer l)
{
    int locations = l.out_w*l.out_h;
    int size = l.size*l.size*l.c*l.n*locations;
    cuda_push_array(l.weights_gpu, l.weights, size);
    cuda_push_array(l.biases_gpu, l.biases, l.outputs);
}
#endif
