#include "deconvolutional_layer.h"
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "utils.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"

#include <stdio.h>
#include <time.h>


static size_t get_workspace_size(layer l){
    return (size_t)l.h*l.w*l.size*l.size*l.n*sizeof(float);
}

void bilinear_init(layer l)
{
    int i,j,f;
    float center = (l.size-1) / 2.;
    for(f = 0; f < l.n; ++f){
        for(j = 0; j < l.size; ++j){
            for(i = 0; i < l.size; ++i){
                float val = (1 - fabs(i - center)) * (1 - fabs(j - center));
                int c = f%l.c;
                int ind = f*l.size*l.size*l.c + c*l.size*l.size + j*l.size + i;
                l.weights[ind] = val;
            }
        }
    }
}

//创建一个反卷积层
layer make_deconvolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int adam)
{
    int i;
    layer l = {0};
    l.type = DECONVOLUTIONAL;

    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.batch = batch;
    l.stride = stride;
    l.size = size;
    //卷积核参数量
    l.nweights = c*n*size*size;
    //偏置参数量
    l.nbiases = n;

    l.weights = calloc(c*n*size*size, sizeof(float));
    l.weight_updates = calloc(c*n*size*size, sizeof(float));

    l.biases = calloc(n, sizeof(float));
    l.bias_updates = calloc(n, sizeof(float));
    //float scale = n/(size*size*c);
    //printf("scale: %f\n", scale);
    float scale = .02;
    //随机初始化权重
    for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_normal();
    //bilinear_init(l);
    //偏置初始化为0
    for(i = 0; i < n; ++i){
        l.biases[i] = 0;
    }
    l.pad = padding;
    //计算输出的尺度，跟卷积输出尺度的计算反过来就行了。。
    l.out_h = (l.h - 1) * l.stride + l.size - 2*l.pad;
    l.out_w = (l.w - 1) * l.stride + l.size - 2*l.pad;
    //输出通道数等于卷积核个数。。。
    l.out_c = n;
    l.outputs = l.out_w * l.out_h * l.out_c;
    l.inputs = l.w * l.h * l.c;
    //具体实现参考src/blas.c
    //将权重的值乘上一个比例值，l.out_w*l.out_h/(l.w*l.h)
    scal_cpu(l.nweights, (float)l.out_w*l.out_h/(l.w*l.h), l.weights, 1);

    l.output = calloc(l.batch*l.outputs, sizeof(float));
    l.delta  = calloc(l.batch*l.outputs, sizeof(float));
    //反卷积层的前向传播，具体细节参考src/deconvolutional_layer.c
    l.forward = forward_deconvolutional_layer;
    //反卷积层的反向传播，具体细节参考src/deconvolutional_layer.c
    l.backward = backward_deconvolutional_layer;
    //反卷积层的参数更新，具体细节参考src/deconvolutional_layer.c
    l.update = update_deconvolutional_layer;

    l.batch_normalize = batch_normalize;

    if(batch_normalize){
        l.scales = calloc(n, sizeof(float));
        l.scale_updates = calloc(n, sizeof(float));
        for(i = 0; i < n; ++i){
            l.scales[i] = 1;
        }

        l.mean = calloc(n, sizeof(float));
        l.variance = calloc(n, sizeof(float));

        l.mean_delta = calloc(n, sizeof(float));
        l.variance_delta = calloc(n, sizeof(float));

        l.rolling_mean = calloc(n, sizeof(float));
        l.rolling_variance = calloc(n, sizeof(float));
        l.x = calloc(l.batch*l.outputs, sizeof(float));
        l.x_norm = calloc(l.batch*l.outputs, sizeof(float));
    }
    if(adam){
        l.m = calloc(c*n*size*size, sizeof(float));
        l.v = calloc(c*n*size*size, sizeof(float));
        l.bias_m = calloc(n, sizeof(float));
        l.scale_m = calloc(n, sizeof(float));
        l.bias_v = calloc(n, sizeof(float));
        l.scale_v = calloc(n, sizeof(float));
    }

#ifdef GPU
    //deconvolutional_layer的gpu版本
    //gpu版本的转置卷积的前向传播，具体实现参考src/deconvolutional_kernels.cu
    l.forward_gpu = forward_deconvolutional_layer_gpu;
    //gpu版本的转置卷积的前向传播，具体实现参考src/deconvolutional_kernels.cu
    l.backward_gpu = backward_deconvolutional_layer_gpu;
    //gpu版本的转置卷积的参数更新，具体实现参考src/deconvolutional_kernels.cu
    l.update_gpu = update_deconvolutional_layer_gpu;

    if(gpu_index >= 0){

        if (adam) {
            l.m_gpu = cuda_make_array(l.m, c*n*size*size);
            l.v_gpu = cuda_make_array(l.v, c*n*size*size);
            l.bias_m_gpu = cuda_make_array(l.bias_m, n);
            l.bias_v_gpu = cuda_make_array(l.bias_v, n);
            l.scale_m_gpu = cuda_make_array(l.scale_m, n);
            l.scale_v_gpu = cuda_make_array(l.scale_v, n);
        }
        l.weights_gpu = cuda_make_array(l.weights, c*n*size*size);
        l.weight_updates_gpu = cuda_make_array(l.weight_updates, c*n*size*size);

        l.biases_gpu = cuda_make_array(l.biases, n);
        l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);

        l.delta_gpu = cuda_make_array(l.delta, l.batch*l.out_h*l.out_w*n);
        l.output_gpu = cuda_make_array(l.output, l.batch*l.out_h*l.out_w*n);

        if(batch_normalize){
            l.mean_gpu = cuda_make_array(0, n);
            l.variance_gpu = cuda_make_array(0, n);

            l.rolling_mean_gpu = cuda_make_array(0, n);
            l.rolling_variance_gpu = cuda_make_array(0, n);

            l.mean_delta_gpu = cuda_make_array(0, n);
            l.variance_delta_gpu = cuda_make_array(0, n);

            l.scales_gpu = cuda_make_array(l.scales, n);
            l.scale_updates_gpu = cuda_make_array(0, n);

            l.x_gpu = cuda_make_array(0, l.batch*l.out_h*l.out_w*n);
            l.x_norm_gpu = cuda_make_array(0, l.batch*l.out_h*l.out_w*n);
        }
    }
    #ifdef CUDNN
        cudnnCreateTensorDescriptor(&l.dstTensorDesc);
        cudnnCreateTensorDescriptor(&l.normTensorDesc);
        cudnnSetTensor4dDescriptor(l.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.batch, l.out_c, l.out_h, l.out_w); 
        cudnnSetTensor4dDescriptor(l.normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l.out_c, 1, 1); 
    #endif
#endif

    l.activation = activation;
    l.workspace_size = get_workspace_size(l);

    fprintf(stderr, "deconv%5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);

    return l;
}

void denormalize_deconvolutional_layer(layer l)
{
    int i, j;
    for(i = 0; i < l.n; ++i){
        float scale = l.scales[i]/sqrt(l.rolling_variance[i] + .00001);
        for(j = 0; j < l.c*l.size*l.size; ++j){
            l.weights[i*l.c*l.size*l.size + j] *= scale;
        }
        l.biases[i] -= l.rolling_mean[i] * scale;
        l.scales[i] = 1;
        l.rolling_mean[i] = 0;
        l.rolling_variance[i] = 1;
    }
}

void resize_deconvolutional_layer(layer *l, int h, int w)
{
    l->h = h;
    l->w = w;
    l->out_h = (l->h - 1) * l->stride + l->size - 2*l->pad;
    l->out_w = (l->w - 1) * l->stride + l->size - 2*l->pad;

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->w * l->h * l->c;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta  = realloc(l->delta,  l->batch*l->outputs*sizeof(float));
    if(l->batch_normalize){
        l->x = realloc(l->x, l->batch*l->outputs*sizeof(float));
        l->x_norm  = realloc(l->x_norm, l->batch*l->outputs*sizeof(float));
    }

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =  cuda_make_array(l->delta,  l->batch*l->outputs);
    l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);

    if(l->batch_normalize){
        cuda_free(l->x_gpu);
        cuda_free(l->x_norm_gpu);

        l->x_gpu = cuda_make_array(l->output, l->batch*l->outputs);
        l->x_norm_gpu = cuda_make_array(l->output, l->batch*l->outputs);
    }
    #ifdef CUDNN
        cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 
        cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1); 
    #endif
#endif
    l->workspace_size = get_workspace_size(*l);
}
//反卷积层的前向传播
void forward_deconvolutional_layer(const layer l, network net)
{
    int i;
    //所有卷积核在一个通道上的参数量
    int m = l.size*l.size*l.n;
    //输入数据一个通道的参数量
    int n = l.h*l.w;
    //输入数据的通道数
    int k = l.c;
    //这里实现的是将l.output初始化成0
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);
    //循环遍历batch中的每一张图的结果
    for(i = 0; i < l.batch; ++i){
        //定位权重存储的位置
        float *a = l.weights;
        //定位某一张图片特征图数据的存储位置
        float *b = net.input + i*l.c*l.h*l.w;
        //定位工作空间的位置
        float *c = net.workspace;
        //这里调用的是gemm_tn函数
        //实现a的转置和b相乘，结果写入c中
        //结合卷积层的前向传播的实现来看，其实很简单。。画个图推一下就出来了
        //首先卷积假如是Y = W*X的话，反卷积或者说转置卷积就是X_t = W^T*Y也就是W的转置和Y做矩阵乘法，X和X_t不一样。。只是维度相同
        //卷积操作中，一张输入图片先展开成一个矩阵，然后和权重矩阵做矩阵乘法得到输出。。。
        //这里转置卷积的操作也就是W^T*Y就是来恢复到图片展开的矩阵的维度。。。,然后再利用下面的col2im_cpu将矩阵恢复成特征图的形式
        gemm_cpu(1,0,m,n,k,1,a,m,b,n,0,c,n);
        //将矩阵恢复成特征图的形式，细节参考src/col2im.c
        col2im_cpu(net.workspace, l.out_c, l.out_h, l.out_w, l.size, l.stride, l.pad, l.output+i*l.outputs);
    }
    if (l.batch_normalize) {
        //如果存在batchnormalize
        //进行batchnorm的前向传播，具体实现参考src/;batchnorm_layer.c
        forward_batchnorm_layer(l, net);
    } else {
        //增加偏置，具体实现参考src/convolutional_layer.c
        add_bias(l.output, l.biases, l.batch, l.n, l.out_w*l.out_h);
    }
    //激活函数。。。具体实现参考src/activate_array.c
    activate_array(l.output, l.batch*l.n*l.out_w*l.out_h, l.activation);
}
//转置卷积的反向传播
void backward_deconvolutional_layer(layer l, network net)
{
    int i;
    //计算激活函数的导数值，将结果跟l.delta中的值相乘，然后写入l.delta中。。。
    //具体实现参考src/activations.c
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);
    //如果进行batchnormalize
    if(l.batch_normalize){
        //进行BN层的反向传播。。。具体实现参考src/batchnorm_layer.c
        backward_batchnorm_layer(l, net);
    } else {
        //计算偏置的更新值，具体实现参考src/convolutional_layer.c
        backward_bias(l.bias_updates, l.delta, l.batch, l.n, l.out_w*l.out_h);
    }

    //if(net.delta) memset(net.delta, 0, l.batch*l.h*l.w*l.c*sizeof(float));
    //遍历batch中给张图片的数据
    for(i = 0; i < l.batch; ++i){
        int m = l.c;
        int n = l.size*l.size*l.n;
        int k = l.h*l.w;
        //a定位net.input输入的位置
        float *a = net.input + i*m*k;
        //b定位工作空间
        float *b = net.workspace;
        //c定位存储权重更新的值的位置
        float *c = l.weight_updates;
        //实际上时跟卷积的前向传播对应的，从转置卷积跟卷积的关系推导出来了
        //将l.delta展开成矩阵
        im2col_cpu(l.delta + i*l.outputs, l.out_c, l.out_h, l.out_w, 
                l.size, l.stride, l.pad, b);
        //进行矩阵计算，具体方式参考转置卷积的反向传播的推导
        gemm_cpu(0,1,m,n,k,1,a,k,b,k,1,c,n);
        //如果存在上一层，开始l.delta的传递
        if(net.delta){
            int m = l.c;
            int n = l.h*l.w;
            int k = l.size*l.size*l.n;

            float *a = l.weights;
            float *b = net.workspace;
            float *c = net.delta + i*n*m;
            //这里同样可以参考转置卷积的反向传播推导
            //结合卷积公式一起推导，就能比较方便的看出来了
            gemm_cpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }
}
//反卷积层的参数更新
void update_deconvolutional_layer(layer l, update_args a)
{   //获取学习率，衰减系数，动量等参数
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    int size = l.size*l.size*l.c*l.n;
    //更新偏置并积累动量
    axpy_cpu(l.n, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.n, momentum, l.bias_updates, 1);

    if(l.scales){
        //更新BN层的系数和动量
        axpy_cpu(l.n, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.n, momentum, l.scale_updates, 1);
    }
    //计算权重衰减，更新权重和动量
    axpy_cpu(size, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(size, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(size, momentum, l.weight_updates, 1);
}



