#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "convolutional_layer.h"
#include "deconvolutional_layer.h"
#include "batchnorm_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "cuda.h"
}
//gpu版本的转置卷积的前向传播
extern "C" void forward_deconvolutional_layer_gpu(layer l, network net)
{
    int i;
    //m表示所有卷积核在一个通道上的参数量
    int m = l.size*l.size*l.n;
    //n表示输出特征图一个通道上的参数量
    int n = l.h*l.w;
    //k表示输入通道数
    int k = l.c;
    //这里实现的是将l.output_gpu位置上的数用0初始化
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    //循环遍历batch中每一张图片的产生的数据
    for(i = 0; i < l.batch; ++i){
        //定位weights_gpu和input_gpu以及工作空间的位置
        float *a = l.weights_gpu;
        float *b = net.input_gpu + i*l.c*l.h*l.w;
        float *c = net.workspace;
        //调用gemm_gpu函数实现转置卷积的计算，这里实现的是weights的转置和input的乘积
        //gemm_gpu的具体实现参考src/gemm.c
        gemm_gpu(1,0,m,n,k,1,a,m,b,n,0,c,n);
        //将workspace中的结果从矩阵的形式转换到特征图的形式
        //col2im_gpu的具体实现参考src/col2im_kernels.cu
        col2im_gpu(net.workspace, l.out_c, l.out_h, l.out_w, l.size, l.stride, l.pad, l.output_gpu+i*l.outputs);
    }
    //如果需要进行bn操作的化
    if (l.batch_normalize) {
        //进行bn层gpu版本的前向传播
        //forward_batchnorm_layer_gpu的具体实现参考src/batchnorm_layer.c
        forward_batchnorm_layer_gpu(l, net);
    } else {
        //不进行bn操作的化,将偏置加到相应的结果上
        //add_bias_gpu的具体实现参考src/bias_kernels.cu
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
    }
    //通过相应的激活函数,activate_array_gpu的具体实现参考src/activation_kernels.cu
    activate_array_gpu(l.output_gpu, l.batch*l.n*l.out_w*l.out_h, l.activation);
}

//gpu版本的转置卷积的反向传播
extern "C" void backward_deconvolutional_layer_gpu(layer l, network net)
{
    int i;

    //constrain_gpu(l.outputs*l.batch, 1, l.delta_gpu, 1);
    //计算激活函数相对于output_gpu的梯度，结果乘到l.delta_gpu上
    //具体实现参考src/activation_kernels.cu
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);

    //如果需要bn
    if(l.batch_normalize){
    //进行gpu版本的bn的反向传播
        backward_batchnorm_layer_gpu(l, net);
    } else {
        //否则计算偏置的更新，backward_bias_gpu的具体实现参考src/bias_kernels.cu
        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);
    }

    //if(net.delta_gpu) memset(net.delta_gpu, 0, l.batch*l.h*l.w*l.c*sizeof(float));

    //遍历batch中每一张图片产生的数据
    for(i = 0; i < l.batch; ++i){

        int m = l.c;
        int n = l.size*l.size*l.n;
        int k = l.h*l.w;
        //定位intput_gpu等等的相应位置
        float *a = net.input_gpu + i*m*k;
        float *b = net.workspace;
        float *c = l.weight_updates_gpu;
        //将delta_gpu的相关部分展开成一个矩阵，写入workspace
        //具体参考转置卷积的反向传播的推导过程
        im2col_gpu(l.delta_gpu + i*l.outputs, l.out_c, l.out_h, l.out_w, 
                l.size, l.stride, l.pad, b);
        //调用gemm函数实现相应的矩阵计算，将weight_update写入相应位置
        gemm_gpu(0,1,m,n,k,1,a,k,b,k,1,c,n);

        //接下来完成net.delta的计算和传递
        if(net.delta_gpu){
            int m = l.c;
            int n = l.h*l.w;
            int k = l.size*l.size*l.n;
            //定位相应的权重等位置
            float *a = l.weights_gpu;
            float *b = net.workspace;
            float *c = net.delta_gpu + i*n*m;
            //调用gemm完成相应的net.delta_gpu的计算
            gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }
}

extern "C" void pull_deconvolutional_layer(layer l)
{
    cuda_pull_array(l.weights_gpu, l.weights, l.c*l.n*l.size*l.size);
    cuda_pull_array(l.biases_gpu, l.biases, l.n);
    cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.c*l.n*l.size*l.size);
    cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
    if (l.batch_normalize){
        cuda_pull_array(l.scales_gpu, l.scales, l.n);
        cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}

extern "C" void push_deconvolutional_layer(layer l)
{
    cuda_push_array(l.weights_gpu, l.weights, l.c*l.n*l.size*l.size);
    cuda_push_array(l.biases_gpu, l.biases, l.n);
    cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.c*l.n*l.size*l.size);
    cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
    if (l.batch_normalize){
        cuda_push_array(l.scales_gpu, l.scales, l.n);
        cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}

//gpu版本的转置卷积的参数更新
void update_deconvolutional_layer_gpu(layer l, update_args a)
{
    //获得当前的学习率权重衰减比例以及动量值
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;
    //如果是adam优化
    if(a.adam){
        //利用adam更新权重和偏置
        //adam_update_gpu的具体实现参考src/blas_kernels.cu
        adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.nweights, batch, a.t);
        adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
        //如果需要scales操作
        if(l.scales_gpu){
        //再一次更新scales的值
            adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
        }
    }else{
        //不采用adam优化的化
        //就利用动量梯度下降进行参数更新
        //先计算权重的衰减
        axpy_gpu(l.nweights, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
        //根据学习率更新权重
        axpy_gpu(l.nweights, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
        //累积相应的动量
        scal_gpu(l.nweights, momentum, l.weight_updates_gpu, 1);
        //更新偏置
        axpy_gpu(l.n, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
        //累积动量
        scal_gpu(l.n, momentum, l.bias_updates_gpu, 1);
        //如果存在scales
        if(l.scales_gpu){
        //同样更新scales的相关内容
            axpy_gpu(l.n, learning_rate/batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
            scal_gpu(l.n, momentum, l.scale_updates_gpu, 1);
        }
    }
}

