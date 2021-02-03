#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "blas.h"
#include <stdio.h>
//创建一个BN层，分配相关内存，赋值相应的参数
layer make_batchnorm_layer(int batch, int w, int h, int c)
{
    fprintf(stderr, "Batch Normalization Layer: %d x %d x %d image\n", w,h,c);
    layer l = {0};
    l.type = BATCHNORM;
    l.batch = batch;
    l.h = l.out_h = h;
    l.w = l.out_w = w;
    l.c = l.out_c = c;
    l.output = calloc(h * w * c * batch, sizeof(float));
    l.delta  = calloc(h * w * c * batch, sizeof(float));
    l.inputs = w*h*c;
    l.outputs = l.inputs;

    l.scales = calloc(c, sizeof(float));
    l.scale_updates = calloc(c, sizeof(float));
    l.biases = calloc(c, sizeof(float));
    l.bias_updates = calloc(c, sizeof(float));
    int i;
    for(i = 0; i < c; ++i){
        l.scales[i] = 1;
    }

    l.mean = calloc(c, sizeof(float));
    l.variance = calloc(c, sizeof(float));

    l.rolling_mean = calloc(c, sizeof(float));
    l.rolling_variance = calloc(c, sizeof(float));
    //BN层的前向传播，具体实现参考src/batchnorm_layer.c
    l.forward = forward_batchnorm_layer;
    //BN层的反向传播，具体实现参考src/batchnorm_layer.c
    l.backward = backward_batchnorm_layer;
    //BN层的gpu版本
#ifdef GPU
    //BN层前向传播的gpu版本，具体实现参考src/batchnorm_layer.c
    l.forward_gpu = forward_batchnorm_layer_gpu;
    //BN层反向传播的gpu版本，具体实现参考src/batchnorm_layer.c
    l.backward_gpu = backward_batchnorm_layer_gpu;

    //给相应的数据在gpu上分配内存
    l.output_gpu =  cuda_make_array(l.output, h * w * c * batch);
    l.delta_gpu =   cuda_make_array(l.delta, h * w * c * batch);

    l.biases_gpu = cuda_make_array(l.biases, c);
    l.bias_updates_gpu = cuda_make_array(l.bias_updates, c);

    l.scales_gpu = cuda_make_array(l.scales, c);
    l.scale_updates_gpu = cuda_make_array(l.scale_updates, c);

    l.mean_gpu = cuda_make_array(l.mean, c);
    l.variance_gpu = cuda_make_array(l.variance, c);

    l.rolling_mean_gpu = cuda_make_array(l.mean, c);
    l.rolling_variance_gpu = cuda_make_array(l.variance, c);

    l.mean_delta_gpu = cuda_make_array(l.mean, c);
    l.variance_delta_gpu = cuda_make_array(l.variance, c);

    l.x_gpu = cuda_make_array(l.output, l.batch*l.outputs);
    l.x_norm_gpu = cuda_make_array(l.output, l.batch*l.outputs);
    #ifdef CUDNN
    cudnnCreateTensorDescriptor(&l.normTensorDesc);
    cudnnCreateTensorDescriptor(&l.dstTensorDesc);
    cudnnSetTensor4dDescriptor(l.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.batch, l.out_c, l.out_h, l.out_w); 
    cudnnSetTensor4dDescriptor(l.normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l.out_c, 1, 1); 

    #endif
#endif
    return l;
}
//backward_scale_cpu(l.x_norm, l.delta, l.batch, l.out_c, l.out_w*l.out_h, l.scale_updates);
void backward_scale_cpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates)
{
    int i,b,f;
    //循环遍历每一个卷积核的输出
    for(f = 0; f < n; ++f){
        float sum = 0;
        //遍历batch中每一张图片的结果
        for(b = 0; b < batch; ++b){
            for(i = 0; i < size; ++i){
                //计算相应的索引，从l.x_norm和l.delta中提取对应的值
                int index = i + size*(f + n*b);
                //累加对应的delta和x_norm的乘积
                sum += delta[index] * x_norm[index];
            }
        }
        //这里的scale_updates中最终存储的是整个batch的每一张图片在每一个输出通道上x_norm和delta乘积的和
        scale_updates[f] += sum;
    }
}
//mean_delta_cpu(l.delta, l.variance, l.batch, l.out_c, l.out_w*l.out_h, l.mean_delta);
//计算均值的delta
void mean_delta_cpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
{

    int i,j,k;
    //循环遍历每一个卷积核的处理结果
    for(i = 0; i < filters; ++i){
        //初始化存储均值delta的数组为0
        mean_delta[i] = 0;
        //遍历batch中每一张图的结果
        for (j = 0; j < batch; ++j) {
            for (k = 0; k < spatial; ++k) {
                //根据相应的索引从l.delta中提取对应的值
                int index = j*filters*spatial + i*spatial + k;
                //将结果累加到mean_delta中
                mean_delta[i] += delta[index];
            }
        }
        //计算mean_delta最终的值，mean_delta[i]表示第i个卷积核在一个batch上的均值的delta
        //(-1./sqrt(variance[i] + .00001f))就是在前向传播中根据均值和方差更新output的公式相对于均值的导数
        //这里根据这个计算并更新这一层均值的delta
        mean_delta[i] *= (-1./sqrt(variance[i] + .00001f));
    }
}
//variance_delta_cpu(l.x, l.delta, l.mean, l.variance, l.batch, l.out_c, l.out_w*l.out_h, l.variance_delta);
//计算方差的delta
void  variance_delta_cpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta)
{

    int i,j,k;
    //循环遍历没一个卷积核的处理结果
    for(i = 0; i < filters; ++i){
        //初始化存储方差delta的数组为0
        variance_delta[i] = 0;
        //循环遍历batch中每一张图片的结果
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                //根据相应的索引取出l.detla中的值
                int index = j*filters*spatial + i*spatial + k;
                //将相应的结果累加到variance_delta中
                //这里这一步和最后那一步完成了反向传播中方差的delta的计算
                variance_delta[i] += delta[index]*(x[index] - mean[i]);
            }
        }
        //计算最终方差的delta这里的-.5 * pow(variance[i] + .00001f, (float)(-3./2.))
        //通过前向传播中均值和方差更新l.x的过程可以推出来
        //最终variance_detla中存储的是不通卷积核作用下的方差的delta
        variance_delta[i] *= -.5 * pow(variance[i] + .00001f, (float)(-3./2.));
    }
}
//normalize_delta_cpu(l.x, l.mean, l.variance, l.mean_delta, l.variance_delta, l.batch, l.out_c, l.out_w*l.out_h, l.delta);
//计算BN层中l.x的delta具体参考BN反向传播的推导
void normalize_delta_cpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta)
{
    int f, j, k;
    //循环遍历batch中的每一张图片的结果
    for(j = 0; j < batch; ++j){
        //循环遍历每一个卷积核的处理结果
        for(f = 0; f < filters; ++f){
            for(k = 0; k < spatial; ++k){
                //根据相应的做标提取数值，计算相关的delta
                int index = j*filters*spatial + f*spatial + k;
                delta[index] = delta[index] * 1./(sqrt(variance[f] + .00001f)) + variance_delta[f] * 2. * (x[index] - mean[f]) / (spatial * batch) + mean_delta[f]/(spatial*batch);
            }
        }
    }
}

void resize_batchnorm_layer(layer *layer, int w, int h)
{
    fprintf(stderr, "Not implemented\n");
}
//进行batchnormalize操作
void forward_batchnorm_layer(layer l, network net)
{
    //如果本身就是batchnorm层的话。。。。
    //copy_cpu的具体实现参考src/blas.c
    if(l.type == BATCHNORM) copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
    //copy的具体实现参考src/blas.c
    //这里实现的操作是将l.output里面存储的值复制了一份到l.x里面
    copy_cpu(l.outputs*l.batch, l.output, 1, l.x, 1);
    //训练过程中
    if(net.train){
        //mean_cpu的具体实现参考src/blas.c
        //将一个batch中的每一张图片对应通道的均值计算出来，存储在l.mean中
        mean_cpu(l.output, l.batch, l.out_c, l.out_h*l.out_w, l.mean);
        //variance_cpu具体实现参考src/blas.c
        //跟上面的均值计算对应，计算方差，存储到l.variance中
        variance_cpu(l.output, l.mean, l.batch, l.out_c, l.out_h*l.out_w, l.variance);
        //scal_cpu具体实现参考src/blas.c
        //这里的l.rooling_mean在make_batchnorm_layer中分配内存，初始化为0。。
        //通过scal_cpu实现的操作是将l.rolling_mean中的每一个值乘上0.99
        scal_cpu(l.out_c, .99, l.rolling_mean, 1);
        //axpy_cpu的具体实现参考src/blas.c
        //通过aspy_cpu实现的是将l.mean中的每一个值乘上0.01加到对应的l.rolling_mean里面，并存储进l.rollmean_ing中
        axpy_cpu(l.out_c, .01, l.mean, 1, l.rolling_mean, 1);
        //这里的两步操作参考上面的均值的操作，实际上是一致的
        //训练过程中利用指数加权平均的方式累积均值和方差。。。。
        scal_cpu(l.out_c, .99, l.rolling_variance, 1);
        axpy_cpu(l.out_c, .01, l.variance, 1, l.rolling_variance, 1);
        //nromalize_cpu的具体实现参考src/blas.c
        //根据上面的均值和方差更新l.output中的数据
        normalize_cpu(l.output, l.mean, l.variance, l.batch, l.out_c, l.out_h*l.out_w);  
        //copy_cpu具体实现参考src/blas.c
        //这里是将经过均值和方差调整之后的数据复制一份到l.x_norm里面了
        //跟一开始将l.output中的数据复制进l.x对应 
        copy_cpu(l.outputs*l.batch, l.output, 1, l.x_norm, 1);
        //如果不是在训练，也就是推理的时候
    } else {
        //从这里对照上面的过程可以得知，l.rolling_mean和l.rolling_variance的作用就是累积每一次的均值和方差，作为最终推理的时候来使用
        normalize_cpu(l.output, l.rolling_mean, l.rolling_variance, l.batch, l.out_c, l.out_h*l.out_w);
    }
    //scale_bias的具体实现参考src/convolutional_layer.c
    //这里的l.sacles同样在make_batchnorm_layer中分配内存，并初始化为1
    //这里具体的操作是将l.output中每一个值乘上l.scales中对应的值并跟新到l.output中
    scale_bias(l.output, l.scales, l.batch, l.out_c, l.out_h*l.out_w);
    //add_bias的具体实现参考src/convolutional_layer.c
    //这里的l.biases同样在make_batchnorm_layer中分配内存，并初始化为0
    //具体实现的内容是将l.output中的值与l.biases中的值相加，并跟新到l.output中
    add_bias(l.output, l.biases, l.batch, l.out_c, l.out_h*l.out_w);
}
//batchnorm_layer的反向传播
//可以参考这个推导过程https://blog.csdn.net/qq_28778507/article/details/84570153
//整个BN的反向传播实际上作用在scale,bias和l.x上了
void backward_batchnorm_layer(layer l, network net)
{
    //如果不在训练
    if(!net.train){
        //更新l.mean和l.variance的值为l.rolling_mean和l.rolling_variance
        l.mean = l.rolling_mean;
        l.variance = l.rolling_variance;
    }
    //具体实现参考src/convolutional_layer.c
    //计算当前层次偏置的更新值，存储到l.bias_updates中
    backward_bias(l.bias_updates, l.delta, l.batch, l.out_c, l.out_w*l.out_h);
    //backward_scale_cpu的具体实现参考src/batchnorm_layer.c
    //计算当前层的l.scale的更新值存储到l.scale_updates中，关于l.scale的作用实际上就是BN层的γ，而上面的BN层的bias就是其中的beta
    //这里计算scale的更新用的是l.x_norm，从前向传播就能看出。。前向传播中实际上就是l.x_norm最终和l.scale相乘并更新了output
    //l.x_norm就是从那个时候存储一份下来的，用来在这里计算scaled的更新delta
    //根据BN层反向传播的推导。。这里计算的是scale的更新值
    backward_scale_cpu(l.x_norm, l.delta, l.batch, l.out_c, l.out_w*l.out_h, l.scale_updates);
    //这里的scale_bias的作用是将l.scale和l.delta对应的数值相乘，更新到l.delta中
    //目的是为了后面的均值和方差的delta的计算，具体参照BN反向传播的推导
    scale_bias(l.delta, l.scales, l.batch, l.out_c, l.out_h*l.out_w);
    //计算当前层均值的delta
    //具体实现参考src/batchnorm_layer.c
    mean_delta_cpu(l.delta, l.variance, l.batch, l.out_c, l.out_w*l.out_h, l.mean_delta);
    //计算当前BN层的方差的delta
    //具体实现参考src/batchnorm_layer.c
    //注意这里传入的是l.x也就是没有经过均值方差调整过之前的数据，前向传播过程中有一个备份
    //具体的细节参考BN反向传播的推理过程
    variance_delta_cpu(l.x, l.delta, l.mean, l.variance, l.batch, l.out_c, l.out_w*l.out_h, l.variance_delta);
    //normalize_delta_cpu的具体实现参考src/batchnorm_layer.c
    //计算BN层中l.x的delta
    normalize_delta_cpu(l.x, l.mean, l.variance, l.mean_delta, l.variance_delta, l.batch, l.out_c, l.out_w*l.out_h, l.delta);
    //如果本身是batchnorm层。。
    //这里实现的操作是将l.delta复制一份到net.delta中
    if(l.type == BATCHNORM) copy_cpu(l.outputs*l.batch, l.delta, 1, net.delta, 1);
}

#ifdef GPU

void pull_batchnorm_layer(layer l)
{
    cuda_pull_array(l.scales_gpu, l.scales, l.c);
    cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.c);
    cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.c);
}
void push_batchnorm_layer(layer l)
{
    cuda_push_array(l.scales_gpu, l.scales, l.c);
    cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.c);
    cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.c);
}
//BN层前向传播的gpu版本
void forward_batchnorm_layer_gpu(layer l, network net)
{   
    //如果本身就是batchnorm层，copy_gpu的实现参考src/blas_kernels.cu
    //这里实现的是将net.input_gpu中的值赋值到l.output_gpu中去
    if(l.type == BATCHNORM) copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
    //这里是将l.output_gpu中的值赋值到l.x_gpu中
    copy_gpu(l.outputs*l.batch, l.output_gpu, 1, l.x_gpu, 1);
    //如果是训练过程中
    if (net.train) {
#ifdef CUDNN
        float one = 1;
        float zero = 0;
        cudnnBatchNormalizationForwardTraining(cudnn_handle(),
                CUDNN_BATCHNORM_SPATIAL,
                &one,
                &zero,
                l.dstTensorDesc,
                l.x_gpu,
                l.dstTensorDesc,
                l.output_gpu,
                l.normTensorDesc,
                l.scales_gpu,
                l.biases_gpu,
                .01,
                l.rolling_mean_gpu,
                l.rolling_variance_gpu,
                .00001,
                l.mean_gpu,
                l.variance_gpu);
#else
        //fast_mean_gpu的具体实现参考src/blas_kernels.cu
        //将一个batch中的每一张图片对应通道的均值计算出来，存储在l.mean_gpu中
        fast_mean_gpu(l.output_gpu, l.batch, l.out_c, l.out_h*l.out_w, l.mean_gpu);
        //fast_variance_gpu的具体实现参考src/blas_kernels.cu
        //和上面的均值计算对应起来计算相应的方差，结果写入l.variance_gpu
        fast_variance_gpu(l.output_gpu, l.mean_gpu, l.batch, l.out_c, l.out_h*l.out_w, l.variance_gpu);
        //scal_gpu的具体实现参考src/blas_kernels.c
        //通过scal_gpu实现的操作是将l.rolling_mean中的每一个值乘上0.99
        scal_gpu(l.out_c, .99, l.rolling_mean_gpu, 1);
        //axpy_gpu的具体实现参考src/blas_kernels.cu
        //将l.mean_gpu中的值乘上0.01在加到l.rollong_mean_gpu中去
        axpy_gpu(l.out_c, .01, l.mean_gpu, 1, l.rolling_mean_gpu, 1);
        //这里实现的是将l.rolling_variance_gpu中的每一个值乘上0.99
        scal_gpu(l.out_c, .99, l.rolling_variance_gpu, 1);
        //将l.variance_gpu中的每一个值乘上0.01加到l.rolling_variance_gpu上去
        axpy_gpu(l.out_c, .01, l.variance_gpu, 1, l.rolling_variance_gpu, 1);
        //将l.output_gpu中的值赋值到l.x_gpu中去
        copy_gpu(l.outputs*l.batch, l.output_gpu, 1, l.x_gpu, 1);
        //normalize_gpu的具体实现参考src/blas_kernels.cu
        //根据相应的均值和方差对数据做归一化
        normalize_gpu(l.output_gpu, l.mean_gpu, l.variance_gpu, l.batch, l.out_c, l.out_h*l.out_w);
        //将l.output_gpu中的值赋值到l.x_norm_gpu中去
        copy_gpu(l.outputs*l.batch, l.output_gpu, 1, l.x_norm_gpu, 1);
        //scale_bias_gpu的具体实现参考src/blas_kernels.cu
        //将l.output_gpu中的值跟l.scales_gpu中的值相乘，结果写入l.output_gpu中
        scale_bias_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);
        //add_bias_gpu的具体实现c参考src/blas_kernels.cu
        //实现的是将l.biases_gpu中的值加到l.output_gpu中去
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.out_c, l.out_w*l.out_h);
#endif
    } else {
        //如果不在训练的化
        //直接根据相应的均值和方差对数据进行归一化
        normalize_gpu(l.output_gpu, l.rolling_mean_gpu, l.rolling_variance_gpu, l.batch, l.out_c, l.out_h*l.out_w);
        //将l.output中的值乘上l.scales_gpu中的值，结果写入l.output_gpu
        scale_bias_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);
        //将l.biases_gpu中的值加到l.output_gpu上去
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.out_c, l.out_w*l.out_h);
    }

}

//BN层的反向传播
void backward_batchnorm_layer_gpu(layer l, network net)
{
    if(!net.train){
        //不在训练的化。。直接更新相应均值和方差的值
        l.mean_gpu = l.rolling_mean_gpu;
        l.variance_gpu = l.rolling_variance_gpu;
    }
#ifdef CUDNN
    float one = 1;
    float zero = 0;
    cudnnBatchNormalizationBackward(cudnn_handle(),
            CUDNN_BATCHNORM_SPATIAL,
            &one,
            &zero,
            &one,
            &one,
            l.dstTensorDesc,
            l.x_gpu,
            l.dstTensorDesc,
            l.delta_gpu,
            l.dstTensorDesc,
            l.x_norm_gpu,
            l.normTensorDesc,
            l.scales_gpu,
            l.scale_updates_gpu,
            l.bias_updates_gpu,
            .00001,
            l.mean_gpu,
            l.variance_gpu);
    copy_gpu(l.outputs*l.batch, l.x_norm_gpu, 1, l.delta_gpu, 1);
#else
    //如果在训练，进行相应的反向传播
    //backward_bias_gpu的具体实现参考src/blas_kernels.cu
    //计算相应的偏置更新值，结果写入l.bias_updates_gpu
    backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.out_c, l.out_w*l.out_h);
    //backwd_scale_gpu的具体实现参考src/blas_kernels.cu
    //计算相应的scale的更新值，结果写入l.scale_updates_gpu中
    backward_scale_gpu(l.x_norm_gpu, l.delta_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.scale_updates_gpu);
    //scale_bias_gpu的具体实现参考src/blas_kernels.cu
    //这里实现的是将l.delta_gpu中的值和l.scales_gpu中的值相乘，结果写入l.delta_gpu,完成相应的delta的更新
    scale_bias_gpu(l.delta_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);
    //fast_mean_delta_gpu的具体实现参考src/blas_kernels.cu
    //计算当前BN层相应均值的delta,结果写入l.mean_delta_gpu中
    fast_mean_delta_gpu(l.delta_gpu, l.variance_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.mean_delta_gpu);
    //fast_variance_delta_gpu的具体实现参考src/blas_kernels.cu
    //计算当前BN层的方差的delta，结果写入l.variance_delta_gpu中
    fast_variance_delta_gpu(l.x_gpu, l.delta_gpu, l.mean_gpu, l.variance_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.variance_delta_gpu);
    //计算BN层中的l.x_gpu的delta，结果写入l.delta_gpu
    //normalize_delta_gpu的具体实现参考src/blas_kernels.cu
    normalize_delta_gpu(l.x_gpu, l.mean_gpu, l.variance_gpu, l.mean_delta_gpu, l.variance_delta_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.delta_gpu);
#endif
    //如果本身就是BATCHNORM层，将l.delta_gpu中的值赋值一份到net.delta_gpu中，完成delta的传递
    if(l.type == BATCHNORM) copy_gpu(l.outputs*l.batch, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif
