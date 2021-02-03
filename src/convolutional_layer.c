#include "convolutional_layer.h"
#include "utils.h"
#include "batchnorm_layer.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>

#ifdef AI2
#include "xnor_layer.h"
#endif

void swap_binary(convolutional_layer *l)
{
    float *swap = l->weights;
    l->weights = l->binary_weights;
    l->binary_weights = swap;

#ifdef GPU
    swap = l->weights_gpu;
    l->weights_gpu = l->binary_weights_gpu;
    l->binary_weights_gpu = swap;
#endif
}

void binarize_weights(float *weights, int n, int size, float *binary)
{
    int i, f;
    for(f = 0; f < n; ++f){
        float mean = 0;
        for(i = 0; i < size; ++i){
            mean += fabs(weights[f*size + i]);
        }
        mean = mean / size;
        for(i = 0; i < size; ++i){
            binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        }
    }
}

void binarize_cpu(float *input, int n, float *binary)
{
    int i;
    for(i = 0; i < n; ++i){
        binary[i] = (input[i] > 0) ? 1 : -1;
    }
}

void binarize_input(float *input, int n, int size, float *binary)
{
    int i, s;
    for(s = 0; s < size; ++s){
        float mean = 0;
        for(i = 0; i < n; ++i){
            mean += fabs(input[i*size + s]);
        }
        mean = mean / n;
        for(i = 0; i < n; ++i){
            binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
        }
    }
}
//计算卷积层输出特征图的height
int convolutional_out_height(convolutional_layer l)
{
    return (l.h + 2*l.pad - l.size) / l.stride + 1;
}
//计算卷积层输出特征图的width
int convolutional_out_width(convolutional_layer l)
{
    //就是计算输出的公式。。。
    return (l.w + 2*l.pad - l.size) / l.stride + 1;
}

image get_convolutional_image(convolutional_layer l)
{
    return float_to_image(l.out_w,l.out_h,l.out_c,l.output);
}

image get_convolutional_delta(convolutional_layer l)
{
    return float_to_image(l.out_w,l.out_h,l.out_c,l.delta);
}
//返回当前层l需要的工作空间的尺度
static size_t get_workspace_size(layer l){
#ifdef CUDNN
    if(gpu_index >= 0){
        size_t most = 0;
        size_t s = 0;
        cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(),
                l.srcTensorDesc,
                l.weightDesc,
                l.convDesc,
                l.dstTensorDesc,
                l.fw_algo,
                &s);
        if (s > most) most = s;
        cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle(),
                l.srcTensorDesc,
                l.ddstTensorDesc,
                l.convDesc,
                l.dweightDesc,
                l.bf_algo,
                &s);
        if (s > most) most = s;
        cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle(),
                l.weightDesc,
                l.ddstTensorDesc,
                l.convDesc,
                l.dsrcTensorDesc,
                l.bd_algo,
                &s);
        if (s > most) most = s;
        return most;
    }
#endif
    return (size_t)l.out_h*l.out_w*l.size*l.size*l.c/l.groups*sizeof(float);
}

#ifdef GPU
#ifdef CUDNN
void cudnn_convolutional_setup(layer *l)
{
    cudnnSetTensor4dDescriptor(l->dsrcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w); 
    cudnnSetTensor4dDescriptor(l->ddstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 

    cudnnSetTensor4dDescriptor(l->srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w); 
    cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 
    cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1); 

    cudnnSetFilter4dDescriptor(l->dweightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, l->n, l->c/l->groups, l->size, l->size); 
    cudnnSetFilter4dDescriptor(l->weightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, l->n, l->c/l->groups, l->size, l->size); 
    #if CUDNN_MAJOR >= 6
    cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    #else
    cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION);
    #endif

    #if CUDNN_MAJOR >= 7
    cudnnSetConvolutionGroupCount(l->convDesc, l->groups);
    #else
    if(l->groups > 1){
        error("CUDNN < 7 doesn't support groups, please upgrade!");
    }
    #endif

    cudnnGetConvolutionForwardAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->weightDesc,
            l->convDesc,
            l->dstTensorDesc,
            CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
            2000000000,
            &l->fw_algo);
    cudnnGetConvolutionBackwardDataAlgorithm(cudnn_handle(),
            l->weightDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dsrcTensorDesc,
            CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
            2000000000,
            &l->bd_algo);
    cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dweightDesc,
            CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
            2000000000,
            &l->bf_algo);
}
#endif
#endif

//对某一个卷积层需要的内存进行分配，为少量参数赋值。。。大部分没有赋值。。只是根据需求分配内存
//make_convolutional_layer(batch,h,w,c,n,groups,size,stride,padding,activation, batch_normalize, binary, xnor, params.net->adam);
//batch是cfg中的batch参数。。h,w,c表示当前创建的卷积层的输入尺度，n表示卷积核数目，groups表示分组卷积组数（这个待定）
//size表示卷积核大小，stride表示卷积步长，padding表示是否填充，activation表示激活函数的类型，batch_normalize表示是否进行batch_normalize操作
//binary和xnor表示二值化网络的权值之类的内容 
//params.net->adam表示是否使用adam优化器。。默认是0
convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam)
{
    int i;
    //convolutional_layer的声明参考src/convolutional_layer.h
    //其实就是一个layer结构。。。
    convolutional_layer l = {0};
    //赋值当前层次的type
    l.type = CONVOLUTIONAL;
    //l.group表示分组卷积的组数。。。然后如果分组的话从channel维度拆分********后面再确认
    l.groups = groups;
    //给相关参数赋值，基本上都在上面解释过了
    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.binary = binary;
    l.xnor = xnor;
    l.batch = batch;
    l.stride = stride;
    l.size = size;
    l.pad = padding;
    l.batch_normalize = batch_normalize;
    //给权重分配内存
    l.weights = calloc(c/groups*n*size*size, sizeof(float));
    //分配权重更新的时候使用的内存
    l.weight_updates = calloc(c/groups*n*size*size, sizeof(float));
    //给偏置和偏置更新的时候分配内存
    l.biases = calloc(n, sizeof(float));
    l.bias_updates = calloc(n, sizeof(float));
    //l.nwights记录当前层卷积核的参数数量。。不包括bias
    l.nweights = c/groups*n*size*size;
    //l,=.nbiases记录当前层次的偏置数量
    l.nbiases = n;

    // float scale = 1./sqrt(size*size*c);
    //sqrt返回其中参数的平方根
    //初始化权重会用到。。。主要是为了将权重初始化到一个比较小的值
    float scale = sqrt(2./(size*size*c/l.groups));
    //printf("convscale %f\n", scale);
    //scale = .02;
    //for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_uniform(-1, 1);
    //rand_normal()用于从"服从指定正态分布的序列"中随机取出指定个数的值。
    //随机初始化权重rand_normal具体实现参考src/utils.c
    //rand_normal利用Box–Muller transform通过两个均匀分布的随机变量构造服从高斯分布的随机变量
    //这里利用scale将权重初始化到一个较小值。。如果初始化权重很大。。输入数据X也很大的时候。。WX +b就会比较大。。假如激活函数是sigmoid的化
    //有可能会输出1或者0...从而引发一系列问题。。比如计算损失的时候会产生log(0)....
    for(i = 0; i < l.nweights; ++i) l.weights[i] = scale*rand_normal();
    //convolutional_out_width计算该层的输出特征图的尺寸，具体实现参考src/convolutional_layer.c
    int out_w = convolutional_out_width(l);
    //跟上面的一样。。计算输出的height。。具体细节参考src/convolutional_layer.c
    int out_h = convolutional_out_height(l);
    //给输出特征图尺度的相关参数赋值
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    //统计输出特征图的数据量个数
    l.outputs = l.out_h * l.out_w * l.out_c;
    //统计当前层次输入特征图的数据量
    l.inputs = l.w * l.h * l.c;
    //给存储当前层输出特征图分配内存。。。注意l.batch*l.outputs。。。
    //这里的l.batch是cfg中的batch参数
    l.output = calloc(l.batch*l.outputs, sizeof(float));
    //同样给l.delta分配内存。。用来存储当前层次上l.output中每一个数值相对于激活函数的导数值，具体可以参考backward_convolutional_layer中的操作
    //会用到这个l.delta
    l.delta  = calloc(l.batch*l.outputs, sizeof(float));
    //这里的三个函数分别是进行某一个卷积层的前向传播，反向传播，参数更新的操作
    //具体实现都可以参考src/convolutional_laye.c
    l.forward = forward_convolutional_layer;
    //实现相应的反向传播的操作
    l.backward = backward_convolutional_layer;
    //实现卷积层的参数更新操作
    l.update = update_convolutional_layer;
    if(binary){
        l.binary_weights = calloc(l.nweights, sizeof(float));
        l.cweights = calloc(l.nweights, sizeof(char));
        l.scales = calloc(n, sizeof(float));
    }
    if(xnor){
        l.binary_weights = calloc(l.nweights, sizeof(float));
        l.binary_input = calloc(l.inputs*l.batch, sizeof(float));
    }

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
        l.m = calloc(l.nweights, sizeof(float));
        l.v = calloc(l.nweights, sizeof(float));
        l.bias_m = calloc(n, sizeof(float));
        l.scale_m = calloc(n, sizeof(float));
        l.bias_v = calloc(n, sizeof(float));
        l.scale_v = calloc(n, sizeof(float));
    }
//gpu中的卷积操作 
#ifdef GPU
    //卷积的前向传播的gpu版本,具体实现参考src/convolutional_kernels.cu
    l.forward_gpu = forward_convolutional_layer_gpu;
    //卷积的反向传播的gpu版本，具体实现参考src/convolutional_layer.cu
    l.backward_gpu = backward_convolutional_layer_gpu;
    //卷积的参数更新的gpu版本，具体实现参考src/convolutional_layer.cu
    l.update_gpu = update_convolutional_layer_gpu;

    if(gpu_index >= 0){
        if (adam) {
            l.m_gpu = cuda_make_array(l.m, l.nweights);
            l.v_gpu = cuda_make_array(l.v, l.nweights);
            l.bias_m_gpu = cuda_make_array(l.bias_m, n);
            l.bias_v_gpu = cuda_make_array(l.bias_v, n);
            l.scale_m_gpu = cuda_make_array(l.scale_m, n);
            l.scale_v_gpu = cuda_make_array(l.scale_v, n);
        }
        
        l.weights_gpu = cuda_make_array(l.weights, l.nweights);
        l.weight_updates_gpu = cuda_make_array(l.weight_updates, l.nweights);

        l.biases_gpu = cuda_make_array(l.biases, n);
        l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);

        l.delta_gpu = cuda_make_array(l.delta, l.batch*out_h*out_w*n);
        l.output_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);

        if(binary){
            l.binary_weights_gpu = cuda_make_array(l.weights, l.nweights);
        }
        if(xnor){
            l.binary_weights_gpu = cuda_make_array(l.weights, l.nweights);
            l.binary_input_gpu = cuda_make_array(0, l.inputs*l.batch);
        }

        if(batch_normalize){
            l.mean_gpu = cuda_make_array(l.mean, n);
            l.variance_gpu = cuda_make_array(l.variance, n);

            l.rolling_mean_gpu = cuda_make_array(l.mean, n);
            l.rolling_variance_gpu = cuda_make_array(l.variance, n);

            l.mean_delta_gpu = cuda_make_array(l.mean, n);
            l.variance_delta_gpu = cuda_make_array(l.variance, n);

            l.scales_gpu = cuda_make_array(l.scales, n);
            l.scale_updates_gpu = cuda_make_array(l.scale_updates, n);

            l.x_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
            l.x_norm_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
        }
#ifdef CUDNN
        cudnnCreateTensorDescriptor(&l.normTensorDesc);
        cudnnCreateTensorDescriptor(&l.srcTensorDesc);
        cudnnCreateTensorDescriptor(&l.dstTensorDesc);
        cudnnCreateFilterDescriptor(&l.weightDesc);
        cudnnCreateTensorDescriptor(&l.dsrcTensorDesc);
        cudnnCreateTensorDescriptor(&l.ddstTensorDesc);
        cudnnCreateFilterDescriptor(&l.dweightDesc);
        cudnnCreateConvolutionDescriptor(&l.convDesc);
        cudnn_convolutional_setup(&l);
#endif
    }
#endif
    //get_workspace_size的具体实现参考src/convolutional_layer.c
    //返回当前层l需要的工作空间的尺度
    l.workspace_size = get_workspace_size(l);
    l.activation = activation;

    fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c, (2.0 * l.n * l.size*l.size*l.c/l.groups * l.out_h*l.out_w)/1000000000.);

    return l;
}

void denormalize_convolutional_layer(convolutional_layer l)
{
    int i, j;
    for(i = 0; i < l.n; ++i){
        float scale = l.scales[i]/sqrt(l.rolling_variance[i] + .00001);
        for(j = 0; j < l.c/l.groups*l.size*l.size; ++j){
            l.weights[i*l.c/l.groups*l.size*l.size + j] *= scale;
        }
        l.biases[i] -= l.rolling_mean[i] * scale;
        l.scales[i] = 1;
        l.rolling_mean[i] = 0;
        l.rolling_variance[i] = 1;
    }
}

/*
void test_convolutional_layer()
{
    convolutional_layer l = make_convolutional_layer(1, 5, 5, 3, 2, 5, 2, 1, LEAKY, 1, 0, 0, 0);
    l.batch_normalize = 1;
    float data[] = {1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3};
    //net.input = data;
    //forward_convolutional_layer(l);
}
*/

void resize_convolutional_layer(convolutional_layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    int out_w = convolutional_out_width(*l);
    int out_h = convolutional_out_height(*l);

    l->out_w = out_w;
    l->out_h = out_h;

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
    cudnn_convolutional_setup(l);
#endif
#endif
    l->workspace_size = get_workspace_size(*l);
}
//add_bias(l.output, l.biases, l.batch, l.out_c, l.out_h*l.out_w);
void add_bias(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    //具体逻辑参考下面的scale_bias，两个是对应的
    //这里实现的是将l.output中的值跟biases中的值相加，更新到l.output中
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}
//scale_bias(l.output, l.scales, l.batch, l.out_c, l.out_h*l.out_w);
void scale_bias(float *output, float *scales, int batch, int n, int size)
{
    int i,j,b;
    //循环遍历batch中每一张图片的处理结果
    for(b = 0; b < batch; ++b){
        //遍历每一张图片输出的通道
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                //根据索引更新l.output中存储的值
                //将l.output中每一个值乘以scale中对应的数值
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}
//backward_bias(l.bias_updates, l.delta, l.batch, l.out_c, l.out_w*l.out_h);
//计算某一层偏置的更新值。。
void backward_bias(float *bias_updates, float *delta, int batch, int n, int size)
{
    int i,b;

    //循环遍历batch中每一张图片的相关数据
    for(b = 0; b < batch; ++b){
        //循环遍历每张图片输出数据的每一个通道
        for(i = 0; i < n; ++i){
            //根据相应的索引提取相应的delta的数值，也就是这里的delta+size*(i+b*n)操作
            //然后计算相应的bias_updates的值写入l.bias_updates中
            //sum_array的具体实现参考src/utils.c,实现的是将l.delta中size个数值累加的操作
            //也就是说l.bias_updates在两层循环结束以后存储的是一个batch中每一张图片输出的值相对于激活函数导数的值的累积的累加
            //就是l.delta中对应数值的累加，这里的n对应每一个卷积核，同时每一个卷积核对应一个偏置
            bias_updates[i] += sum_array(delta+size*(i+b*n), size);
        }
    }
}
//实现某一卷积层的前向传播
void forward_convolutional_layer(convolutional_layer l, network net)
{
    int i, j;
    //fill_cpu的具体实现参考src/blas.c
    //根据这里传入的参数。。这里实际上是将l.output里面存储的参数都初始化成0了
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);
    //如果是二值化网络。。。这部分暂时不写了。。和二值化网络有关
    if(l.xnor){
        binarize_weights(l.weights, l.n, l.c/l.groups*l.size*l.size, l.binary_weights);
        swap_binary(&l);
        binarize_cpu(net.input, l.c*l.h*l.w*l.batch, l.binary_input);
        net.input = l.binary_input;
    }
    //l.n表示当前层卷积核的个数。。如果是分组卷积。。m表示将卷积核对应分组之后的数量
    int m = l.n/l.groups;
    //l.size表示卷积核尺寸，l.c表示当前层输入的通道数
    //这里的k表示一次卷积的参数量
    int k = l.size*l.size*l.c/l.groups;
    //这里的n表示输出特征图w,h维度的参数量
    int n = l.out_w*l.out_h;
    //通过循环取每一张图和每一个groups,多数情况groups都是1。。。
    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            //a指向相应的权重位置根据groups定位
            float *a = l.weights + j*l.nweights/l.groups;
            //net.workspace是整个网络的工作空间，其元素个数为所有层中最大的l.workspace_size = l.out_h*l.out_w*l.size*l.size*l.c
            //充当一个临时工作空间的作用
            float *b = net.workspace;
            //c指向存储输出的l.output的相应位置，此轮计算结束，定位输出从哪里开始存储
            float *c = l.output + (i*l.groups + j)*n*m;
            //net.input存储了当前网络的输入数据，一般是一个batch的数量，这里是定位当前处理的图片数据的位置
            //如果是中间的卷积层，就是定位当前处理的特征图的位置，注意这里用的是l.c,l.w和l.h
            float *im =  net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w;
            //如果卷积核的尺寸是1
            if (l.size == 1) {
                //工作空间直接载入一张图片数据
                b = im;
                //如果卷积核尺寸不是1
            } else {
                //im2col_cpu的具体实现参考src/im2col.c
                //im2col的操作是将输入图像或者说输入特征图由多维将其转换成一个二维矩阵，使得卷积计算能够通过简单的两个矩阵相乘来实现
                //在这个过程中，会产生多余的内存，算是用空间换取计算效率的方式了
                //将一个卷积变成两个矩阵A和B的点积，然后假如A是卷积核的展开，B是图像的展开的话，那么A中的每一行就是一个多尺度卷积核行优先，然后按照channel展开的结果
                //而B中的每一列就是输入特征图上对应要尽行卷积运算的那一部分按照同样的规则对应展开的数据，因为stride的存在，所以B中存在大量重复的部分
                //会产生很多多余的内存
                //卷积核展开的矩阵行数是卷积核的个数，列数是一个卷积核的参数量，特征图的展开矩阵行数是一个卷积核的参数量，列数是输出特征图的W*h
                //这里就获得了图像展开后的矩阵，存储到了临时工作空间中
                im2col_cpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            }
            //gemm的具体实现参考src/gemm.c
            //实现的是两个矩阵点积的操作，大体过程就是取一个权重值，一次完成该权值在output相应位置上的全部更新，并不是按照理论上的一次计算完一个位置上的全部卷积计算
            gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }
    //是否进行batch_normalize计算
    if(l.batch_normalize){
        //进行batchnormalize。。
        //具体实现参考src/batchnorm_layer.c
        forward_batchnorm_layer(l, net);
    } else {
        //如果不进行batch_normalize操作
        //直接将l.output中的值跟l.biases中的值相加，并跟新到l.output中
        //这里就是把偏置b添加进去了
        //具体细节参考src/convolutional_layer.c
        add_bias(l.output, l.biases, l.batch, l.n, l.out_h*l.out_w);
    }
    //activate_array的具体实现参考src/activations.c
    //根据相应的激活函数实现激活操作
    activate_array(l.output, l.outputs*l.batch, l.activation);
    //二值化网络的相关内容。。后面在更新****************
    if(l.binary || l.xnor) swap_binary(&l);
}
//卷积的反向传播
void backward_convolutional_layer(convolutional_layer l, network net)
{
    int i, j;
    //这里的m表示一次计算的卷积核数量，通常l.groups是1
    int m = l.n/l.groups;
    //n表示一次计算所需要的参数量，这里的参数是输入特征图上需要与相应卷积核进行计算的数值
    int n = l.size*l.size*l.c/l.groups;
    //k表示输出的一个通道的参数量
    int k = l.out_w*l.out_h;
    //gradient_array的具体实现参考src/activations.c
    //这里的l.delta在make_convolutional_layer中分配内存并初始化为0，但是反向传播是从后往前传播的，所以l.delta在计算的时候的初始化值不是0
    //而是在网络的最后一层会再次赋初始值
    //计算l.output中每一个数值相对于激活函数的导数值并跟原来存储在l.delta中的值相乘，将结果存储到l.delta中
    //计算的是相对于激活的delta
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);
    //如果要进行batch_normalize
    if(l.batch_normalize){
        //batch_norm_layer的反向传播
        //具体实现参考src/batchnorm_layer.c
        backward_batchnorm_layer(l, net);
    } else {
        //如果不进行batchnorm
        //具体实现参考src/convolutional_layer.c
        //这里计算的是偏置的delta
        backward_bias(l.bias_updates, l.delta, l.batch, l.n, k);
    }
    //遍历batch中的每一张图片的数据
    for(i = 0; i < l.batch; ++i){
        //遍历groups。。通常是1
        for(j = 0; j < l.groups; ++j){
            //a定位到l.delta的相应的位置，这里为什么是m*k,因为l.delta可以看成是损失函数对于当前层l的加权输入的导数，
            //从而个数始于l.outputs对应的，从而是m*k
            float *a = l.delta + (i*l.groups + j)*m*k;
            //b定位当前的工作空间
            float *b = net.workspace;
            //c定位权重更新的值的位置
            float *c = l.weight_updates + j*l.nweights/l.groups;
            //在backward_network中已经将net.input赋值为当前层的前一层的输出，比如说当前是l层
            //那么这里的net.input就是l-1层的l.output
            //因为是一个batch一个batch的处理，所以这里也是定位相应的位置
            float *im  = net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w;
            //同样的net.delta就是上一层的l.delta，逻辑和上面的net.input一致
            float *imd = net.delta + (i*l.groups + j)*l.c/l.groups*l.h*l.w;
            //如果是1x1的卷积。。。
            if(l.size == 1){
                b = im;
            } else {
                //如果不是1x1的卷积
                //im2col_cpu的具体实现参考src/im2col.c
                //实现的是将输入的多尺度特征图转换成矩阵，便于卷积计算,转换的结果放到了工作空间b中
                im2col_cpu(im, l.c/l.groups, l.h, l.w, 
                        l.size, l.stride, l.pad, b);
            }
            //gemm的具体实现参考src/gemm.c，实际上这里调用的是gemm_nt
            //这里实现的具体的操作是l.delta与(l-1).output展开矩阵的转置的乘积
            //认真推理其中的矩阵维度也能得到这个结果，最好是和卷积的反向传播对应上
            //结果存储在了c指向的weight_updates里面
            //具体参考卷积反向传播的推导过程，关于l.delta，比如当前层的输入x,权重w,偏置b，激活函数f，令wx+b=z的话，那么当前层的输出output就是f（z）
            //而这里的l.delta就是损失函数相对于z的偏导数，其实是一个嵌套的循环关系，l.delta看成是损失函数相对于f(z)的导数也一样。。。但是从backward_bias
            //来看，偏置的更新值是那里的l.delta的累加。。所以那个时候的l.delta表示的是损失相对于当前层l的加权输入的导数，结合activation_layer里面传递的l.delta
            //l.delta里面的值看成是损失函数相对于当前层加权输入的导数更合适
            //具体的细节看一下region_layer.c中给l.delta赋值的操作。。后面确认一下**************************
            //就像是前向传播中传递的是l.output，反向传播中传递的是l.delta
            //同时这里的gemm最终是在c上累加的，并不是重写，完成多个batch的weight_updates的累积
            gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);
            //如果net.delta存在..
            //通常在network.c中的backward_network中会指定net.delta的值为上一层的delta也就是（l-1）.delta
            if (net.delta) {
                //定位相应权重的位置
                a = l.weights + j*l.nweights/l.groups;
                //同时定位相应的l.delta的位置
                b = l.delta + (i*l.groups + j)*m*k;
                //再次指向工作空间
                c = net.workspace;
                //如果是1x1的卷积
                if (l.size == 1) {
                    c = imd;
                }
                //调用gemm函数，具体实现参考src/gemm.c实际上这里调用的是gemm_tn
                //具体实现的操作是将a中的l.weigths展开矩阵的转置和b中l.delta的展开矩阵的乘积写入c中
                //这里计算的是损失函数相对于当前层的input的导数，也是上一层即l-1层的输出output的导数 
                //这一步结束之后得到的还不是最终的损失函数相对于上一层l-1层的outputd的导数，因为这里是特征图展开后的结果，还需要恢复到特征图展开之前
                gemm(1,0,n,k,m,1,a,n,b,k,0,c,k);

                if (l.size != 1) {
                    //这里将上面的c里面的内容恢复成特征图的形态，得到了最终的损失函数相对于前一层的输出output的导数，注意这时候还不是最终的经常用到的l.delta的值
                    //然后这里将结果写入了net.delta中，完成了对前一层即l-1层的delta的赋值，这个时候里面的值还是损失相对于前一层output的导数值
                    //下一次调用这个函数经过gradient_array之后就变成了损失函数相对于前一层的加权输入的导数值了
                    //col2im_cpu的具体实现参考src/col2im.c，注意这里的col2im_cpu里面是对imd进行的累加，不是重写，这一步累加完成了一个batch_size中的delta的累加
                    col2im_cpu(net.workspace, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, imd);
                }
            }
        }
    }
}
//根据相应优化算法更新卷积层的参数，这里采用的是动量梯度下降
void update_convolutional_layer(convolutional_layer l, update_args a)
{
    //计算最新的learning_rate
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    //获取相应的momentum参数
    float momentum = a.momentum;
    //获取衰减参数
    float decay = a.decay;
    //batch_size的值
    int batch = a.batch;
    //具体实现参考src/blas.c
    //这里的操作是完成偏置的更新，将l.bias_updates*(learning_rate/batch)加到l.biases上
    //这里要除以batch也就是batch_size是因为l.bias_update是整个batch_size上的累加，这里更新要取均值
    axpy_cpu(l.n, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    //具体实现参考src/blas.c
    //实现的操作是将l.bias_updates上的每一个值乘上momentum
    scal_cpu(l.n, momentum, l.bias_updates, 1);
    //batchnormalize中的scales
    if(l.scales){
        //利用l.scale_updates去更新scale，这两行和上面的逻辑一样
        axpy_cpu(l.n, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.n, momentum, l.scale_updates, 1);
    }
    //接下来就是权重的更新了
    //这里实现的操作是l.weight_updates += (-decay*batch)*l.weights
    //这里除以batch的操作原理同上，此处计算的是权重衰减值，每次将权重衰减一定的比例，防止过拟合
    //跟直接减少l.weights其实是一样的，两步的公式写到一起就能看出来了
    axpy_cpu(l.nweights, -decay*batch, l.weights, 1, l.weight_updates, 1);
    //接着上面的操作这里就是lweights += l.weight_updates*(learning_rate/batch)
    axpy_cpu(l.nweights, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    //l.weight_updates *= momentum
    scal_cpu(l.nweights, momentum, l.weight_updates, 1);
}


image get_convolutional_weight(convolutional_layer l, int i)
{
    int h = l.size;
    int w = l.size;
    int c = l.c/l.groups;
    return float_to_image(w,h,c,l.weights+i*h*w*c);
}

void rgbgr_weights(convolutional_layer l)
{
    int i;
    for(i = 0; i < l.n; ++i){
        image im = get_convolutional_weight(l, i);
        if (im.c == 3) {
            rgbgr_image(im);
        }
    }
}

void rescale_weights(convolutional_layer l, float scale, float trans)
{
    int i;
    for(i = 0; i < l.n; ++i){
        image im = get_convolutional_weight(l, i);
        if (im.c == 3) {
            scale_image(im, scale);
            float sum = sum_array(im.data, im.w*im.h*im.c);
            l.biases[i] += sum*trans;
        }
    }
}

image *get_weights(convolutional_layer l)
{
    image *weights = calloc(l.n, sizeof(image));
    int i;
    for(i = 0; i < l.n; ++i){
        weights[i] = copy_image(get_convolutional_weight(l, i));
        normalize_image(weights[i]);
        /*
           char buff[256];
           sprintf(buff, "filter%d", i);
           save_image(weights[i], buff);
         */
    }
    //error("hey");
    return weights;
}

image *visualize_convolutional_layer(convolutional_layer l, char *window, image *prev_weights)
{
    image *single_weights = get_weights(l);
    show_images(single_weights, l.n, window);

    image delta = get_convolutional_image(l);
    image dc = collapse_image_layers(delta, 1);
    char buff[256];
    sprintf(buff, "%s: Output", window);
    //show_image(dc, buff);
    //save_image(dc, buff);
    free_image(dc);
    return single_weights;
}

