#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "cuda.h"
}

__global__ void binarize_kernel(float *x, int n, float *binary)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    binary[i] = (x[i] >= 0) ? 1 : -1;
}

void binarize_gpu(float *x, int n, float *binary)
{
    binarize_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, binary);
    check_error(cudaPeekAtLastError());
}

__global__ void binarize_input_kernel(float *input, int n, int size, float *binary)
{
    int s = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (s >= size) return;
    int i = 0;
    float mean = 0;
    for(i = 0; i < n; ++i){
        mean += fabsf(input[i*size + s]);
    }
    mean = mean / n;
    for(i = 0; i < n; ++i){
        binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
    }
}

void binarize_input_gpu(float *input, int n, int size, float *binary)
{
    binarize_input_kernel<<<cuda_gridsize(size), BLOCK>>>(input, n, size, binary);
    check_error(cudaPeekAtLastError());
}


__global__ void binarize_weights_kernel(float *weights, int n, int size, float *binary)
{
    int f = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (f >= n) return;
    int i = 0;
    float mean = 0;
    for(i = 0; i < size; ++i){
        mean += fabsf(weights[f*size + i]);
    }
    mean = mean / size;
    for(i = 0; i < size; ++i){
        binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        //binary[f*size + i] = weights[f*size + i];
    }
}

void binarize_weights_gpu(float *weights, int n, int size, float *binary)
{
    binarize_weights_kernel<<<cuda_gridsize(n), BLOCK>>>(weights, n, size, binary);
    check_error(cudaPeekAtLastError());
}

//卷积层前向传播的gpu版本
void forward_convolutional_layer_gpu(convolutional_layer l, network net)
{
    
    //将l.output_gpu中的值用0初始化
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    //如果需要二值化的话。。。
    if(l.binary){
        binarize_weights_gpu(l.weights_gpu, l.n, l.c/l.groups*l.size*l.size, l.binary_weights_gpu);
        swap_binary(&l);
    }

    if(l.xnor){
        binarize_weights_gpu(l.weights_gpu, l.n, l.c/l.groups*l.size*l.size, l.binary_weights_gpu);
        swap_binary(&l);
        binarize_gpu(net.input_gpu, l.c*l.h*l.w*l.batch, l.binary_input_gpu);
        net.input_gpu = l.binary_input_gpu;
    }

#ifdef CUDNN
    float one = 1;
    cudnnConvolutionForward(cudnn_handle(),
                &one,
                l.srcTensorDesc,
                net.input_gpu,
                l.weightDesc,
                l.weights_gpu,
                l.convDesc,
                l.fw_algo,
                net.workspace,
                l.workspace_size,
                &one,
                l.dstTensorDesc,
                l.output_gpu);

#else
    //通常都是进入了这里
    int i, j;
    //表示当前层卷积核的个数，如果采用分组卷积，m表示分组卷积分组后的卷积核个数
    int m = l.n/l.groups;
    //k表示参与一次卷积计算的参数量
    int k = l.size*l.size*l.c/l.groups;
    //n表示输出特征图上的参数量
    int n = l.out_w*l.out_h;
    //遍历batch中每张图片的处理数据
    for(i = 0; i < l.batch; ++i){
    //如果采取分组卷积，遍历每一个分组
        for(j = 0; j < l.groups; ++j){
            //计算相应的权重指针
            float *a = l.weights_gpu + j*l.nweights/l.groups;
            //定位工作空间的位置，工作空间在parser.c中进行内存分配...
            float *b = net.workspace;
            //定位相应的输出保存的位置
            float *c = l.output_gpu + (i*l.groups + j)*n*m;
            //定位相应输入特征图
            float *im = net.input_gpu + (i*l.groups + j)*l.c/l.groups*l.h*l.w;
            //如果是1x1的卷积。。。
            if (l.size == 1){
                b = im;
            } else {
            //如果不是1x1的卷积
            //im2col_gpu的具体实现参考src/im2col_kernels.cu
            //实现的是将相应的图片参与卷积的数据展开成一个大矩阵,这里通过多线程按照卷积次数进行展开
                im2col_gpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            }
            //调用gemm_gpu函数实现权重矩阵和展开特征图矩阵的乘积操作，结果写入l.output_gpu中
            //具体实现参考src/gemm.c
            gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }
#endif
    //如果有bn操作
    if (l.batch_normalize) {
    //进行bn的前行传播,具体实现参考src/batchnorm_layer.c
        forward_batchnorm_layer_gpu(l, net);
    } else {
        //没有bn的话。。将偏置加到l.output_gpu上去
        //add_bias_gpu的具体实现参考src/blas_kernels.cu
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
    }
    //在gpu上计算激活值，具体实现参考src/activation_kernels.cu
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
    //if(l.dot > 0) dot_error_gpu(l);
    if(l.binary || l.xnor) swap_binary(&l);
}

//smooth_kernel<<<cuda_gridsize(n), BLOCK>>>(l.output_gpu, n, l.w, l.h, l.c, size, rate, l.delta_gpu);
//对l.delta_gpu中的值做一个微调
__global__ void smooth_kernel(float *x, int n, int w, int h, int c, int size, float rate, float *delta)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int j = id % w;
    id /= w;
    int i = id % h;
    id /= h;
    int k = id % c;
    id /= c;
    int b = id;

    int w_offset = -(size/2.f);
    int h_offset = -(size/2.f);

    int out_index = j + w*(i + h*(k + c*b));
    int l, m;
    for(l = 0; l < size; ++l){
        for(m = 0; m < size; ++m){
            int cur_h = h_offset + i + l;
            int cur_w = w_offset + j + m;
            int index = cur_w + w*(cur_h + h*(k + b*c));
            int valid = (cur_h >= 0 && cur_h < h &&
                    cur_w >= 0 && cur_w < w);
            delta[out_index] += valid ? rate*(x[index] - x[out_index]) : 0;
        }
    }
}

//smooth_layer(l, 5, l.smooth);
extern "C" void smooth_layer(layer l, int size, float rate)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.out_c;

    size_t n = h*w*c*l.batch;
    //smooth_kernel的实现参考src/convolutional_kernels.cu
    smooth_kernel<<<cuda_gridsize(n), BLOCK>>>(l.output_gpu, n, l.w, l.h, l.c, size, rate, l.delta_gpu);
    check_error(cudaPeekAtLastError());
}

//卷积操作反向传播的gpu版本
void backward_convolutional_layer_gpu(convolutional_layer l, network net)
{
    //如果需要smooth
    if(l.smooth){
        //smooth_layer的实现参考src/convolutional_kernels.cu
        //实现的是对l.delta_gpu中的值进行一个细微的调整
        smooth_layer(l, 5, l.smooth);
    }
    //计算激活函数相对于output的导数，结果乘到l.delta_gpu里面
    //gradient_array_gpu的具体实现参考src/activation_kernels.cu
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);

    //如果需要bn操作
    if(l.batch_normalize){
    //进行bn层的反向传播
        backward_batchnorm_layer_gpu(l, net);
    } else {
    //不需要bn操作的话，进行偏置的更新值的计算
    //具体实现参考src/blas_kernels.cu
        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);
    }
    //定位net.input_gpu的位置
    float *original_input = net.input_gpu;

    if(l.xnor) net.input_gpu = l.binary_input_gpu;
#ifdef CUDNN
    float one = 1;
    cudnnConvolutionBackwardFilter(cudnn_handle(),
            &one,
            l.srcTensorDesc,
            net.input_gpu,
            l.ddstTensorDesc,
            l.delta_gpu,
            l.convDesc,
            l.bf_algo,
            net.workspace,
            l.workspace_size,
            &one,
            l.dweightDesc,
            l.weight_updates_gpu);

    if(net.delta_gpu){
        if(l.binary || l.xnor) swap_binary(&l);
        cudnnConvolutionBackwardData(cudnn_handle(),
                &one,
                l.weightDesc,
                l.weights_gpu,
                l.ddstTensorDesc,
                l.delta_gpu,
                l.convDesc,
                l.bd_algo,
                net.workspace,
                l.workspace_size,
                &one,
                l.dsrcTensorDesc,
                net.delta_gpu);
        if(l.binary || l.xnor) swap_binary(&l);
        if(l.xnor) gradient_array_gpu(original_input, l.batch*l.c*l.h*l.w, HARDTAN, net.delta_gpu);
    }

#else
    int m = l.n/l.groups;
    int n = l.size*l.size*l.c/l.groups;
    int k = l.out_w*l.out_h;

    //循环遍历batch中每张图的处理数据
    int i, j;
    for(i = 0; i < l.batch; ++i){
    //遍历每一个通道
        for(j = 0; j < l.groups; ++j){
        //计算权重的更新值，参考卷积的反向传播推导
            float *a = l.delta_gpu + (i*l.groups + j)*m*k;
            float *b = net.workspace;
            float *c = l.weight_updates_gpu + j*l.nweights/l.groups;

            float *im  = net.input_gpu+(i*l.groups + j)*l.c/l.groups*l.h*l.w;
            float *imd = net.delta_gpu+(i*l.groups + j)*l.c/l.groups*l.h*l.w;
            //和cpu版本主要功能一致
            //调用im2col和gemm完成权重更新值的计算
            im2col_gpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            gemm_gpu(0,1,m,n,k,1,a,k,b,k,1,c,n);
            //计算net.delta_gpu，完成delta的传递
            if (net.delta_gpu) {
                if (l.binary || l.xnor) swap_binary(&l);
                a = l.weights_gpu + j*l.nweights/l.groups;
                b = l.delta_gpu + (i*l.groups + j)*m*k;
                c = net.workspace;
                if (l.size == 1) {
                    c = imd;
                }
                //这里也可以参考cpu版本的注释。。一样的流程
                gemm_gpu(1,0,n,k,m,1,a,n,b,k,0,c,k);

                if (l.size != 1) {
                //col2im_gpu实现的是im2col_gpu相反的功能，将矩阵转换成特征图的形式
                //具体实现参考src/col2im_kernels.cu
                    col2im_gpu(net.workspace, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, imd);
                }
                if(l.binary || l.xnor) {
                    swap_binary(&l);
                }
            }
            if(l.xnor) gradient_array_gpu(original_input + i*l.c*l.h*l.w, l.c*l.h*l.w, HARDTAN, net.delta_gpu + i*l.c*l.h*l.w);
        }
    }
#endif
}

void pull_convolutional_layer(layer l)
{
    cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
    cuda_pull_array(l.biases_gpu, l.biases, l.n);
    cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
    cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
    if (l.batch_normalize){
        cuda_pull_array(l.scales_gpu, l.scales, l.n);
        cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}
//将卷积层的权重上传到gpu设备上
//关于这个。。实际上就是把主机上的值推送到设备的过程，从src/parser.c中能看出，实际上也是先将weights读取到
//相应层次的cpu版本中比如说先读取到相应层次的l.weights里面，再从l..weights中推送到设备上的l.weights_gpu
void push_convolutional_layer(layer l)
{
    //cuda_push_array的具体实现参考src/cuda.c
    //实现将当前卷积层的权重矩阵复制到设备上相应的内存里
    cuda_push_array(l.weights_gpu, l.weights, l.nweights);
    //同样的操作，这里是复制当前层的偏置
    cuda_push_array(l.biases_gpu, l.biases, l.n);
    //复制权重跟新值。。训练的时候前向传播里面会用到，，关于这个的作用。参考cpu版本的反向传播
    cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
    //跟上面一样。。复制偏置更新值到gpu
    cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
    //如果当前卷积层存在batch_normalize...
    if (l.batch_normalize){
    //将Batch_normalize的相关参数复制到gpu上
        cuda_push_array(l.scales_gpu, l.scales, l.n);
        cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}
//gpu版本的卷积参数更新
void update_convolutional_layer_gpu(layer l, update_args a)
{
    //获得当前的学习率和动量系数等参数
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;
    //adam优化器
    if(a.adam){
        adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.nweights, batch, a.t);
        adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
        if(l.scales_gpu){
            adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
        }
    }else{
    //这里是实现的是动量梯度下降
    //计算权重的衰减和更新权重
    //axpy_gpu的实现参考src/blas_kernels.cu
        axpy_gpu(l.nweights, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
        axpy_gpu(l.nweights, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
        //计算并积累相应的动量
        //l.weigths_updates_gpu *= momentum
        scal_gpu(l.nweights, momentum, l.weight_updates_gpu, 1);
        //和上面的一样，计算偏执更新值
        //axpy_gpu的具体实现参考src/blas_kernels.cu
        axpy_gpu(l.n, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
        scal_gpu(l.n, momentum, l.bias_updates_gpu, 1);

        if(l.scales_gpu){
            axpy_gpu(l.n, learning_rate/batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
            scal_gpu(l.n, momentum, l.scale_updates_gpu, 1);
        }
    }
    //如果需要裁剪
    if(l.clip){
    //constrain_gpu的具体实现参考src/blas_kernels.cu
    //将l.weights_gpu的值修正到-l.clip和l.clip之间
        constrain_gpu(l.nweights, l.clip, l.weights_gpu, 1);
    }
}


