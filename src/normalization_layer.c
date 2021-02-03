#include "normalization_layer.h"
#include "blas.h"

#include <stdio.h>
//创建一个layer normalizaation层
//一般NLP中用layer_normalization，cv中一般用batch normalization
//简单来说就是BN是取不同样本的同一通道上的特征做归一化
//LN是取同一个样本的不同通道上的数据做归一化
//但是这里创建的是一个局部相应归一化层。。这里不是NLP中的LN。。
//关于LRN的具体介绍参考https://blog.csdn.net/u014296502/article/details/78839881
layer make_normalization_layer(int batch, int w, int h, int c, int size, float alpha, float beta, float kappa)
{
    fprintf(stderr, "Local Response Normalization Layer: %d x %d x %d image, %d size\n", w,h,c,size);
    layer layer = {0};
    layer.type = NORMALIZATION;
    layer.batch = batch;
    //输入和输出的尺度是一样的
    layer.h = layer.out_h = h;
    layer.w = layer.out_w = w;
    layer.c = layer.out_c = c;
    //kappa是公式推到里面的k
    layer.kappa = kappa;
    //size表示累加的通道范围
    layer.size = size;
    layer.alpha = alpha;
    layer.beta = beta;
    layer.output = calloc(h * w * c * batch, sizeof(float));
    layer.delta = calloc(h * w * c * batch, sizeof(float));
    layer.squared = calloc(h * w * c * batch, sizeof(float));
    layer.norms = calloc(h * w * c * batch, sizeof(float));
    layer.inputs = w*h*c;
    layer.outputs = layer.inputs;
    //正则化层的前向传播，具体实现参考src/normalization_layer.c
    layer.forward = forward_normalization_layer;
    //正则化层的反向传播，具体实现参考src/normalization_layer.c
    layer.backward = backward_normalization_layer;
    #ifdef GPU
    //gpu版本的normalization前向传播，具体实现参考src/normalizetion_layer.c
    layer.forward_gpu = forward_normalization_layer_gpu;
    //gpu版本的normalization的反向传播，具体实现参考src/normalization_layer.c
    layer.backward_gpu = backward_normalization_layer_gpu;

    layer.output_gpu =  cuda_make_array(layer.output, h * w * c * batch);
    layer.delta_gpu =   cuda_make_array(layer.delta, h * w * c * batch);
    layer.squared_gpu = cuda_make_array(layer.squared, h * w * c * batch);
    layer.norms_gpu =   cuda_make_array(layer.norms, h * w * c * batch);
    #endif
    return layer;
}

void resize_normalization_layer(layer *layer, int w, int h)
{
    int c = layer->c;
    int batch = layer->batch;
    layer->h = h;
    layer->w = w;
    layer->out_h = h;
    layer->out_w = w;
    layer->inputs = w*h*c;
    layer->outputs = layer->inputs;
    layer->output = realloc(layer->output, h * w * c * batch * sizeof(float));
    layer->delta = realloc(layer->delta, h * w * c * batch * sizeof(float));
    layer->squared = realloc(layer->squared, h * w * c * batch * sizeof(float));
    layer->norms = realloc(layer->norms, h * w * c * batch * sizeof(float));
#ifdef GPU
    cuda_free(layer->output_gpu);
    cuda_free(layer->delta_gpu); 
    cuda_free(layer->squared_gpu); 
    cuda_free(layer->norms_gpu);   
    layer->output_gpu =  cuda_make_array(layer->output, h * w * c * batch);
    layer->delta_gpu =   cuda_make_array(layer->delta, h * w * c * batch);
    layer->squared_gpu = cuda_make_array(layer->squared, h * w * c * batch);
    layer->norms_gpu =   cuda_make_array(layer->norms, h * w * c * batch);
#endif
}
//LRN层的前向传播
//主要关注的是norms中数值的计算。。。
//根据LRN的公式很好理解这个实现过程。。
void forward_normalization_layer(const layer layer, network net)
{
    int k,b;
    int w = layer.w;
    int h = layer.h;
    int c = layer.c;
    //这里实现的是将layer.squared里面的值置0
    scal_cpu(w*h*c*layer.batch, 0, layer.squared, 1);
    //遍历batch中每一张图片的处理数据
    for(b = 0; b < layer.batch; ++b){
        //定位相应的squared和norms以及input的位置
        float *squared = layer.squared + w*h*c*b;
        float *norms   = layer.norms + w*h*c*b;
        float *input   = net.input + w*h*c*b;
        //pow_cpu的具体实现参考src/blas.c
        //具体实现的是将input中相应值的平方赋值到squared中
        //取得是input中一个样本的数据量
        pow_cpu(w*h*c, 2, input, 1, squared, 1);
        //const_cpu的具体实现参考src/blas.c
        //这里实现的是将norm中的值初始化成layer.kappa
        //初始化了一个通道数据量的值
        const_cpu(w*h, layer.kappa, norms, 1);
        //也就是遍历前layer.size/2个通道的数据
        for(k = 0; k < layer.size/2; ++k){
            //将squared中相应的值乘上layer.alpha加到原来的norms中存储的值上去，结果写入norms
            //循环结束norms中存储了layer.size/2个通道上数据平方的和，也就是将一个样本中前layer.size/2个通道上的数据
            //每一个值先平方，在根据位置累加到一个通道大小的空间上了
            axpy_cpu(w*h, layer.alpha, squared + w*h*k, 1, norms, 1);
        }
        //从后面能看出虽然k=1开始，但是实际上还是遍历每一个通道
        for(k = 1; k < layer.c; ++k){
            //这里实现的是将norms中前一个通道数据赋值到后面存储一个通道数据的位置上
            copy_cpu(w*h, norms + w*h*(k-1), 1, norms + w*h*k, 1);
            //prev和next分别指向当前位置前一个步骤处理的通到的起始位置
            //和接下来后一个步骤要处理的通道的起始位置
            int prev = k - ((layer.size-1)/2) - 1;
            int next = k + (layer.size/2);
            //如果前一个步骤存在
            //将前一个步骤相应squared中的值乘上-layer.alpha,加到norms中对应的位置上
            //实际上就是通过prev和next保证对应位置上只有size+1个值的加权平方和
            //以及通过循环实现位置的移动
            if(prev >= 0)      axpy_cpu(w*h, -layer.alpha, squared + w*h*prev, 1, norms + w*h*k, 1);
            //如果下一个步骤存在的话
            //将squared中的值乘上layer.alpha，加到norms中对应的位置上
            if(next < layer.c) axpy_cpu(w*h,  layer.alpha, squared + w*h*next, 1, norms + w*h*k, 1);
        }
    }
    //这里实现的是将norms中对应值的-beta次方写入output
    //到了这里，执行这一步之前，norms中每个位置上的值代表的是原图片相应位置上的值跟沿着通道方向上前后各size/2个值的平方和，一共是size+1个值的平方和
    //并且加上了偏置kappa，对应的也乘上了系数alpha。。。
    pow_cpu(w*h*c*layer.batch, -layer.beta, layer.norms, 1, layer.output, 1);
    //接着将input中的值跟当前output中的对应值相乘，结果写入output
    mul_cpu(w*h*c*layer.batch, net.input, 1, layer.output, 1);
}
//局部正则化的反向传播
//其实就是和前向传播对应的
//从前向传播来看输入input乘上norms的-beta次方就成了输出的output
void backward_normalization_layer(const layer layer, network net)
{
    // TODO This is approximate ;-)
    // Also this should add in to delta instead of overwritting.

    int w = layer.w;
    int h = layer.h;
    int c = layer.c;
    //将norms中的值的-beta次方写入net.delta
    pow_cpu(w*h*c*layer.batch, -layer.beta, layer.norms, 1, net.delta, 1);
    //将layer.delta中的值乘到net.delta中去，结果写入net.delta
    mul_cpu(w*h*c*layer.batch, layer.delta, 1, net.delta, 1);
}

#ifdef GPU
//normalization层前向传播的gpu实现。。对照cpu版本。。基本过程一致
void forward_normalization_layer_gpu(const layer layer, network net)
{
    int k,b;
    int w = layer.w;
    int h = layer.h;
    int c = layer.c;
    scal_gpu(w*h*c*layer.batch, 0, layer.squared_gpu, 1);

    for(b = 0; b < layer.batch; ++b){
        float *squared = layer.squared_gpu + w*h*c*b;
        float *norms   = layer.norms_gpu + w*h*c*b;
        float *input   = net.input_gpu + w*h*c*b;
        pow_gpu(w*h*c, 2, input, 1, squared, 1);

        const_gpu(w*h, layer.kappa, norms, 1);
        for(k = 0; k < layer.size/2; ++k){
            axpy_gpu(w*h, layer.alpha, squared + w*h*k, 1, norms, 1);
        }

        for(k = 1; k < layer.c; ++k){
            copy_gpu(w*h, norms + w*h*(k-1), 1, norms + w*h*k, 1);
            int prev = k - ((layer.size-1)/2) - 1;
            int next = k + (layer.size/2);
            if(prev >= 0)      axpy_gpu(w*h, -layer.alpha, squared + w*h*prev, 1, norms + w*h*k, 1);
            if(next < layer.c) axpy_gpu(w*h,  layer.alpha, squared + w*h*next, 1, norms + w*h*k, 1);
        }
    }
    pow_gpu(w*h*c*layer.batch, -layer.beta, layer.norms_gpu, 1, layer.output_gpu, 1);
    mul_gpu(w*h*c*layer.batch, net.input_gpu, 1, layer.output_gpu, 1);
}

//normalization层的反向传播的gpu实现。。对照cpu版本..基本过程一致
void backward_normalization_layer_gpu(const layer layer, network net)
{
    // TODO This is approximate ;-)

    int w = layer.w;
    int h = layer.h;
    int c = layer.c;
    pow_gpu(w*h*c*layer.batch, -layer.beta, layer.norms_gpu, 1, net.delta_gpu, 1);
    mul_gpu(w*h*c*layer.batch, layer.delta_gpu, 1, net.delta_gpu, 1);
}
#endif
