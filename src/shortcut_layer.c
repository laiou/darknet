#include "shortcut_layer.h"
#include "cuda.h"
#include "blas.h"
#include "activations.h"

#include <stdio.h>
#include <assert.h>
//创建一个shortcut层，也就是resnet中的跳连接
layer make_shortcut_layer(int batch, int index, int w, int h, int c, int w2, int h2, int c2)
{
    fprintf(stderr, "res  %3d                %4d x%4d x%4d   ->  %4d x%4d x%4d\n",index, w2,h2,c2, w,h,c);
    layer l = {0};
    l.type = SHORTCUT;
    l.batch = batch;
    //l.w,l.h.l.c代表跳连接传入的数据尺度
    l.w = w2;
    l.h = h2;
    l.c = c2;
    //out_w这些指的是当前层的输出尺度，也是如果不存在跳连接的话这一层的原本的输入尺度
    l.out_w = w;
    l.out_h = h;
    l.out_c = c;
    l.outputs = w*h*c;
    l.inputs = l.outputs;

    l.index = index;

    l.delta =  calloc(l.outputs*batch, sizeof(float));
    l.output = calloc(l.outputs*batch, sizeof(float));;
    //shortcut的前向传播，具体细节参考src/shortcut_layer.c
    l.forward = forward_shortcut_layer;
    //shortcut的反向传播，具体细节参考src/shortcut_layer.c
    l.backward = backward_shortcut_layer;
    #ifdef GPU
    //shortcut层前向传播的gpu实现，具体实现参考src/shortcut_layer.c
    l.forward_gpu = forward_shortcut_layer_gpu;
    //shortcut层反向传播的gpu实现，具体实现参考src/shortcut_layer.c
    l.backward_gpu = backward_shortcut_layer_gpu;

    l.delta_gpu =  cuda_make_array(l.delta, l.outputs*batch);
    l.output_gpu = cuda_make_array(l.output, l.outputs*batch);
    #endif
    return l;
}

void resize_shortcut_layer(layer *l, int w, int h)
{
    assert(l->w == l->out_w);
    assert(l->h == l->out_h);
    l->w = l->out_w = w;
    l->h = l->out_h = h;
    l->outputs = w*h*l->out_c;
    l->inputs = l->outputs;
    l->delta =  realloc(l->delta, l->outputs*l->batch*sizeof(float));
    l->output = realloc(l->output, l->outputs*l->batch*sizeof(float));

#ifdef GPU
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->output_gpu  = cuda_make_array(l->output, l->outputs*l->batch);
    l->delta_gpu   = cuda_make_array(l->delta,  l->outputs*l->batch);
#endif
    
}

//shortcut层的前向传播
void forward_shortcut_layer(const layer l, network net)
{   
    //将net.input中的值赋值到l.output中去
    copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
    //shortcut_cpu的具体实现参考src/blas.c
    shortcut_cpu(l.batch, l.w, l.h, l.c, net.layers[l.index].output, l.out_w, l.out_h, l.out_c, l.alpha, l.beta, l.output);
    //计算l.output的激活值，结果写入l.output
    activate_array(l.output, l.outputs*l.batch, l.activation);
}
//shortcut的反向传播
void backward_shortcut_layer(const layer l, network net)
{
    //跟前向传播对应的。。先计算激活函数相对于l.output的的梯度，然后将这个结果乘到l.delta中对应的位置上，结果写入l.delta
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);
    //这里实现的是将l.delta中的值和l.alpha的值相乘然后加到net.delta中，计算的实际上也是损失函数相对于跳连接层的输出output的梯度
    //结合公式，更容易看出来，或者说计算的是l.delta里面应该传递到上一层去的梯度
    axpy_cpu(l.outputs*l.batch, l.alpha, l.delta, 1, net.delta, 1);
    //然后这里就是计算跳连接传入的那一层的梯度，或者说应该传入到跳连接连接的那一层的梯度
    //具体实现参考src/blas.c，这些内容结合相应的公式就比较容易看出来了，实际上都是对应的
    shortcut_cpu(l.batch, l.out_w, l.out_h, l.out_c, l.delta, l.w, l.h, l.c, 1, l.beta, net.layers[l.index].delta);
}

#ifdef GPU
//shortcut前向传播的gpu实现
void forward_shortcut_layer_gpu(const layer l, network net)
{
    copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
    //shortcut_gpu的具体实现参考src/blas_kernels.cu
    shortcut_gpu(l.batch, l.w, l.h, l.c, net.layers[l.index].output_gpu, l.out_w, l.out_h, l.out_c, l.alpha, l.beta, l.output_gpu);
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
}

//shortcut层反向传播的gpu实现
void backward_shortcut_layer_gpu(const layer l, network net)
{
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    axpy_gpu(l.outputs*l.batch, l.alpha, l.delta_gpu, 1, net.delta_gpu, 1);
    shortcut_gpu(l.batch, l.out_w, l.out_h, l.out_c, l.delta_gpu, l.w, l.h, l.c, 1, l.beta, net.layers[l.index].delta_gpu);
}
#endif
