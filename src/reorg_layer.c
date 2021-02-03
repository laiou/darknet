#include "reorg_layer.h"
#include "cuda.h"
#include "blas.h"

#include <stdio.h>

//创建一个reorg层，用在yolov2中
layer make_reorg_layer(int batch, int w, int h, int c, int stride, int reverse, int flatten, int extra)
{
    
    layer l = {0};
    l.type = REORG;
    l.batch = batch;
    //这里的stride的默认值是1
    l.stride = stride;
    //extra是cfg中读取的参数。。通常是默认值0
    l.extra = extra;
    l.h = h;
    l.w = w;
    l.c = c;
    //在yolov2的cfg中flatten用的也是默认值0
    l.flatten = flatten;
    if(reverse){
        l.out_w = w*stride;
        l.out_h = h*stride;
        l.out_c = c/(stride*stride);
    }else{
        l.out_w = w/stride;
        l.out_h = h/stride;
        l.out_c = c*(stride*stride);
    }
    //yolov2的cfg中reverse默认是0
    l.reverse = reverse;

    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    if(l.extra){
        l.out_w = l.out_h = l.out_c = 0;
        l.outputs = l.inputs + l.extra;
    }

    if(extra){
        fprintf(stderr, "reorg              %4d   ->  %4d\n",  l.inputs, l.outputs);
    } else {
        fprintf(stderr, "reorg              /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n",  stride, w, h, c, l.out_w, l.out_h, l.out_c);
    }
    int output_size = l.outputs * batch;
    l.output =  calloc(output_size, sizeof(float));
    l.delta =   calloc(output_size, sizeof(float));
    //reorg层的前向传播，具体实现参考src/reorg_layer.c
    l.forward = forward_reorg_layer;
    //reorg层的反向传播，具体实现参考src/reorg_layer.c
    l.backward = backward_reorg_layer;
#ifdef GPU
    //gpu版本的reorg层的前向传播，具体实现参考src/reorg_layer.c
    l.forward_gpu = forward_reorg_layer_gpu;
    //gpu版本的reorg层的反向传播，具体实现参考src/reorg_layer.c
    l.backward_gpu = backward_reorg_layer_gpu;

    l.output_gpu  = cuda_make_array(l.output, output_size);
    l.delta_gpu   = cuda_make_array(l.delta, output_size);
#endif
    return l;
}

void resize_reorg_layer(layer *l, int w, int h)
{
    int stride = l->stride;
    int c = l->c;

    l->h = h;
    l->w = w;

    if(l->reverse){
        l->out_w = w*stride;
        l->out_h = h*stride;
        l->out_c = c/(stride*stride);
    }else{
        l->out_w = w/stride;
        l->out_h = h/stride;
        l->out_c = c*(stride*stride);
    }

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->outputs;
    int output_size = l->outputs * l->batch;

    l->output = realloc(l->output, output_size * sizeof(float));
    l->delta = realloc(l->delta, output_size * sizeof(float));

#ifdef GPU
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->output_gpu  = cuda_make_array(l->output, output_size);
    l->delta_gpu   = cuda_make_array(l->delta,  output_size);
#endif
}
//reorg层的前向传播
void forward_reorg_layer(const layer l, network net)
{
    int i;
    //如果需要flatten
    if(l.flatten){
        //将net.input中的输入复制一份到l.output中
        memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));
        //通常reverse是0，表示flatten的反过程，具体参考flatten的实现
        if(l.reverse){
            flatten(l.output, l.w*l.h, l.c, l.batch, 0);
        }else{
            //flatten的实现参考src/blas.c
            flatten(l.output, l.w*l.h, l.c, l.batch, 1);
        }
    } else if (l.extra) {
        //如果需要extra
        //还是先遍历batch中每一个数据的处理结果
        for(i = 0; i < l.batch; ++i){
            //这里将net.input中的值复制到l.output中
            copy_cpu(l.inputs, net.input + i*l.inputs, 1, l.output + i*l.outputs, 1);
        }
        //如果需要reverse
    } else if (l.reverse){
        //rerog_cpu的实现参考src/blas.c
        //这里实现的是将相应的数据重新进行整合，具体参考reorg_cpu的具体实现
        reorg_cpu(net.input, l.w, l.h, l.c, l.batch, l.stride, 1, l.output);
    } else {
        //和上面一样。。参考rerog_cpu的实现
        reorg_cpu(net.input, l.w, l.h, l.c, l.batch, l.stride, 0, l.output);
    }
}
//rerog的反向传播
void backward_reorg_layer(const layer l, network net)
{
    int i;
    //和前向传播对应吧。。根据参数情况不同进入不通模块
    if(l.flatten){
        //将l.delta中的值复制到net.delta中
        memcpy(net.delta, l.delta, l.outputs*l.batch*sizeof(float));
        if(l.reverse){
            //然后根据前向传播的情况进行相应的数据还原，只不过这里还原的是l.delta
            //其他的都是一致的。。包括后面一些操作。。
            flatten(net.delta, l.w*l.h, l.c, l.batch, 1);
        }else{
            flatten(net.delta, l.w*l.h, l.c, l.batch, 0);
        }
    } else if(l.reverse){
        reorg_cpu(l.delta, l.w, l.h, l.c, l.batch, l.stride, 0, net.delta);
    } else if (l.extra) {
        for(i = 0; i < l.batch; ++i){
            copy_cpu(l.inputs, l.delta + i*l.outputs, 1, net.delta + i*l.inputs, 1);
        }
    }else{
        reorg_cpu(l.delta, l.w, l.h, l.c, l.batch, l.stride, 1, net.delta);
    }
}

#ifdef GPU
//gpu版本的reorg层的前向传播
void forward_reorg_layer_gpu(layer l, network net)
{
    int i;
    if(l.flatten){
        if(l.reverse){
            //flatten_gpu的具体实现参考src/blas_kernels.cu
            //flatten的gpu版本
            flatten_gpu(net.input_gpu, l.w*l.h, l.c, l.batch, 0, l.output_gpu);
        }else{
            flatten_gpu(net.input_gpu, l.w*l.h, l.c, l.batch, 1, l.output_gpu);
        }
    } else if (l.extra) {
        for(i = 0; i < l.batch; ++i){
            copy_gpu(l.inputs, net.input_gpu + i*l.inputs, 1, l.output_gpu + i*l.outputs, 1);
        }
    } else if (l.reverse) {
        //reorg_gpu的具体实现参考src/blas_kernels.cu
        reorg_gpu(net.input_gpu, l.w, l.h, l.c, l.batch, l.stride, 1, l.output_gpu);
    }else {
        reorg_gpu(net.input_gpu, l.w, l.h, l.c, l.batch, l.stride, 0, l.output_gpu);
    }
}
//gpu版本的reorg层的反向传播
void backward_reorg_layer_gpu(layer l, network net)
{
    if(l.flatten){
        if(l.reverse){
            flatten_gpu(l.delta_gpu, l.w*l.h, l.c, l.batch, 1, net.delta_gpu);
        }else{
            flatten_gpu(l.delta_gpu, l.w*l.h, l.c, l.batch, 0, net.delta_gpu);
        }
    } else if (l.extra) {
        int i;
        for(i = 0; i < l.batch; ++i){
            copy_gpu(l.inputs, l.delta_gpu + i*l.outputs, 1, net.delta_gpu + i*l.inputs, 1);
        }
    } else if(l.reverse){
        reorg_gpu(l.delta_gpu, l.w, l.h, l.c, l.batch, l.stride, 0, net.delta_gpu);
    } else {
        reorg_gpu(l.delta_gpu, l.w, l.h, l.c, l.batch, l.stride, 1, net.delta_gpu);
    }
}
#endif
