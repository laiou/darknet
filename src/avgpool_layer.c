#include "avgpool_layer.h"
#include "cuda.h"
#include <stdio.h>
//给一个平均池化层分配内存，赋值相关的参数
//这里的输入时w,h,c输出是1，1，c
avgpool_layer make_avgpool_layer(int batch, int w, int h, int c)
{
    fprintf(stderr, "avg                     %4d x%4d x%4d   ->  %4d\n",  w, h, c, c);
    //给相关参数赋值
    avgpool_layer l = {0};
    l.type = AVGPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.out_w = 1;
    l.out_h = 1;
    l.out_c = c;
    l.outputs = l.out_c;
    l.inputs = h*w*c;
    int output_size = l.outputs * batch;
    l.output =  calloc(output_size, sizeof(float));
    l.delta =   calloc(output_size, sizeof(float));
    //平均池化层的前向传播
    //具体实现参考src/avgpool_layer.c
    l.forward = forward_avgpool_layer;
    //平均池化层的反向传播
    l.backward = backward_avgpool_layer;
    //gpu版本的平均池化层
    #ifdef GPU
    //gpu版本avg_pool的前向传播,具体实现参考src/avgpool_layer_kernels.cu
    l.forward_gpu = forward_avgpool_layer_gpu;
    //gpu版本的avg_pool的反向传播,具体实现参考src/avgpool_layer_kernels.cu
    l.backward_gpu = backward_avgpool_layer_gpu;
    //给相应的output_gpu核delta_gpu分配内存
    l.output_gpu  = cuda_make_array(l.output, output_size);
    l.delta_gpu   = cuda_make_array(l.delta, output_size);
    #endif
    return l;
}

void resize_avgpool_layer(avgpool_layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    l->inputs = h*w*l->c;
}
//平均池化层的前向传播
void forward_avgpool_layer(const avgpool_layer l, network net)
{
    int b,i,k;
    //循环遍历batch中每张图的结果
    for(b = 0; b < l.batch; ++b){
        //遍历每张图片输出特征图下的每一个通道
        for(k = 0; k < l.c; ++k){
            //提取相应的l.output中的索引
            int out_index = k + b*l.c;
            l.output[out_index] = 0;
            //遍历每个通道上w*h个数据，计算均值
            for(i = 0; i < l.h*l.w; ++i){
                //计算索引，提取数值，累加
                int in_index = i + l.h*l.w*(k + b*l.c);
                l.output[out_index] += net.input[in_index];
            }
            //求出最终的均值
            l.output[out_index] /= l.h*l.w;
        }
    }
}
//平均池化层的反向传播
void backward_avgpool_layer(const avgpool_layer l, network net)
{
    int b,i,k;
    //循环遍历batch中每张图的结果
    for(b = 0; b < l.batch; ++b){
        //遍历每张图结果下的每一个channel
        for(k = 0; k < l.c; ++k){
            //计算索引
            int out_index = k + b*l.c;
            //根据平均池化的计算公式的导数，向上一层传递l.delta
            for(i = 0; i < l.h*l.w; ++i){
                int in_index = i + l.h*l.w*(k + b*l.c);
                net.delta[in_index] += l.delta[out_index] / (l.h*l.w);
            }
        }
    }
}

