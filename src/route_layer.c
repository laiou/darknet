#include "route_layer.h"
#include "cuda.h"
#include "blas.h"

#include <stdio.h>
//创建一个route层，实现的功能就是将几个输入层拼接到一起
route_layer make_route_layer(int batch, int n, int *input_layers, int *input_sizes)
{
    fprintf(stderr,"route ");
    route_layer l = {0};
    l.type = ROUTE;
    l.batch = batch;
    //这里的n表示输入层的个数，也就是要拼接几个层
    l.n = n;
    l.input_layers = input_layers;
    l.input_sizes = input_sizes;
    int i;
    int outputs = 0;
    for(i = 0; i < n; ++i){
        fprintf(stderr," %d", input_layers[i]);
        outputs += input_sizes[i];
    }
    fprintf(stderr, "\n");
    l.outputs = outputs;
    l.inputs = outputs;
    l.delta =  calloc(outputs*batch, sizeof(float));
    l.output = calloc(outputs*batch, sizeof(float));;
    //route层的前向传播，具体实现参考src/route_layer.c
    l.forward = forward_route_layer;
    //route层的反向传播，具体实现参考src/route_layer.c
    l.backward = backward_route_layer;
    #ifdef GPU
    //route层前向传播的gpu实现，具体实现参考src/route_layer.c
    l.forward_gpu = forward_route_layer_gpu;
    //route层反向传播的gpu实现，具体实现参考src/route_layer.c
    l.backward_gpu = backward_route_layer_gpu;

    l.delta_gpu =  cuda_make_array(l.delta, outputs*batch);
    l.output_gpu = cuda_make_array(l.output, outputs*batch);
    #endif
    return l;
}

void resize_route_layer(route_layer *l, network *net)
{
    int i;
    layer first = net->layers[l->input_layers[0]];
    l->out_w = first.out_w;
    l->out_h = first.out_h;
    l->out_c = first.out_c;
    l->outputs = first.outputs;
    l->input_sizes[0] = first.outputs;
    for(i = 1; i < l->n; ++i){
        int index = l->input_layers[i];
        layer next = net->layers[index];
        l->outputs += next.outputs;
        l->input_sizes[i] = next.outputs;
        if(next.out_w == first.out_w && next.out_h == first.out_h){
            l->out_c += next.out_c;
        }else{
            printf("%d %d, %d %d\n", next.out_w, next.out_h, first.out_w, first.out_h);
            l->out_h = l->out_w = l->out_c = 0;
        }
    }
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
//route层的前向传播
void forward_route_layer(const route_layer l, network net)
{
    int i, j;
    int offset = 0;
    //遍历要拼接的每一个输入层
    for(i = 0; i < l.n; ++i){
        //index表示输入的相应的拼接层的索引，这个是在整个网络层次中的层次索引
        int index = l.input_layers[i];
        //根据索引抽取相应层次的output
        float *input = net.layers[index].output;
        //提取当前route层的input_size
        int input_size = l.input_sizes[i];
        //遍历batch中每张图的处理数据
        for(j = 0; j < l.batch; ++j){
            //将上面Input中对应的数据赋值到l.output的对应位置上
            copy_cpu(input_size, input + j*input_size, 1, l.output + offset + j*l.outputs, 1);
        }
        //更新offset的定位，也就是记录output中存储下一个拼接层数据的起始位置
        offset += input_size;
    }
}
//route层的反向传播
void backward_route_layer(const route_layer l, network net)
{
    int i, j;
    int offset = 0;
    //遍历拼接的每一层
    for(i = 0; i < l.n; ++i){
        //还是提取原来的拼接层在整个网络层次中的索引
        int index = l.input_layers[i];
        //定位相应参与拼接的这些层次的deltad额存储位置
        float *delta = net.layers[index].delta;
        //提取相应的layers的size值，也就是参与拼接的这些层次的尺度
        int input_size = l.input_sizes[i];
        //遍历batch中每张图的处理结果
        for(j = 0; j < l.batch; ++j){
            //将l.delta中对应的值加到上面得到的delta中去，完成相应层次的delta的传递
            axpy_cpu(input_size, 1, l.delta + offset + j*l.outputs, 1, delta + j*input_size, 1);
        }
        //记录下一次提取delta的起始位置
        offset += input_size;
    }
}

#ifdef GPU
//route层前向传播的gpu实现
void forward_route_layer_gpu(const route_layer l, network net)
{
    int i, j;
    int offset = 0;
    for(i = 0; i < l.n; ++i){
        int index = l.input_layers[i];
        float *input = net.layers[index].output_gpu;
        int input_size = l.input_sizes[i];
        for(j = 0; j < l.batch; ++j){
            copy_gpu(input_size, input + j*input_size, 1, l.output_gpu + offset + j*l.outputs, 1);
        }
        offset += input_size;
    }
}
//route层反向传播的gpu实现
void backward_route_layer_gpu(const route_layer l, network net)
{
    int i, j;
    int offset = 0;
    for(i = 0; i < l.n; ++i){
        int index = l.input_layers[i];
        float *delta = net.layers[index].delta_gpu;
        int input_size = l.input_sizes[i];
        for(j = 0; j < l.batch; ++j){
            axpy_gpu(input_size, 1, l.delta_gpu + offset + j*l.outputs, 1, delta + j*input_size, 1);
        }
        offset += input_size;
    }
}
#endif
