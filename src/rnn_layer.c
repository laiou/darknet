#include "rnn_layer.h"
#include "connected_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void increment_layer(layer *l, int steps)
{
    int num = l->outputs*l->batch*steps;
    l->output += num;
    l->delta += num;
    l->x += num;
    l->x_norm += num;

#ifdef GPU
    l->output_gpu += num;
    l->delta_gpu += num;
    l->x_gpu += num;
    l->x_norm_gpu += num;
#endif
}
//创建一个RNN层，有些参数之类的参考其他的比如GRU，LSTM，CRNN等层次的解释。。
layer make_rnn_layer(int batch, int inputs, int outputs, int steps, ACTIVATION activation, int batch_normalize, int adam)
{
    fprintf(stderr, "RNN Layer: %d inputs, %d outputs\n", inputs, outputs);
    batch = batch / steps;
    layer l = {0};
    l.batch = batch;
    l.type = RNN;
    l.steps = steps;
    l.inputs = inputs;

    l.state = calloc(batch*outputs, sizeof(float));
    l.prev_state = calloc(batch*outputs, sizeof(float));

    l.input_layer = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.input_layer) = make_connected_layer(batch*steps, inputs, outputs, activation, batch_normalize, adam);
    l.input_layer->batch = batch;

    l.self_layer = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.self_layer) = make_connected_layer(batch*steps, outputs, outputs, activation, batch_normalize, adam);
    l.self_layer->batch = batch;

    l.output_layer = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.output_layer) = make_connected_layer(batch*steps, outputs, outputs, activation, batch_normalize, adam);
    l.output_layer->batch = batch;

    l.outputs = outputs;
    l.output = l.output_layer->output;
    l.delta = l.output_layer->delta;
    //RNN的前向传播，具体实现参考src/rnn_layer.c
    l.forward = forward_rnn_layer;
    //RNN的反向传播，具体实现参考src/rnn_layer.c
    l.backward = backward_rnn_layer;
    //RNN的参数更新，具体实现参考src/rnn_layer.c
    l.update = update_rnn_layer;
#ifdef GPU
    //gpu版本的rnn层的前向传播，具体实现参考src/rnn_layer.c
    l.forward_gpu = forward_rnn_layer_gpu;
    //gpu版本的rnn层的反向传播，具体实现参考src/rnn_layer.c
    l.backward_gpu = backward_rnn_layer_gpu;
    //gpu版本的rnn层的参数更新，具体实现参考src/rnn_layer.c
    l.update_gpu = update_rnn_layer_gpu;
    l.state_gpu = cuda_make_array(0, batch*outputs);
    l.prev_state_gpu = cuda_make_array(0, batch*outputs);
    l.output_gpu = l.output_layer->output_gpu;
    l.delta_gpu = l.output_layer->delta_gpu;
#ifdef CUDNN
    cudnnSetTensor4dDescriptor(l.input_layer->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.input_layer->out_c, l.input_layer->out_h, l.input_layer->out_w); 
    cudnnSetTensor4dDescriptor(l.self_layer->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.self_layer->out_c, l.self_layer->out_h, l.self_layer->out_w); 
    cudnnSetTensor4dDescriptor(l.output_layer->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.output_layer->out_c, l.output_layer->out_h, l.output_layer->out_w); 
#endif
#endif

    return l;
}
//RNN的参数更新。。三个前链接子层的参数更新。。。
void update_rnn_layer(layer l, update_args a)
{
    update_connected_layer(*(l.input_layer),  a);
    update_connected_layer(*(l.self_layer),   a);
    update_connected_layer(*(l.output_layer), a);
}
//RNN前向传播
void forward_rnn_layer(layer l, network net)
{
    //创建一个network的中间变量s
    network s = net;
    s.train = net.train;
    int i;
    //定位其中的三个子层次
    layer input_layer = *(l.input_layer);
    layer self_layer = *(l.self_layer);
    layer output_layer = *(l.output_layer);
    //将相应的delta初始化成0
    fill_cpu(l.outputs * l.batch * l.steps, 0, output_layer.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, self_layer.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, input_layer.delta, 1);
    //如果是在训练，将l.state中的状态值清零
    if(net.train) fill_cpu(l.outputs * l.batch, 0, l.state, 1);
    //循环进行每一个step
    for (i = 0; i < l.steps; ++i) {
        //将输入换成当前层次当前steps下的输入
        s.input = net.input;
        //进行input层的前向传播
        forward_connected_layer(input_layer, s);
        //将s的input换成l.state，实际上也是上一个时刻的状态值
        //具体参考后面的实现
        s.input = l.state;
        //进行self层的前向传播
        forward_connected_layer(self_layer, s);
        //将当前的l.state定位成old_state
        float *old_state = l.state;
        //如果在训练，将l.state的位置往后移动一个steps的位置
        if(net.train) l.state += l.outputs*l.batch;
        //如果存在跳连接，就根据跳连接更新当前l.state指向的位置的相应的值
        if(l.shortcut){
            copy_cpu(l.outputs * l.batch, old_state, 1, l.state, 1);
        }else{
            //没有跳连接的话还是初始化成0
            fill_cpu(l.outputs * l.batch, 0, l.state, 1);
        }
        //这里实现的是将input_layer层的输出加到l.state里面
        axpy_cpu(l.outputs * l.batch, 1, input_layer.output, 1, l.state, 1);
        //然后将self_layer的输出也再一次加到l.state上去，结果写入l.state
        axpy_cpu(l.outputs * l.batch, 1, self_layer.output, 1, l.state, 1);
        //将s的输入换成l.state
        s.input = l.state;
        //进行output_layer的反向传播
        forward_connected_layer(output_layer, s);
        //将相应的input的各个子层次的相关指针向后移动相应的位置
        net.input += l.inputs*l.batch;
        increment_layer(&input_layer, 1);
        increment_layer(&self_layer, 1);
        increment_layer(&output_layer, 1);
    }
}

//RNN的反向传播
void backward_rnn_layer(layer l, network net)
{
    network s = net;
    s.train = net.train;
    int i;
    //定位相应的子层次的位置
    layer input_layer = *(l.input_layer);
    layer self_layer = *(l.self_layer);
    layer output_layer = *(l.output_layer);
    //将相关的指针移动到最后一个steps产生的数据的起始处
    increment_layer(&input_layer, l.steps-1);
    increment_layer(&self_layer, l.steps-1);
    increment_layer(&output_layer, l.steps-1);

    l.state += l.outputs*l.batch*l.steps;
    //从最后一个steps开始往前做反向传播
    for (i = l.steps-1; i >= 0; --i) {
        //这里实现的是将input_layer层的output赋值到l.state中
        copy_cpu(l.outputs * l.batch, input_layer.output, 1, l.state, 1);
        //然后将self_layer层的输出加到l.state上
        axpy_cpu(l.outputs * l.batch, 1, self_layer.output, 1, l.state, 1);
        //将s的input换成l.state
        s.input = l.state;
        //同时delta指向的位置换成self_layer层的delta
        s.delta = self_layer.delta;
        //进行output_layer层的反向传播
        backward_connected_layer(output_layer, s);
        //将l.state的指向向前移动一个steps的位置
        l.state -= l.outputs*l.batch;
        /*
           if(i > 0){
           copy_cpu(l.outputs * l.batch, input_layer.output - l.outputs*l.batch, 1, l.state, 1);
           axpy_cpu(l.outputs * l.batch, 1, self_layer.output - l.outputs*l.batch, 1, l.state, 1);
           }else{
           fill_cpu(l.outputs * l.batch, 0, l.state, 1);
           }
         */
        //将s的输入换成l.state中
        s.input = l.state;
        //将delta换成self层的delta，同时指向上一个steps的delta的存储位置
        s.delta = self_layer.delta - l.outputs*l.batch;
        //如果反向传播到了第一个step，将delta的值置0
        if (i == 0) s.delta = 0;
        //进行self_layer层的反向传播
        backward_connected_layer(self_layer, s);
        //这里将self层的delta赋值到input层的delta中
        copy_cpu(l.outputs*l.batch, self_layer.delta, 1, input_layer.delta, 1);
        //跟前向传播对应，如果需要shortcut的话。。。
        if (i > 0 && l.shortcut) axpy_cpu(l.outputs*l.batch, 1, self_layer.delta, 1, self_layer.delta - l.outputs*l.batch, 1);
        //将相应的input换成对应的steps的输入
        s.input = net.input + i*l.inputs*l.batch;
        //如果当前的层次还存在前一个层次的话
        //将delta换成相应的下一个steps的位置，这里是层间的delta传播。前面delta的相关内容基本上都是层间的不同steps中的传播
        if(net.delta) s.delta = net.delta + i*l.inputs*l.batch;
        else s.delta = 0;
        //进行input层的反向传播
        backward_connected_layer(input_layer, s);
        //将相应层次的指针往前移动一个steps的位置
        increment_layer(&input_layer, -1);
        increment_layer(&self_layer, -1);
        increment_layer(&output_layer, -1);
    }
}

#ifdef GPU

void pull_rnn_layer(layer l)
{
    pull_connected_layer(*(l.input_layer));
    pull_connected_layer(*(l.self_layer));
    pull_connected_layer(*(l.output_layer));
}

void push_rnn_layer(layer l)
{
    push_connected_layer(*(l.input_layer));
    push_connected_layer(*(l.self_layer));
    push_connected_layer(*(l.output_layer));
}
//rnn层gpu版本的参数更新
void update_rnn_layer_gpu(layer l, update_args a)
{
    update_connected_layer_gpu(*(l.input_layer),  a);
    update_connected_layer_gpu(*(l.self_layer),   a);
    update_connected_layer_gpu(*(l.output_layer), a);
}
//rnn层前向传播的gpu实现，对照cpu版本的实现和公式推导
void forward_rnn_layer_gpu(layer l, network net)
{
    network s = {0};
    s.train = net.train;
    int i;
    layer input_layer = *(l.input_layer);
    layer self_layer = *(l.self_layer);
    layer output_layer = *(l.output_layer);

    fill_gpu(l.outputs * l.batch * l.steps, 0, output_layer.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, self_layer.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, input_layer.delta_gpu, 1);

    if(net.train) {
        fill_gpu(l.outputs * l.batch * l.steps, 0, l.delta_gpu, 1);
        copy_gpu(l.outputs*l.batch, l.state_gpu, 1, l.prev_state_gpu, 1);
    }

    for (i = 0; i < l.steps; ++i) {
        s.input_gpu = net.input_gpu;
        forward_connected_layer_gpu(input_layer, s);

        s.input_gpu = l.state_gpu;
        forward_connected_layer_gpu(self_layer, s);

        fill_gpu(l.outputs * l.batch, 0, l.state_gpu, 1);
        axpy_gpu(l.outputs * l.batch, 1, input_layer.output_gpu, 1, l.state_gpu, 1);
        axpy_gpu(l.outputs * l.batch, 1, self_layer.output_gpu, 1, l.state_gpu, 1);

        s.input_gpu = l.state_gpu;
        forward_connected_layer_gpu(output_layer, s);

        net.input_gpu += l.inputs*l.batch;
        increment_layer(&input_layer, 1);
        increment_layer(&self_layer, 1);
        increment_layer(&output_layer, 1);
    }
}
//rnn层反向传播的gpu实现。。对照cpu版本和公式推导
void backward_rnn_layer_gpu(layer l, network net)
{
    network s = {0};
    s.train = net.train;
    int i;
    layer input_layer = *(l.input_layer);
    layer self_layer = *(l.self_layer);
    layer output_layer = *(l.output_layer);
    increment_layer(&input_layer,  l.steps - 1);
    increment_layer(&self_layer,   l.steps - 1);
    increment_layer(&output_layer, l.steps - 1);
    float *last_input = input_layer.output_gpu;
    float *last_self = self_layer.output_gpu;
    for (i = l.steps-1; i >= 0; --i) {
        fill_gpu(l.outputs * l.batch, 0, l.state_gpu, 1);
        axpy_gpu(l.outputs * l.batch, 1, input_layer.output_gpu, 1, l.state_gpu, 1);
        axpy_gpu(l.outputs * l.batch, 1, self_layer.output_gpu, 1, l.state_gpu, 1);

        s.input_gpu = l.state_gpu;
        s.delta_gpu = self_layer.delta_gpu;
        backward_connected_layer_gpu(output_layer, s);

        if(i != 0) {
            fill_gpu(l.outputs * l.batch, 0, l.state_gpu, 1);
            axpy_gpu(l.outputs * l.batch, 1, input_layer.output_gpu - l.outputs*l.batch, 1, l.state_gpu, 1);
            axpy_gpu(l.outputs * l.batch, 1, self_layer.output_gpu - l.outputs*l.batch, 1, l.state_gpu, 1);
        }else {
            copy_gpu(l.outputs*l.batch, l.prev_state_gpu, 1, l.state_gpu, 1);
        }

        copy_gpu(l.outputs*l.batch, self_layer.delta_gpu, 1, input_layer.delta_gpu, 1);

        s.input_gpu = l.state_gpu;
        s.delta_gpu = (i > 0) ? self_layer.delta_gpu - l.outputs*l.batch : 0;
        if (i == 0) s.delta_gpu = 0;
        backward_connected_layer_gpu(self_layer, s);

        s.input_gpu = net.input_gpu + i*l.inputs*l.batch;
        if(net.delta_gpu) s.delta_gpu = net.delta_gpu + i*l.inputs*l.batch;
        else s.delta_gpu = 0;
        backward_connected_layer_gpu(input_layer, s);

        increment_layer(&input_layer,  -1);
        increment_layer(&self_layer,   -1);
        increment_layer(&output_layer, -1);
    }
    fill_gpu(l.outputs * l.batch, 0, l.state_gpu, 1);
    axpy_gpu(l.outputs * l.batch, 1, last_input, 1, l.state_gpu, 1);
    axpy_gpu(l.outputs * l.batch, 1, last_self, 1, l.state_gpu, 1);
}
#endif
