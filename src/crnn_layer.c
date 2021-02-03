#include "crnn_layer.h"
#include "convolutional_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//调整某一层的相关指针的位置
static void increment_layer(layer *l, int steps)
{
    //这里传入的steps是1
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
//创建一个CRNN层实际上就是连续的三个卷积层
//而且最终卷积输出的w,h和输入一致
layer make_crnn_layer(int batch, int h, int w, int c, int hidden_filters, int output_filters, int steps, ACTIVATION activation, int batch_normalize)
{
    fprintf(stderr, "CRNN Layer: %d x %d x %d image, %d filters\n", h,w,c,output_filters);
    //这里的steps是cfg中[net]组里面的time-steps参数。通常是初始化值为1
    //这里将整个batch根据steps划分成了多个更小的batch_step
    batch = batch / steps;
    layer l = {0};
    //batch是cfg中的batch
    //这里的l.batch实际上就是根据steps划分的跟小的batch_step的大小了
    l.batch = batch;
    l.type = CRNN;
    //l.steps表示划分的步长steps
    l.steps = steps;
    l.h = h;
    l.w = w;
    l.c = c;
    l.out_h = h;
    l.out_w = w;
    //CRNN输出通道数
    l.out_c = output_filters;
    l.inputs = h*w*c;
    //这里的hidden_filters是指中间隐藏层的卷积核数量
    //l.hidden是中间隐藏层的输出数据量
    l.hidden = h * w * hidden_filters;
    l.outputs = l.out_h * l.out_w * l.out_c;
    //l.state主要是存储每个batch_steps在隐藏层产生的输出，具体参考forward_crnn_layer中的实现
    //这里的l.hidden*batch*(steps+1) = h*w*hidden_filters*l.batch*l.steps+h*w*hidden_filters*l.batch
    //总体上来说他的内存比整个batch在隐藏层产生的输出数量还要大上一个小batch_step数据在隐藏层产生的输出数据量
    //为什么要多出一个batch_steps的内存从forward_crnn_layer中就能看出来了。。存储中间变量的时候，最开始有一段起始值。。。
    l.state = calloc(l.hidden*batch*(steps+1), sizeof(float));

    l.input_layer = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    //这里创建一个卷积层。。
    //这里之所以采用batch*steps是为了分配足够的内存，避免出问题
    *(l.input_layer) = make_convolutional_layer(batch*steps, h, w, c, hidden_filters, 1, 3, 1, 1,  activation, batch_normalize, 0, 0, 0);
    //然后将其中的batch参数换成了这里的batch_steps
    //这里再把batch调小是为了后面使用的时候。。具体参考forward_crnn_layer中的内容
    l.input_layer->batch = batch;

    l.self_layer = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.self_layer) = make_convolutional_layer(batch*steps, h, w, hidden_filters, hidden_filters, 1, 3, 1, 1,  activation, batch_normalize, 0, 0, 0);
    l.self_layer->batch = batch;

    l.output_layer = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.output_layer) = make_convolutional_layer(batch*steps, h, w, hidden_filters, output_filters, 1, 3, 1, 1,  activation, batch_normalize, 0, 0, 0);
    l.output_layer->batch = batch;
    //指定这里的l.output和l.delta的值
    //最终返回能够得到的l.output指向了output_layer的output，具体参考forward_crnn_layer和backwd_crnn_layer
    //传递到下一层的output数据量是整个batch_size数据的输出
    //虽然在forward_crnn_layer中output_layer的output指针会移动。。但是这里已经记录了起始位置。。。。
    l.output = l.output_layer->output;
    l.delta = l.output_layer->delta;
    //CRNN的前向传播
    //具体参考src/crnn_layer.c
    l.forward = forward_crnn_layer;
    //CRNN的反向传播
    //细节参考src/crnn_layer.c
    l.backward = backward_crnn_layer;
    //CRNN层的参数更新，具体参考src/crnn_layer.c
    l.update = update_crnn_layer;

#ifdef GPU
    //gpu版本的crnn层前向传播，具体实现参考src/crnn_layer.c
    l.forward_gpu = forward_crnn_layer_gpu;
    //gpu版本的crnn层反向传播，具体实现参考src/crnn_layer.c
    l.backward_gpu = backward_crnn_layer_gpu;
    //gpu版本的crnn层参数更新，具体实现参考src/crnn_layer.c
    l.update_gpu = update_crnn_layer_gpu;

    l.state_gpu = cuda_make_array(l.state, l.hidden*batch*(steps+1));
    l.output_gpu = l.output_layer->output_gpu;
    l.delta_gpu = l.output_layer->delta_gpu;
#endif

    return l;
}
//更新crnn层的参数
void update_crnn_layer(layer l, update_args a)
{   //分别更新三个卷积层的参数，具体的实现细节参考src/convolutional_layer.c
    update_convolutional_layer(*(l.input_layer),  a);
    update_convolutional_layer(*(l.self_layer),   a);
    update_convolutional_layer(*(l.output_layer), a);
}
//CRNN层的前向传播
//最终在crnn层返回能够得到的l.output指向了其中output_layer的output的位置，取到全部的batch_size的输出
//实际上就是将一个batch的图片根据步长划分成时序的数据，然后用三个卷积单元模拟rnn的时序操作。。。
//不用考虑的太复杂。。这里就是一个很简单的rnn单元。只不过里面的各种计算和处理换成了三个卷积层，因为输入数据也是类似图片的数据整合方式
void forward_crnn_layer(layer l, network net)
{   //新建一个network结构
    network s = net;
    //提取其中的训练状态标志
    s.train = net.train;
    int i;
    //定位crnn中的三个卷积层
    layer input_layer = *(l.input_layer);
    layer self_layer = *(l.self_layer);
    layer output_layer = *(l.output_layer);
    //fill_cpu具体实现参考src/blas.c
    //这里实现的是将crnn中三个卷积层里面的delta初始化成0
    fill_cpu(l.outputs * l.batch * l.steps, 0, output_layer.delta, 1);
    fill_cpu(l.hidden * l.batch * l.steps, 0, self_layer.delta, 1);
    fill_cpu(l.hidden * l.batch * l.steps, 0, input_layer.delta, 1);
    //如果正在训练将l.state的前l.hidden*l.batch个位置初始化成0
    //注意一下这里的l.batch跟其他层次的l.batch不同，具体参考make_crnn_layer里面的操作
    //将最前面存储一个batch_steps数据产生的输出置0
    if(net.train) fill_cpu(l.hidden * l.batch, 0, l.state, 1);
    //在不同的l.steps下
    for (i = 0; i < l.steps; ++i) {
        //提取当前的crnn层的输入特征图位置
        s.input = net.input;
        //卷积的前向传播，细节参考src/convolutional_layer.c
        //这里的前向传播的时候，input_layer的batch参数已经是小的batch_steps了
        //这里的inputlayer就是转换一下输入数据
        forward_convolutional_layer(input_layer, s);
        //更改s.input的值为l.state
        //进行self_layer的前向传播
        //如果这时候的i==0的话，实际上这里的l.state中前面一部分已经在开始的时候置0了
        s.input = l.state;
        //这里从输入能够判断，输入实际上是上一个step的状态
        //所以这里实际上可以看成是rnn里对状态值进行处理的部分
        forward_convolutional_layer(self_layer, s);
        //记录上一轮l.state的位置
        float *old_state = l.state;
        //如果在训练，更新l.state到下一个位置
        if(net.train) l.state += l.hidden*l.batch;
        //如果存在shortcut，cfg中的shortcut参数，通常初始化为0
        if(l.shortcut){
            //实现的是将old_state位置开始的l.hidden * l.batch数量的值复制到当前位置的l.state中
            copy_cpu(l.hidden * l.batch, old_state, 1, l.state, 1);
        }else{
            //如果不存在shortcut
            //将l.state后面相应的batch_steps数据位置0
            fill_cpu(l.hidden * l.batch, 0, l.state, 1);
        }
        //axpy_cpu的具体实现参考src/blas.c
        //将input_layer和self_layer产生的输出累加到l.state中
        //生成下一个状态值
        axpy_cpu(l.hidden * l.batch, 1, input_layer.output, 1, l.state, 1);
        axpy_cpu(l.hidden * l.batch, 1, self_layer.output, 1, l.state, 1);
        //将上面两层输出的累加作为最后output_layer的输入
        s.input = l.state;
        //利用一个卷积层生成当前step的输出
        forward_convolutional_layer(output_layer, s);
        //更新net.input的值到下一个batch_steps
        net.input += l.inputs*l.batch;
        //increment_layer的具体实现参考src/crnn_layer.c
        //调整三个卷积层内部相关指针的位置到下一个batch-steps
        increment_layer(&input_layer, 1);
        increment_layer(&self_layer, 1);
        increment_layer(&output_layer, 1);
    }
}
//crnn层的反向传播 
void backward_crnn_layer(layer l, network net)
{
    //定义一个新的network结构作为中间变量
    network s = net;
    int i;
    //和前向传播中一样，定位三个卷积单元
    layer input_layer = *(l.input_layer);
    layer self_layer = *(l.self_layer);
    layer output_layer = *(l.output_layer);
    
    //跟前向传播对应。。 
    //这里的目的是将三个层的output指针指向最后一个steps的输出数据的起始位置
    increment_layer(&input_layer, l.steps-1);
    increment_layer(&self_layer, l.steps-1);
    increment_layer(&output_layer, l.steps-1);
    l.state += l.hidden*l.batch*l.steps;
    //从后往前遍历l.state的内容
    //也就是反向传播的时候从最后一个steps往前看
    for (i = l.steps-1; i >= 0; --i) {
        //这里实现的是将input_layer.output中的值赋值到l.state中去，这里也只是赋值了一个batch_steps数据的值
        copy_cpu(l.hidden * l.batch, input_layer.output, 1, l.state, 1);
        //这里实现的是将self_layer.output中的值加到上面提取的input_layer.output上，结果存入l.state
        //作为反向传播中的状态值，实际上是inpout层最后一个steps的输出和self层最后一个steps输出的和
        //作为反向传播开始时的前一个状态值
        axpy_cpu(l.hidden * l.batch, 1, self_layer.output, 1, l.state, 1);

        s.input = l.state;
        //指定self层梯度存储的位置
        s.delta = self_layer.delta;
        //进行其中一个卷积层的反向传播，具体实现参考src/convolutional_layer.c
        //由于RNN的特点，这里最先进行的时output_layer的反向传播，然后其上一层从前向传播就能看出是self层
        //所以这里的s.delta指定到了self_layer
        //这里output_layer的输入从前向传播能看出input实际上就是这里l.state指向的两层输出的叠加，然后他的上一层是self层
        //则delta指向self层
        backward_convolutional_layer(output_layer, s);
        //这个时候将l.state往前移动一个step
        l.state -= l.hidden*l.batch;
        /*
           if(i > 0){
           copy_cpu(l.hidden * l.batch, input_layer.output - l.hidden*l.batch, 1, l.state, 1);
           axpy_cpu(l.hidden * l.batch, 1, self_layer.output - l.hidden*l.batch, 1, l.state, 1);
           }else{
           fill_cpu(l.hidden * l.batch, 0, l.state, 1);
           }
         */
        //接着进行self层的反向传播，从前向传播中能看到，self层的上一层实际上是上一个step的self层
        //他的输入就是上一个state的里面存储的输入，这个时候state中存储的内容已经是input和相关状态叠加后的结果了
        //从上面的前向传播能看到
        //所以上面将l.state向前移动，这里将其指定作为输入
        //同样的delta的位置也在self层的delta存储位置向前移动一个step即可
        s.input = l.state;
        s.delta = self_layer.delta - l.hidden*l.batch;
        //i==0表示进入这个rnn的第一个steps了，直接赋值为0即可
        //因为这时候的self_layer前面没有其他的self层了
        if (i == 0) s.delta = 0;
        //完成self层的反向传播
        backward_convolutional_layer(self_layer, s);
        //这里实现的是将self_layer中的梯度赋值到input_layer的梯度中去
        copy_cpu(l.hidden*l.batch, self_layer.delta, 1, input_layer.delta, 1);
        //如果不是反向传播到第一个steps,将上面反向传播得到的self的梯度传到前一个step的self层的梯度里面去，完成delta的传递,
        //这里是在一层crnn内部，两个相邻的step中的self层的delta传递
        if (i > 0 && l.shortcut) axpy_cpu(l.hidden*l.batch, 1, self_layer.delta, 1, self_layer.delta - l.hidden*l.batch, 1);
        //然后进行input层的反向传播，从前向传播可以看出，input层的输入其实是上一层的输出，也就是这里net.input指向的值，根据steps调整到相应位置
        s.input = net.input + i*l.inputs*l.batch;
        //如果不是反向传播到第一个steps，将梯度传递到上一个crnn层的delta里，完成层间的delta传递
        if(net.delta) s.delta = net.delta + i*l.inputs*l.batch;
        //反向传播到了第一个steps,前面没有更多的steps了，直接置0
        else s.delta = 0;
        //进行input_layer的反向传播
        backward_convolutional_layer(input_layer, s);
        //然后将相应层次的数据指向往前一个steps处移动
        increment_layer(&input_layer, -1);
        increment_layer(&self_layer, -1);
        increment_layer(&output_layer, -1);
    }
}

#ifdef GPU

void pull_crnn_layer(layer l)
{
    pull_convolutional_layer(*(l.input_layer));
    pull_convolutional_layer(*(l.self_layer));
    pull_convolutional_layer(*(l.output_layer));
}

void push_crnn_layer(layer l)
{
    push_convolutional_layer(*(l.input_layer));
    push_convolutional_layer(*(l.self_layer));
    push_convolutional_layer(*(l.output_layer));
}

//gpu版本的crnn层的参数更新
void update_crnn_layer_gpu(layer l, update_args a)
{   
    //update_convolutional_layer_gpu的具体实现参考src/convolution_kernels.cu
    update_convolutional_layer_gpu(*(l.input_layer),  a);
    update_convolutional_layer_gpu(*(l.self_layer),   a);
    update_convolutional_layer_gpu(*(l.output_layer), a);
}
//gpu版本的crnn的前向传播
void forward_crnn_layer_gpu(layer l, network net)
{
    //创建一个network中间变量
    network s = net;
    int i;
    //定位相应的子层次
    layer input_layer = *(l.input_layer);
    layer self_layer = *(l.self_layer);
    layer output_layer = *(l.output_layer);
    //将相应的delta_gpu初始化为0
    fill_gpu(l.outputs * l.batch * l.steps, 0, output_layer.delta_gpu, 1);
    fill_gpu(l.hidden * l.batch * l.steps, 0, self_layer.delta_gpu, 1);
    fill_gpu(l.hidden * l.batch * l.steps, 0, input_layer.delta_gpu, 1);
    //如果是训练的化，将l.state_gpu中第一个steps数据下的值置0
    if(net.train) fill_gpu(l.hidden * l.batch, 0, l.state_gpu, 1);

    //循环进行每一个steps
    for (i = 0; i < l.steps; ++i) {
        //将s的input换成net.input。。。
        s.input_gpu = net.input_gpu;
        //进行input_layer层的前向传播
        //具体实现参考src/convolutional_kernels.cu
        forward_convolutional_layer_gpu(input_layer, s);

        //将s的input换成l.state_gpu。。实际上是上一个steps的中间状态值，具体参考后面的实现
        s.input_gpu = l.state_gpu;
        //进行self_layer层的前向传播
        forward_convolutional_layer_gpu(self_layer, s);

        //将当前state的指针记录下来
        float *old_state = l.state_gpu;
        //如果在训练的话，将l.state_gpu的位置向后移动一个steps数据的位置
        if(net.train) l.state_gpu += l.hidden*l.batch;
        if(l.shortcut){
            //存在跳连接的话。。。
            //将old_state中的值复制到l.state_gpu中
            copy_gpu(l.hidden * l.batch, old_state, 1, l.state_gpu, 1);
        }else{
            //不存在跳连接的话，将l.state_gpu中的值置为0
            fill_gpu(l.hidden * l.batch, 0, l.state_gpu, 1);
        }
        //这里实现的是将input_layer的输出加到l.state_gpu上
        //axpy_gpu的具体实现参考src/blas_kernels.cu
        axpy_gpu(l.hidden * l.batch, 1, input_layer.output_gpu, 1, l.state_gpu, 1);
        //然后将self_layer层的输出也加到l.state_gpu上去
        //从这里也看出来了实际上l.state_gpu中存储的就是上一个steps的中间状态
        axpy_gpu(l.hidden * l.batch, 1, self_layer.output_gpu, 1, l.state_gpu, 1);

        //将s的input换成l.state_gpu
        s.input_gpu = l.state_gpu;
        //进行output_layer层的前向传播
        forward_convolutional_layer_gpu(output_layer, s);

        //将相应层次的指针向后移动一个steps数据的位置
        net.input_gpu += l.inputs*l.batch;
        increment_layer(&input_layer, 1);
        increment_layer(&self_layer, 1);
        increment_layer(&output_layer, 1);
    }
}

//gpu版本的crnn层的反向传播
void backward_crnn_layer_gpu(layer l, network net)
{
    //还是先创建一个network中间变量
    network s = net;
    s.train = net.train;
    int i;
    //定位相应的子层次的位置
    layer input_layer = *(l.input_layer);
    layer self_layer = *(l.self_layer);
    layer output_layer = *(l.output_layer);
    //将相应的指针指向最后一个steps数据的起始位置
    increment_layer(&input_layer,  l.steps - 1);
    increment_layer(&self_layer,   l.steps - 1);
    increment_layer(&output_layer, l.steps - 1);
    //将state_gpu的指针移动到最后一个steps隐藏数据的起始位置
    l.state_gpu += l.hidden*l.batch*l.steps;
    //从后往前进行每一个steps的反向传播
    for (i = l.steps-1; i >= 0; --i) {
        //将input_layer层的输出复制到l.state_gpu上
        copy_gpu(l.hidden * l.batch, input_layer.output_gpu, 1, l.state_gpu, 1);
        //将self_layer层的输出加到l.state_gpu上
        axpy_gpu(l.hidden * l.batch, 1, self_layer.output_gpu, 1, l.state_gpu, 1);

        //将s的input换成l.state_gpu的值
        s.input_gpu = l.state_gpu;
        //将s.delta_gpu指向self_layer.delta_gpu
        s.delta_gpu = self_layer.delta_gpu;
        //进行output_layer的反向传播
        backward_convolutional_layer_gpu(output_layer, s);

        //将l.state_gpu的指针向前移动一个steps的位置
        l.state_gpu -= l.hidden*l.batch;

        //将s的输入换成l.state_gpu
        s.input_gpu = l.state_gpu;
        //将s.delta_gpu的指针指向self_layer层的前一个steps的delta，这里完成的实际上是一个层内各个steps之间的delta的传递
        s.delta_gpu = self_layer.delta_gpu - l.hidden*l.batch;
        //如果反向传播到了第一个steps，将s.delta_gpu置0
        if (i == 0) s.delta_gpu = 0;
        //进行self_layer层的反向传播
        backward_convolutional_layer_gpu(self_layer, s);

        //将self_layer层的delta复制到input_layer层的delta上，实际上只是复制了一个steps的数据量
        copy_gpu(l.hidden*l.batch, self_layer.delta_gpu, 1, input_layer.delta_gpu, 1);
        //如果存在短链接，将self_layer层的delta加到self_layer层上一个step的delta里面，和前向传播也是对应的
        if (i > 0 && l.shortcut) axpy_gpu(l.hidden*l.batch, 1, self_layer.delta_gpu, 1, self_layer.delta_gpu - l.hidden*l.batch, 1);
        //将s的input换成对应的下一个循环需要用的nei.input_gpu
        s.input_gpu = net.input_gpu + i*l.inputs*l.batch;
        //如果net.delta_gpu存志。。。将s.delta_gpu换成net.delta_gpu相应steps的位置上去
        if(net.delta_gpu) s.delta_gpu = net.delta_gpu + i*l.inputs*l.batch;
        //如果到了第一个大的层次，network中不再有更前面的层了，s.delta_gpu置0
        else s.delta_gpu = 0;
        //进行input层的反向传播，完成层间的delta的传递
        backward_convolutional_layer_gpu(input_layer, s);
        //将相应层次的指针向前移动一个steps数据的位置。。。
        increment_layer(&input_layer,  -1);
        increment_layer(&self_layer,   -1);
        increment_layer(&output_layer, -1);
    }
}
#endif
