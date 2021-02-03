#include "lstm_layer.h"
#include "connected_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//调整各层次相应指针的steo指向位置
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
//创建一个lstm层
//具体的算法层面的理论推导参考https://www.cnblogs.com/liujshi/p/6159007.html
//这里的实现实际上是对理论推导的拆解
//感觉这里的实现有一点点问题。。。没有l.delta的内存分配，但是后面用到了
//整个前向和反向的过程对照着公式推导去看。。。。还是比较容易看明白的
layer make_lstm_layer(int batch, int inputs, int outputs, int steps, int batch_normalize, int adam)
{
    fprintf(stderr, "LSTM Layer: %d inputs, %d outputs\n", inputs, outputs);
    //跟普通的RNN和GRU一样，这里也是将一个batch的输入分成了steps个输入
    batch = batch / steps;
    layer l = { 0 };
    l.batch = batch;
    l.type = LSTM;
    l.steps = steps;
    l.inputs = inputs;
    //这里用多个全连接层模拟lstm的内部操作，和gru类似
    l.uf = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.uf) = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize, adam);
    l.uf->batch = batch;

    l.ui = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.ui) = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize, adam);
    l.ui->batch = batch;

    l.ug = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.ug) = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize, adam);
    l.ug->batch = batch;

    l.uo = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.uo) = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize, adam);
    l.uo->batch = batch;

    l.wf = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.wf) = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize, adam);
    l.wf->batch = batch;

    l.wi = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.wi) = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize, adam);
    l.wi->batch = batch;

    l.wg = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.wg) = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize, adam);
    l.wg->batch = batch;

    l.wo = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.wo) = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize, adam);
    l.wo->batch = batch;
    //l.delta的内存分配原来没有。。但是还是应该补上。。
    l.delta = calloc(outputs*batch*steps, sizeof(float));
    l.batch_normalize = batch_normalize;
    l.outputs = outputs;
    //这里output的维度自然是大的batch_size的输出产生的维度
    //outputs代表的是网络层次在理论推理上产生的维度
    //从后面的实现能看到这里存储的就是当前层次的隐层状态h_t
    l.output = calloc(outputs*batch*steps, sizeof(float));
    
    l.state = calloc(outputs*batch, sizeof(float));
    //lstm的前向传播，具体实现参考src/lstm_layer.c
    l.forward = forward_lstm_layer;
    //lstm的反向传播，具体实现参考src/lstm_layer.c
    l.backward = backward_lstm_layer;
    //lstm的参数更新，具体实现参考src/lstm_layer.c
    l.update = update_lstm_layer;

    l.prev_state_cpu =  calloc(batch*outputs, sizeof(float));
    l.prev_cell_cpu =   calloc(batch*outputs, sizeof(float));
    //从后面的实现中能看到这里存储的是当前层次的遗忘状态值
    l.cell_cpu =        calloc(batch*outputs*steps, sizeof(float));

    l.f_cpu =           calloc(batch*outputs, sizeof(float));
    l.i_cpu =           calloc(batch*outputs, sizeof(float));
    l.g_cpu =           calloc(batch*outputs, sizeof(float));
    l.o_cpu =           calloc(batch*outputs, sizeof(float));
    l.c_cpu =           calloc(batch*outputs, sizeof(float));
    l.h_cpu =           calloc(batch*outputs, sizeof(float));
    l.temp_cpu =        calloc(batch*outputs, sizeof(float));
    l.temp2_cpu =       calloc(batch*outputs, sizeof(float));
    l.temp3_cpu =       calloc(batch*outputs, sizeof(float));
    l.dc_cpu =          calloc(batch*outputs, sizeof(float));
    l.dh_cpu =          calloc(batch*outputs, sizeof(float));

#ifdef GPU
    //gpu版本的lstm的前向传播，具体实现参考src/lstm_layer.c
    l.forward_gpu = forward_lstm_layer_gpu;
    //gpu版本的lstm的反向传播，具体实现参考src/lstm_layer.c
    l.backward_gpu = backward_lstm_layer_gpu;
    //gpu版本的lstm的参数更新，具体实现参考src/lstm_layer.c
    l.update_gpu = update_lstm_layer_gpu;

    l.output_gpu = cuda_make_array(0, batch*outputs*steps);
    l.delta_gpu = cuda_make_array(0, batch*l.outputs*steps);

    l.prev_state_gpu = cuda_make_array(0, batch*outputs);
    l.prev_cell_gpu = cuda_make_array(0, batch*outputs);
    l.cell_gpu = cuda_make_array(0, batch*outputs*steps);

    l.f_gpu = cuda_make_array(0, batch*outputs);
    l.i_gpu = cuda_make_array(0, batch*outputs);
    l.g_gpu = cuda_make_array(0, batch*outputs);
    l.o_gpu = cuda_make_array(0, batch*outputs);
    l.c_gpu = cuda_make_array(0, batch*outputs);
    l.h_gpu = cuda_make_array(0, batch*outputs);
    l.temp_gpu =  cuda_make_array(0, batch*outputs);
    l.temp2_gpu = cuda_make_array(0, batch*outputs);
    l.temp3_gpu = cuda_make_array(0, batch*outputs);
    l.dc_gpu = cuda_make_array(0, batch*outputs);
    l.dh_gpu = cuda_make_array(0, batch*outputs);
#ifdef CUDNN
        cudnnSetTensor4dDescriptor(l.wf->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.wf->out_c, l.wf->out_h, l.wf->out_w); 
        cudnnSetTensor4dDescriptor(l.wi->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.wi->out_c, l.wi->out_h, l.wi->out_w); 
        cudnnSetTensor4dDescriptor(l.wg->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.wg->out_c, l.wg->out_h, l.wg->out_w); 
        cudnnSetTensor4dDescriptor(l.wo->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.wo->out_c, l.wo->out_h, l.wo->out_w); 

        cudnnSetTensor4dDescriptor(l.uf->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.uf->out_c, l.uf->out_h, l.uf->out_w); 
        cudnnSetTensor4dDescriptor(l.ui->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.ui->out_c, l.ui->out_h, l.ui->out_w); 
        cudnnSetTensor4dDescriptor(l.ug->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.ug->out_c, l.ug->out_h, l.ug->out_w); 
        cudnnSetTensor4dDescriptor(l.uo->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.uo->out_c, l.uo->out_h, l.uo->out_w); 
#endif

#endif

    return l;
}
//更新lstm层的参数
void update_lstm_layer(layer l, update_args a)
{
    update_connected_layer(*(l.wf), a);
    update_connected_layer(*(l.wi), a);
    update_connected_layer(*(l.wg), a);
    update_connected_layer(*(l.wo), a);
    update_connected_layer(*(l.uf), a);
    update_connected_layer(*(l.ui), a);
    update_connected_layer(*(l.ug), a);
    update_connected_layer(*(l.uo), a);
}
//lstm的前向传播
//需要注意一下的是这里的实现忽略了各层次的偏置b
void forward_lstm_layer(layer l, network state)
{   
    //创建一个中转的network结构s
    network s = { 0 };
    s.train = state.train;
    int i;
    //定位内部各个子层次的位置
    layer wf = *(l.wf);
    layer wi = *(l.wi);
    layer wg = *(l.wg);
    layer wo = *(l.wo);

    layer uf = *(l.uf);
    layer ui = *(l.ui);
    layer ug = *(l.ug);
    layer uo = *(l.uo);
    //fill_cpu的具体实现参考src/blas.c
    //这里一段实现的是将各个层次的delta置0，进行初始话
    fill_cpu(l.outputs * l.batch * l.steps, 0, wf.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, wi.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, wg.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, wo.delta, 1);

    fill_cpu(l.outputs * l.batch * l.steps, 0, uf.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, ui.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, ug.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, uo.delta, 1);
    //如果是在训练
    if (state.train) {
        //将这一个层次的l.delta置0
        fill_cpu(l.outputs * l.batch * l.steps, 0, l.delta, 1);
    }
    //通过循环实现每一个step的进行
    //这里要注意的是lstm中的l.h_cpu表示的是上一个单元的隐藏状态，然后就是上一层的遗忘状态值用其他的表示，从后面的实现中能看出
    //这个上一层的遗忘状态值用l.c_cpu来表示
    //这里内部推导的拆分和gru是类似的
    for (i = 0; i < l.steps; ++i) {
        //将s的input换成l.h_cpu
        s.input = l.h_cpu;
        //这里的4个前向传播使用前一层的隐层状态h_t-1作为输入
        //具体表征的意义结合前向传播的公式推导很容易看出来
        forward_connected_layer(wf, s);							
        forward_connected_layer(wi, s);							
        forward_connected_layer(wg, s);							
        forward_connected_layer(wo, s);							
        //然后将输入换成当前时刻的输入数据x_t
        s.input = state.input;
        //这里的4个前向传播使用当前时刻的输入数据x_t来进行操作
        //代表的意义同样可以参考lstm的公式得到。。实际上是和上面4个相互对应的
        forward_connected_layer(uf, s);							
        forward_connected_layer(ui, s);							
        forward_connected_layer(ug, s);							
        forward_connected_layer(uo, s);	
        //后面4组操作就能看到上面的8层之间的对应联系了						
        //将wf.out的输出赋值到l.f_cpu中
        copy_cpu(l.outputs*l.batch, wf.output, 1, l.f_cpu, 1);
        //这里实现的是将uf.output中的值加到l.f_cpu中去，结果写入l.f_cpu
        axpy_cpu(l.outputs*l.batch, 1, uf.output, 1, l.f_cpu, 1);
        //这里实现的是将wi.output的值赋值到l.i_cpu中
        copy_cpu(l.outputs*l.batch, wi.output, 1, l.i_cpu, 1);	
        //然后将ui.output中的值加到l.i_cpu中去，结果写入l.i_cpu中
        axpy_cpu(l.outputs*l.batch, 1, ui.output, 1, l.i_cpu, 1);	
        //同样的，将wg.output中的值赋值到l.g_cpu中
        copy_cpu(l.outputs*l.batch, wg.output, 1, l.g_cpu, 1);	
        //同时将ug.output中的结果加到l.g_cpu里面，结果写入l.g_cpu内
        axpy_cpu(l.outputs*l.batch, 1, ug.output, 1, l.g_cpu, 1);	
        //这里将wo.output的值赋值到l.o_cpu中
        copy_cpu(l.outputs*l.batch, wo.output, 1, l.o_cpu, 1);
        //然后将uo.output中的值加到l.o_cpu内，结果写入l.o_cpu中	
        axpy_cpu(l.outputs*l.batch, 1, uo.output, 1, l.o_cpu, 1);	
        //通过相应的激活函数
        //分别对l.f_cpu，l.i_cpu，l.g_cpu和l.o_cpu进行处理
        //注意激活函数的不同，这个参考具体的公式推导
        activate_array(l.f_cpu, l.outputs*l.batch, LOGISTIC);		
        activate_array(l.i_cpu, l.outputs*l.batch, LOGISTIC);		
        activate_array(l.g_cpu, l.outputs*l.batch, TANH);			
        activate_array(l.o_cpu, l.outputs*l.batch, LOGISTIC);		
        //这里是将l.i_cpu中的值赋值到l.temp_cpu中
        //为什么要额外用一个temp来转换是因为l.i_cpu中的值要存下来进行反向传播的时候使用，不能被重写
        copy_cpu(l.outputs*l.batch, l.i_cpu, 1, l.temp_cpu, 1);	
        //mul_cpu的具体实现参考src/blas.c这里是实现的是将l.temp_cpu中的值和l.g_cpu中的值相乘，结果写入l.temp_cpu，实际上也是内部每一个对应的值进行相乘
        mul_cpu(l.outputs*l.batch, l.g_cpu, 1, l.temp_cpu, 1);	
        //这里是将l.c_cpu中的值和l.f_cpu中的值相乘，结果写入l.c_cpu
        //这里的l.c_cpu表示上一层的遗忘状态值，lstm接收上一层的两个输入，一个是隐层状态h_t-1，另一个是遗忘状态s_t-1，这里的l.c_cpu实际上表示的就是s_t-1或者s_t
        mul_cpu(l.outputs*l.batch, l.f_cpu, 1, l.c_cpu, 1);		
        //然后这里实现的是将l.c_cpu中得知和l.temp_cpu中的值相加，结果写入l.c_cpu中，完成了对这一层遗忘状态s_t的更新
        axpy_cpu(l.outputs*l.batch, 1, l.temp_cpu, 1, l.c_cpu, 1);	
        //这里是将l.c_cpu中的值赋值到l.h_cpu中
        copy_cpu(l.outputs*l.batch, l.c_cpu, 1, l.h_cpu, 1);
        //然后这里是将l.h_cpu中的值进行相应的激活，结果写入l.h_cpu			
        activate_array(l.h_cpu, l.outputs*l.batch, TANH);
        //这里是将l.h_cpu的值和l.o_cpu的值相乘，结果写入l.h_cpu完成对这一层隐藏状态h_t的更新		
        mul_cpu(l.outputs*l.batch, l.o_cpu, 1, l.h_cpu, 1);	
        //然后再将l.c_cpu中的值赋值到l.cell_cpu中，也就是将遗忘状态通过l.cell_cpu传递到下一层
        copy_cpu(l.outputs*l.batch, l.c_cpu, 1, l.cell_cpu, 1);	
        //同样的利用l.ouput将当前曾的隐藏状态l.h_cpu传递到下一层	
        copy_cpu(l.outputs*l.batch, l.h_cpu, 1, l.output, 1);
        //接着就是循环step的过程了，将各个层次的位置往后移一个对应位置
        state.input += l.inputs*l.batch;
        l.output    += l.outputs*l.batch;
        l.cell_cpu      += l.outputs*l.batch;

        increment_layer(&wf, 1);
        increment_layer(&wi, 1);
        increment_layer(&wg, 1);
        increment_layer(&wo, 1);

        increment_layer(&uf, 1);
        increment_layer(&ui, 1);
        increment_layer(&ug, 1);
        increment_layer(&uo, 1);
    }
}

//lstm的反向传播
//这里的实现过程最好结合前向传播和反向传播的公式推导来看。。结合公式理解起来其实并不难
//需要注意的是lstm层里面，输出到下一个层次的除了隐层状态h_t以外还有一个遗忘状态c_t，但是c_t从公式推导能看出有其他模块组成
//所以最终反向传播中传递到上一个lstm层的梯度l.delta只包含了隐层状态h的梯度。至于遗忘状态的梯度为什么不传递。。。实际上从公式推导就能看出
//遗忘状态的梯度可以很容易的由隐层状态的梯度推导出来，所以传不传都是一样的。。。
void backward_lstm_layer(layer l, network state)
{
    //这里还是创建一个network作为中间变量
    network s = { 0 };
    s.train = state.train;
    int i;
    //定位多个子层次的位置
    layer wf = *(l.wf);
    layer wi = *(l.wi);
    layer wg = *(l.wg);
    layer wo = *(l.wo);

    layer uf = *(l.uf);
    layer ui = *(l.ui);
    layer ug = *(l.ug);
    layer uo = *(l.uo);
    //将相应的指针调整到最后一个steps产生的数据的开始
    //关于这里为什么从一开始进行移动，是应为src/network.c中传入的是当前层次的副本，具体参考src/network.c中的forward_network函数
    increment_layer(&wf, l.steps - 1);
    increment_layer(&wi, l.steps - 1);
    increment_layer(&wg, l.steps - 1);
    increment_layer(&wo, l.steps - 1);

    increment_layer(&uf, l.steps - 1);
    increment_layer(&ui, l.steps - 1);
    increment_layer(&ug, l.steps - 1);
    increment_layer(&uo, l.steps - 1);
    //同时也将输入的input也就是当前单元的输入数据x_t调整到对应位置
    state.input += l.inputs*l.batch*(l.steps - 1);
    //相应的net.delta的位置也要进行调整
    if (state.delta) state.delta += l.inputs*l.batch*(l.steps - 1);
    //然后就是当前层次的output和delta以及cell_cpu等等都要调整到最后一个steps开始的位置
    l.output += l.outputs*l.batch*(l.steps - 1);
    l.cell_cpu += l.outputs*l.batch*(l.steps - 1);

    l.delta += l.outputs*l.batch*(l.steps - 1);
    //将从最后一个steps开始进行反向传播
    for (i = l.steps - 1; i >= 0; --i) {
        //如果没有反向传播到第一个steps
        //这里是将l.cell_cpu中对应位置的值赋值到l.prev_cell_cpu中去
        //这里假如是第一次进入这个循环，那么此时l.cell_cpu指向的是l.cell_cpu中最后一个steps数据的开始
        //这里向l.prev_cell_cpu赋值的是上一个时刻产生的遗忘状态
        if (i != 0) copy_cpu(l.outputs*l.batch, l.cell_cpu - l.outputs*l.batch, 1, l.prev_cell_cpu, 1);
        //这里将当前时刻的遗忘状态从l.cell_cpu中赋值到l.c_cpu内
        copy_cpu(l.outputs*l.batch, l.cell_cpu, 1, l.c_cpu, 1);
        //同样的，如果没有反向传播到第一个steps
        //就将output中上一个时刻的h_t-1的状态值赋值到l.prev_state_cpu
        if (i != 0) copy_cpu(l.outputs*l.batch, l.output - l.outputs*l.batch, 1, l.prev_state_cpu, 1);
        //这里是将l.output中当前时刻的隐藏状态值赋值到l.h_cpu中
        copy_cpu(l.outputs*l.batch, l.output, 1, l.h_cpu, 1);
        //这里可以看成是对l.delta的初始化吧，如果已经到了第一个steps那么直接将delta置0，因为第一个steps前不再有其他的steps了
        //所以这里直接置0即可，如果不是到了第一个steps，则将l.dh_cpu初始化到指向上一个steps的delta的位置上去
        //这里l.delta指向的是当前层的delta
        l.dh_cpu = (i == 0) ? 0 : l.delta - l.outputs*l.batch;
        //这里的几组操作实际上从前向传播的过程也能看出来，实际上也是根据公式求偏导数的过程，和前向传播对应起来的
        //一番操作之后，在进入后面的activate_array之前，l.f_cpu，l.i_cpu，l.g_cpu，l.o_cpu中的值变成了相应steps下相关操作输入的加权和，这个参考前向计算的公式就能看出
        //这里是将wf.output中相应steps的值赋值到l.f_cpu中
        copy_cpu(l.outputs*l.batch, wf.output, 1, l.f_cpu, 1);	
        //然后将uf.output中相应的值加到l.f_cpu上，然后结果写入l.f_cpu中		
        axpy_cpu(l.outputs*l.batch, 1, uf.output, 1, l.f_cpu, 1);			
        //这里实现的是将wi.output中的值赋值到l.i_cpu中去
        copy_cpu(l.outputs*l.batch, wi.output, 1, l.i_cpu, 1);	
        //然后将ui.output中的值加到l.i_cpu上，结果写入l.i_cpu		
        axpy_cpu(l.outputs*l.batch, 1, ui.output, 1, l.i_cpu, 1);			
        //这里是将wg.output中的值赋值到l.g_cpu中
        copy_cpu(l.outputs*l.batch, wg.output, 1, l.g_cpu, 1);	
        //然后将ug.input中的值加到l.g_cpu内，结果写到l.g_cpu		
        axpy_cpu(l.outputs*l.batch, 1, ug.output, 1, l.g_cpu, 1);			
        //将wo.output中的值赋值到l.o_cpu内
        copy_cpu(l.outputs*l.batch, wo.output, 1, l.o_cpu, 1);	
        //接着将uo.output中的值加到l.o_cpu上		
        axpy_cpu(l.outputs*l.batch, 1, uo.output, 1, l.o_cpu, 1);			

        //承接上面，分别计算对应激活函数相对于l.f_cpu,l.i_cpu，l.g_cpu，l.o_cpu的偏导数
        //结果写入l.f_cpu等之中
        activate_array(l.f_cpu, l.outputs*l.batch, LOGISTIC);			
        activate_array(l.i_cpu, l.outputs*l.batch, LOGISTIC);		
        activate_array(l.g_cpu, l.outputs*l.batch, TANH);			
        activate_array(l.o_cpu, l.outputs*l.batch, LOGISTIC);		
        //这里实现的是将l.delta中相应steps值赋值到l.temp3_cpu中
        copy_cpu(l.outputs*l.batch, l.delta, 1, l.temp3_cpu, 1);		
        //这里是将l.c_cpu内的值赋值到l.temp_cpu中，也就是当前时刻的遗忘状态
        copy_cpu(l.outputs*l.batch, l.c_cpu, 1, l.temp_cpu, 1);	
        //对l.temp_cpu计算激活，结果写入l.temp_cpu		
        activate_array(l.temp_cpu, l.outputs*l.batch, TANH);			
        //这里将l.temp3_cpu中的值赋值到l.temp2_cpu中
        copy_cpu(l.outputs*l.batch, l.temp3_cpu, 1, l.temp2_cpu, 1);
        //这里是将l.o_cpu中的值和l.temp2_cpu中的值相乘，结果写入l.temp2_cpu中		
        mul_cpu(l.outputs*l.batch, l.o_cpu, 1, l.temp2_cpu, 1);			
        //然后对l.temp_cpu求激活函数的导数，结果写入l.temp2_cpu,就是把l.temp_cpu对相应激活的导数跟原来l.temp2_cpu中的值相乘，结果写入l.temp2_cpu
        gradient_array(l.temp_cpu, l.outputs*l.batch, TANH, l.temp2_cpu);
        //这里实现的是将l.dc_cpu中的值加到l.temp2_cpu上去，结果写入l.temp2_cpu
        //这一步结束，l.temp2_cpu中存储的就是损失函数相对于当前节点当前steps下的遗忘状态的偏导数
        //这里的l.dc_cpu里面存储的是损失函数相对于当前节点当前steps下前一个steps的遗忘状态的偏导数
        //在后面能看到l.dc_cpu的计算方式，就这一行而言，这里的l.dc_cpu代表的是方向传播中比如说现在是step4，那么这里就是step5的时候计算的s_t-1的偏导
        axpy_cpu(l.outputs*l.batch, 1, l.dc_cpu, 1, l.temp2_cpu, 1);		
        //这里是将l.c_cpu中的值赋值到l.temp_cpu中去
        copy_cpu(l.outputs*l.batch, l.c_cpu, 1, l.temp_cpu, 1);	
        //这里是算l.temp_cpu通过激活函数后的值，结果写入l.temp_cpu		
        activate_array(l.temp_cpu, l.outputs*l.batch, TANH);
        //这里实现的是将l.temp3_cpu中的值乘到l.temp_cpu中去，结果写入l.temp_cpu		
        mul_cpu(l.outputs*l.batch, l.temp3_cpu, 1, l.temp_cpu, 1);
        //然后计算l.o_cpu相对于激活函数的导数，结果和l.tem_cpu中的值相乘，结果写入l.temp_cpu		
        gradient_array(l.o_cpu, l.outputs*l.batch, LOGISTIC, l.temp_cpu);
        //将l.temp_cpu中的值赋值到wo.delta
        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, wo.delta, 1);
        //然后将s的input换成上一时刻的隐层状态
        s.input = l.prev_state_cpu;
        //s.delta指向存储上一时刻隐层状态导数的位置
        s.delta = l.dh_cpu;
        //进行wo的反向传播															
        backward_connected_layer(wo, s);	
        //将l.temp_cpu中的值赋值到uo.delta
        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, uo.delta, 1);
        //将s.input当前层次当前steps的输入x_t
        s.input = state.input;
        //同时将s.delta定位成上一层次的l.delta的位置，从这里能看出，l.delta指向的是隐层状态h的偏导数，这里是层间隐层状态delta的传递
        s.delta = state.delta;
        //进行uo层的反向传播
        backward_connected_layer(uo, s);									
        //将l.temp2_cpu中的值赋值到l.temp_cpu中
        copy_cpu(l.outputs*l.batch, l.temp2_cpu, 1, l.temp_cpu, 1);
        //将l.i_cpu中的值乘到l.temp_cpu上去，结果写入l.temp_cpu			
        mul_cpu(l.outputs*l.batch, l.i_cpu, 1, l.temp_cpu, 1);	
        //计算l.g_cpu相对于激活的偏导数，结果和存储在l.temp_cpu中的值相乘，然后把相乘的结果写入l.temp_cpu中			
        gradient_array(l.g_cpu, l.outputs*l.batch, TANH, l.temp_cpu);	
        //将l.temp_cpu中的值赋值到wg.delta
        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, wg.delta, 1);
        //将s.input换成上一个时刻的隐层状态值
        s.input = l.prev_state_cpu;
        //将相应的s.delta指向当前层次，上一时刻产生的隐藏状态的偏导数的位置，属于层内不同steps之间的delta传递
        s.delta = l.dh_cpu;	
        //进行wg层的反向传播												
        backward_connected_layer(wg, s);	
        //将l.temp_cpu中的值赋值到ug.delta中
        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, ug.delta, 1);
        //将s.input换成当前层次当前时刻相应的输入x_t
        s.input = state.input;
        //s.delta指向上一层相应的delta的位置
        s.delta = state.delta;
        //进行ug层的反向传播
        backward_connected_layer(ug, s);																
        //这里实现的是将l.temp2_cpu中的值赋值到l.temp_cpu
        copy_cpu(l.outputs*l.batch, l.temp2_cpu, 1, l.temp_cpu, 1);	
        //这里实现的是将l.g_cpu中的值乘到l.temp_cpu对应的值上去，结果写入l.temp_cpu		
        mul_cpu(l.outputs*l.batch, l.g_cpu, 1, l.temp_cpu, 1);
        //计算l.i_cpu相对于激活函数的导数，然后将结果乘到l.tem_cpu的值上，最终乘完的结果写入l.temp_cpu				
        gradient_array(l.i_cpu, l.outputs*l.batch, LOGISTIC, l.temp_cpu);
        //这里实现的是将;l.temp_cpu中的值赋值到wi.delta中	
        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, wi.delta, 1);
        //将s.input换成上一时刻的隐层态值
        s.input = l.prev_state_cpu;
        //将s.delta换成上一时刻的隐层偏导数。。层内不同step之间的delta传递
        s.delta = l.dh_cpu;
        //wi层的反向传播
        backward_connected_layer(wi, s);						
        //这里实现的是将l.temp_cpu中的值赋值到ui.delta中
        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, ui.delta, 1);
        //将s.input换成当前层相应step的输入x_t
        s.input = state.input;
        //s.delta换成上一层次存储delta的位置
        s.delta = state.delta;
        //进行ui层的反向传播
        backward_connected_layer(ui, s);									
        //这里实现的是将l.temp2_cpu中的值赋值到l.temp_cpu中
        copy_cpu(l.outputs*l.batch, l.temp2_cpu, 1, l.temp_cpu, 1);	
        //这里实现的是将l.prev_cell_cpu中的值乘到l.temp_cpu上，结果写入l.temp_cpu	
        mul_cpu(l.outputs*l.batch, l.prev_cell_cpu, 1, l.temp_cpu, 1);
        //计算l.f_cpu相对于激活函数的导数，结果和l.temp_cpu中的值相乘，并将相乘的结果写入l.temp_cpu
        gradient_array(l.f_cpu, l.outputs*l.batch, LOGISTIC, l.temp_cpu);
        //这里实现的是将l.temp_cpu中的值赋值到wf.delta中
        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, wf.delta, 1);
        //将s.input换成上一时刻的隐层状态值
        s.input = l.prev_state_cpu;
        //delta也进行对应的改变
        s.delta = l.dh_cpu;
        //进行wf层的反向传播
        backward_connected_layer(wf, s);						
        //将l.temp_cpu中的值赋值到uf.delta
        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, uf.delta, 1);
        //将s.input的输入换成当前层次当前step的输入x_t
        s.input = state.input;
        //delta也进行改变
        s.delta = state.delta;
        //进行uf层的反向传播
        backward_connected_layer(uf, s);									
        //将l.temp2_cpu中的值赋值到l.temp_cpu
        copy_cpu(l.outputs*l.batch, l.temp2_cpu, 1, l.temp_cpu, 1);	
        //将l.f_cpu中的值乘到l.temp_cpu中，结果写入l.temp_cpu		
        mul_cpu(l.outputs*l.batch, l.f_cpu, 1, l.temp_cpu, 1);	
        //将l.temp_cpu中的值赋值到l.dc_cpu中	,注意这里计算的是损失函数相对于上一个steps的遗忘状态的偏导数，完成遗忘状态偏导的层内存传递		
        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, l.dc_cpu, 1);				
        //将相应层次和位置的指针往前移动一个step位置
        state.input -= l.inputs*l.batch;
        //这里判断一下上一层的l.delta是否到了起点，或者说反向传播是否到了第一个steps
        //state.delta完成层间隐层状态的梯度传递
        if (state.delta) state.delta -= l.inputs*l.batch;
        l.output -= l.outputs*l.batch;
        l.cell_cpu -= l.outputs*l.batch;
        l.delta -= l.outputs*l.batch;

        increment_layer(&wf, -1);
        increment_layer(&wi, -1);
        increment_layer(&wg, -1);
        increment_layer(&wo, -1);

        increment_layer(&uf, -1);
        increment_layer(&ui, -1);
        increment_layer(&ug, -1);
        increment_layer(&uo, -1);
    }
}

#ifdef GPU
void update_lstm_layer_gpu(layer l, update_args a)
{
    update_connected_layer_gpu(*(l.wf), a);
    update_connected_layer_gpu(*(l.wi), a);
    update_connected_layer_gpu(*(l.wg), a);
    update_connected_layer_gpu(*(l.wo), a);
    update_connected_layer_gpu(*(l.uf), a);
    update_connected_layer_gpu(*(l.ui), a);
    update_connected_layer_gpu(*(l.ug), a);
    update_connected_layer_gpu(*(l.uo), a);
}
//gpu版本的lstm的前向传播。。结合cpu版本和前向传播公式推导
void forward_lstm_layer_gpu(layer l, network state)
{
    network s = { 0 };
    s.train = state.train;
    int i;
    layer wf = *(l.wf);
    layer wi = *(l.wi);
    layer wg = *(l.wg);
    layer wo = *(l.wo);

    layer uf = *(l.uf);
    layer ui = *(l.ui);
    layer ug = *(l.ug);
    layer uo = *(l.uo);

    fill_gpu(l.outputs * l.batch * l.steps, 0, wf.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, wi.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, wg.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, wo.delta_gpu, 1);

    fill_gpu(l.outputs * l.batch * l.steps, 0, uf.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, ui.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, ug.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, uo.delta_gpu, 1);
    if (state.train) {
        fill_gpu(l.outputs * l.batch * l.steps, 0, l.delta_gpu, 1);
    }

    for (i = 0; i < l.steps; ++i) {
        s.input_gpu = l.h_gpu;
        forward_connected_layer_gpu(wf, s);							
        forward_connected_layer_gpu(wi, s);							
        forward_connected_layer_gpu(wg, s);							
        forward_connected_layer_gpu(wo, s);							

        s.input_gpu = state.input_gpu;
        forward_connected_layer_gpu(uf, s);							
        forward_connected_layer_gpu(ui, s);							
        forward_connected_layer_gpu(ug, s);							
        forward_connected_layer_gpu(uo, s);							

        copy_gpu(l.outputs*l.batch, wf.output_gpu, 1, l.f_gpu, 1);
        axpy_gpu(l.outputs*l.batch, 1, uf.output_gpu, 1, l.f_gpu, 1);

        copy_gpu(l.outputs*l.batch, wi.output_gpu, 1, l.i_gpu, 1);	
        axpy_gpu(l.outputs*l.batch, 1, ui.output_gpu, 1, l.i_gpu, 1);	

        copy_gpu(l.outputs*l.batch, wg.output_gpu, 1, l.g_gpu, 1);	
        axpy_gpu(l.outputs*l.batch, 1, ug.output_gpu, 1, l.g_gpu, 1);	

        copy_gpu(l.outputs*l.batch, wo.output_gpu, 1, l.o_gpu, 1);	
        axpy_gpu(l.outputs*l.batch, 1, uo.output_gpu, 1, l.o_gpu, 1);	

        activate_array_gpu(l.f_gpu, l.outputs*l.batch, LOGISTIC);		
        activate_array_gpu(l.i_gpu, l.outputs*l.batch, LOGISTIC);		
        activate_array_gpu(l.g_gpu, l.outputs*l.batch, TANH);			
        activate_array_gpu(l.o_gpu, l.outputs*l.batch, LOGISTIC);		

        copy_gpu(l.outputs*l.batch, l.i_gpu, 1, l.temp_gpu, 1);		
        mul_gpu(l.outputs*l.batch, l.g_gpu, 1, l.temp_gpu, 1);		
        mul_gpu(l.outputs*l.batch, l.f_gpu, 1, l.c_gpu, 1);			
        axpy_gpu(l.outputs*l.batch, 1, l.temp_gpu, 1, l.c_gpu, 1);	

        copy_gpu(l.outputs*l.batch, l.c_gpu, 1, l.h_gpu, 1);			
        activate_array_gpu(l.h_gpu, l.outputs*l.batch, TANH);		
        mul_gpu(l.outputs*l.batch, l.o_gpu, 1, l.h_gpu, 1);	

        copy_gpu(l.outputs*l.batch, l.c_gpu, 1, l.cell_gpu, 1);		
        copy_gpu(l.outputs*l.batch, l.h_gpu, 1, l.output_gpu, 1);

        state.input_gpu += l.inputs*l.batch;
        l.output_gpu    += l.outputs*l.batch;
        l.cell_gpu      += l.outputs*l.batch;

        increment_layer(&wf, 1);
        increment_layer(&wi, 1);
        increment_layer(&wg, 1);
        increment_layer(&wo, 1);

        increment_layer(&uf, 1);
        increment_layer(&ui, 1);
        increment_layer(&ug, 1);
        increment_layer(&uo, 1);
    }
}
//gpu版本的lstm的反向传播，结合相应的反向传播公式推导来看
void backward_lstm_layer_gpu(layer l, network state)
{
    network s = { 0 };
    s.train = state.train;
    int i;
    layer wf = *(l.wf);
    layer wi = *(l.wi);
    layer wg = *(l.wg);
    layer wo = *(l.wo);

    layer uf = *(l.uf);
    layer ui = *(l.ui);
    layer ug = *(l.ug);
    layer uo = *(l.uo);

    increment_layer(&wf, l.steps - 1);
    increment_layer(&wi, l.steps - 1);
    increment_layer(&wg, l.steps - 1);
    increment_layer(&wo, l.steps - 1);

    increment_layer(&uf, l.steps - 1);
    increment_layer(&ui, l.steps - 1);
    increment_layer(&ug, l.steps - 1);
    increment_layer(&uo, l.steps - 1);

    state.input_gpu += l.inputs*l.batch*(l.steps - 1);
    if (state.delta_gpu) state.delta_gpu += l.inputs*l.batch*(l.steps - 1);

    l.output_gpu += l.outputs*l.batch*(l.steps - 1);
    l.cell_gpu += l.outputs*l.batch*(l.steps - 1);
    l.delta_gpu += l.outputs*l.batch*(l.steps - 1);

    for (i = l.steps - 1; i >= 0; --i) {
        if (i != 0) copy_gpu(l.outputs*l.batch, l.cell_gpu - l.outputs*l.batch, 1, l.prev_cell_gpu, 1);
        copy_gpu(l.outputs*l.batch, l.cell_gpu, 1, l.c_gpu, 1);
        if (i != 0) copy_gpu(l.outputs*l.batch, l.output_gpu - l.outputs*l.batch, 1, l.prev_state_gpu, 1);
        copy_gpu(l.outputs*l.batch, l.output_gpu, 1, l.h_gpu, 1);

        l.dh_gpu = (i == 0) ? 0 : l.delta_gpu - l.outputs*l.batch;

        copy_gpu(l.outputs*l.batch, wf.output_gpu, 1, l.f_gpu, 1);			
        axpy_gpu(l.outputs*l.batch, 1, uf.output_gpu, 1, l.f_gpu, 1);			

        copy_gpu(l.outputs*l.batch, wi.output_gpu, 1, l.i_gpu, 1);			
        axpy_gpu(l.outputs*l.batch, 1, ui.output_gpu, 1, l.i_gpu, 1);			

        copy_gpu(l.outputs*l.batch, wg.output_gpu, 1, l.g_gpu, 1);			
        axpy_gpu(l.outputs*l.batch, 1, ug.output_gpu, 1, l.g_gpu, 1);			

        copy_gpu(l.outputs*l.batch, wo.output_gpu, 1, l.o_gpu, 1);			
        axpy_gpu(l.outputs*l.batch, 1, uo.output_gpu, 1, l.o_gpu, 1);			

        activate_array_gpu(l.f_gpu, l.outputs*l.batch, LOGISTIC);			
        activate_array_gpu(l.i_gpu, l.outputs*l.batch, LOGISTIC);		
        activate_array_gpu(l.g_gpu, l.outputs*l.batch, TANH);			
        activate_array_gpu(l.o_gpu, l.outputs*l.batch, LOGISTIC);		

        copy_gpu(l.outputs*l.batch, l.delta_gpu, 1, l.temp3_gpu, 1);		

        copy_gpu(l.outputs*l.batch, l.c_gpu, 1, l.temp_gpu, 1);			
        activate_array_gpu(l.temp_gpu, l.outputs*l.batch, TANH);			

        copy_gpu(l.outputs*l.batch, l.temp3_gpu, 1, l.temp2_gpu, 1);		
        mul_gpu(l.outputs*l.batch, l.o_gpu, 1, l.temp2_gpu, 1);			

        gradient_array_gpu(l.temp_gpu, l.outputs*l.batch, TANH, l.temp2_gpu);
        axpy_gpu(l.outputs*l.batch, 1, l.dc_gpu, 1, l.temp2_gpu, 1);		

        copy_gpu(l.outputs*l.batch, l.c_gpu, 1, l.temp_gpu, 1);			
        activate_array_gpu(l.temp_gpu, l.outputs*l.batch, TANH);			
        mul_gpu(l.outputs*l.batch, l.temp3_gpu, 1, l.temp_gpu, 1);		
        gradient_array_gpu(l.o_gpu, l.outputs*l.batch, LOGISTIC, l.temp_gpu);
        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, wo.delta_gpu, 1);
        s.input_gpu = l.prev_state_gpu;
        s.delta_gpu = l.dh_gpu;															
        backward_connected_layer_gpu(wo, s);	

        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, uo.delta_gpu, 1);
        s.input_gpu = state.input_gpu;
        s.delta_gpu = state.delta_gpu;
        backward_connected_layer_gpu(uo, s);									

        copy_gpu(l.outputs*l.batch, l.temp2_gpu, 1, l.temp_gpu, 1);			
        mul_gpu(l.outputs*l.batch, l.i_gpu, 1, l.temp_gpu, 1);				
        gradient_array_gpu(l.g_gpu, l.outputs*l.batch, TANH, l.temp_gpu);		
        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, wg.delta_gpu, 1);
        s.input_gpu = l.prev_state_gpu;
        s.delta_gpu = l.dh_gpu;														
        backward_connected_layer_gpu(wg, s);	

        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, ug.delta_gpu, 1);
        s.input_gpu = state.input_gpu;
        s.delta_gpu = state.delta_gpu;
        backward_connected_layer_gpu(ug, s);																

        copy_gpu(l.outputs*l.batch, l.temp2_gpu, 1, l.temp_gpu, 1);			
        mul_gpu(l.outputs*l.batch, l.g_gpu, 1, l.temp_gpu, 1);				
        gradient_array_gpu(l.i_gpu, l.outputs*l.batch, LOGISTIC, l.temp_gpu);	
        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, wi.delta_gpu, 1);
        s.input_gpu = l.prev_state_gpu;
        s.delta_gpu = l.dh_gpu;
        backward_connected_layer_gpu(wi, s);						

        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, ui.delta_gpu, 1);
        s.input_gpu = state.input_gpu;
        s.delta_gpu = state.delta_gpu;
        backward_connected_layer_gpu(ui, s);									

        copy_gpu(l.outputs*l.batch, l.temp2_gpu, 1, l.temp_gpu, 1);		
        mul_gpu(l.outputs*l.batch, l.prev_cell_gpu, 1, l.temp_gpu, 1);
        gradient_array_gpu(l.f_gpu, l.outputs*l.batch, LOGISTIC, l.temp_gpu);
        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, wf.delta_gpu, 1);
        s.input_gpu = l.prev_state_gpu;
        s.delta_gpu = l.dh_gpu;
        backward_connected_layer_gpu(wf, s);						

        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, uf.delta_gpu, 1);
        s.input_gpu = state.input_gpu;
        s.delta_gpu = state.delta_gpu;
        backward_connected_layer_gpu(uf, s);									

        copy_gpu(l.outputs*l.batch, l.temp2_gpu, 1, l.temp_gpu, 1);			
        mul_gpu(l.outputs*l.batch, l.f_gpu, 1, l.temp_gpu, 1);				
        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, l.dc_gpu, 1);				

        state.input_gpu -= l.inputs*l.batch;
        if (state.delta_gpu) state.delta_gpu -= l.inputs*l.batch;
        l.output_gpu -= l.outputs*l.batch;
        l.cell_gpu -= l.outputs*l.batch;
        l.delta_gpu -= l.outputs*l.batch;

        increment_layer(&wf, -1);
        increment_layer(&wi, -1);
        increment_layer(&wg, -1);
        increment_layer(&wo, -1);

        increment_layer(&uf, -1);
        increment_layer(&ui, -1);
        increment_layer(&ug, -1);
        increment_layer(&uo, -1);
    }
}
#endif
