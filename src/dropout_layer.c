#include "dropout_layer.h"
#include "utils.h"
#include "cuda.h"
#include <stdlib.h>
#include <stdio.h>
//创建一个dropout层，这一层的l.delta和l.output的指向参考src/parser.c的调用
dropout_layer make_dropout_layer(int batch, int inputs, float probability)
{
    dropout_layer l = {0};
    l.type = DROPOUT;
    //drop 的概率，初始化值是0.5
    l.probability = probability;
    l.inputs = inputs;
    l.outputs = inputs;
    l.batch = batch;
    //存储随机数的内存，具体参考forward_dropout_layer
    l.rand = calloc(inputs*batch, sizeof(float));
    //具体参考forward_dropout_layer中的调用
    l.scale = 1./(1.-probability);
    //dropout的前向传播。具体参考src/dropout_layer.c
    l.forward = forward_dropout_layer;
    //dropout的反向传播，具体参考src/dropout_layer.c
    l.backward = backward_dropout_layer;
    #ifdef GPU
    //dropout层的gpu版本
    //gpu版本drop层的前向传播，具体实现参考src/dropout_layer_kernels.cu
    l.forward_gpu = forward_dropout_layer_gpu;
    //gpu版本的drop层的反向传播,具体实现参考src/dropout_layer_kernels.cu
    l.backward_gpu = backward_dropout_layer_gpu;
    l.rand_gpu = cuda_make_array(l.rand, inputs*batch);
    #endif
    fprintf(stderr, "dropout       p = %.2f               %4d  ->  %4d\n", probability, inputs, inputs);
    return l;
} 

void resize_dropout_layer(dropout_layer *l, int inputs)
{
    l->rand = realloc(l->rand, l->inputs*l->batch*sizeof(float));
    #ifdef GPU
    cuda_free(l->rand_gpu);

    l->rand_gpu = cuda_make_array(l->rand, inputs*l->batch);
    #endif
}
//dropout的前向传播
void forward_dropout_layer(dropout_layer l, network net)
{
    int i;
    //drop只在训练的时候进行。。
    if (!net.train) return;
    //遍历每一个个输入值
    for(i = 0; i < l.batch * l.inputs; ++i){
        //产生0，1之间的随机值，具体实现参考src/utils.c
        float r = rand_uniform(0, 1);
        //将随机数赋值到l.rand中
        l.rand[i] = r;
        //如果随机的值小于预设的概率，剪除相应的链接
        if(r < l.probability) net.input[i] = 0;
        //如果保存这条链接的话，将相应的input进行一定调整
        else net.input[i] *= l.scale;
    }
}
//dropout的反向传播
void backward_dropout_layer(dropout_layer l, network net)
{
    int i;
    //如果没有上一层。就可以结束了
    if(!net.delta) return;
    //遍历输入的每一个值
    for(i = 0; i < l.batch * l.inputs; ++i){
        //抽取l.rand的值
        float r = l.rand[i];
        //将相应剪除的链接置0
        if(r < l.probability) net.delta[i] = 0;
        //因为上面乘了一个l.scale,导数也是一样的。。传递l.delta
        else net.delta[i] *= l.scale;
    }
}

