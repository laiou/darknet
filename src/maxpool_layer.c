#include "maxpool_layer.h"
#include "cuda.h"
#include <stdio.h>

image get_maxpool_image(maxpool_layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    return float_to_image(w,h,c,l.output);
}

image get_maxpool_delta(maxpool_layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    return float_to_image(w,h,c,l.delta);
}
//创建一个maxpool层
maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding)
{
    maxpool_layer l = {0};
    l.type = MAXPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.pad = padding;
    //计算输出的维度
    l.out_w = (w + padding - size)/stride + 1;
    l.out_h = (h + padding - size)/stride + 1;
    l.out_c = c;
    //输出的数据量。。指的是一张图的输出
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    //这里的size表示的是池化计算区域的尺寸。。。类似于卷积的卷积核大小
    l.size = size;
    l.stride = stride;
    int output_size = l.out_h * l.out_w * l.out_c * batch;
    l.indexes = calloc(output_size, sizeof(int));
    l.output =  calloc(output_size, sizeof(float));
    l.delta =   calloc(output_size, sizeof(float));
    //maxpool的前向传播，具体实现参考src/maxpool_layer.c
    l.forward = forward_maxpool_layer;
    //maxpool的反向传播，具体实现参考src/maxpool_layer.c
    l.backward = backward_maxpool_layer;
    #ifdef GPU
    //gpu版本的maxpool层的前向传播，具体实现参考src/maxpool_layer_kernels.cu
    l.forward_gpu = forward_maxpool_layer_gpu;
    //gpu版本的maxpool层的反向传播，具体实现参考src/maxpool_layer_kernels.cu
    l.backward_gpu = backward_maxpool_layer_gpu;
    l.indexes_gpu = cuda_make_int_array(0, output_size);
    l.output_gpu  = cuda_make_array(l.output, output_size);
    l.delta_gpu   = cuda_make_array(l.delta, output_size);
    #endif
    fprintf(stderr, "max          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}

void resize_maxpool_layer(maxpool_layer *l, int w, int h)
{
    l->h = h;
    l->w = w;
    l->inputs = h*w*l->c;

    l->out_w = (w + l->pad - l->size)/l->stride + 1;
    l->out_h = (h + l->pad - l->size)/l->stride + 1;
    l->outputs = l->out_w * l->out_h * l->c;
    int output_size = l->outputs * l->batch;

    l->indexes = realloc(l->indexes, output_size * sizeof(int));
    l->output = realloc(l->output, output_size * sizeof(float));
    l->delta = realloc(l->delta, output_size * sizeof(float));

    #ifdef GPU
    cuda_free((float *)l->indexes_gpu);
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->indexes_gpu = cuda_make_int_array(0, output_size);
    l->output_gpu  = cuda_make_array(l->output, output_size);
    l->delta_gpu   = cuda_make_array(l->delta,  output_size);
    #endif
}   
//maxpool的前向传播
void forward_maxpool_layer(const maxpool_layer l, network net)
{   
    int b,i,j,k,m,n;

    int w_offset = -l.pad/2;
    int h_offset = -l.pad/2;
    //h,w,c表示输出的h,w,c
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    //遍历batch中每一张图的处理结果
    for(b = 0; b < l.batch; ++b){
        //遍历相应输出的每一个通道
        for(k = 0; k < c; ++k){
            //遍历相应通道下的数据
            //这里对应的是输出图像上的遍历
            for(i = 0; i < h; ++i){
                for(j = 0; j < w; ++j){
                    //计算相应结果在输出的output中的索引
                    int out_index = j + w*(i + h*(k + c*b));
                    float max = -FLT_MAX;
                    int max_i = -1;
                    //遍历一次计算maxpool的区域
                    //这里对应在输入图片上的遍历，目的是为了计算两者对应区域上的最大值
                    for(n = 0; n < l.size; ++n){
                        for(m = 0; m < l.size; ++m){
                            //输出图片上每移动一个位置，代表输入图像上池化区域一个stride的移动
                            //假如不填充。暂且将h_offset和w_offset置0
                            //那么一眼就看出cur_h和cur_w表示的是一个池化区域。。结合m和n的循环来看
                            //代表一个池化区域内图像上点的位置
                            int cur_h = h_offset + i*l.stride + n;
                            int cur_w = w_offset + j*l.stride + m;
                            //这里就是将上面的位置转换成input中的坐标
                            int index = cur_w + l.w*(cur_h + l.h*(k + b*l.c));
                            //这里通过valid来判断是否越界
                            int valid = (cur_h >= 0 && cur_h < l.h &&
                                         cur_w >= 0 && cur_w < l.w);
                            //没有越界就用对应的值，越界之后将相应的值置为-FLT_MAX,也就是一个极小值，保证了填充值不会是maxpool结果里面的最大值
                            float val = (valid != 0) ? net.input[index] : -FLT_MAX;
                            //这里将当前得到的最大值的索引赋值给max_i
                            max_i = (val > max) ? index : max_i;
                            //将最大值赋值给max
                            max   = (val > max) ? val   : max;
                        }
                    }
                    //一个区域结束，将相应的值写入output
                    l.output[out_index] = max;
                    //将取到的最大值在输入中的索引写入l.indexes记录下来
                    l.indexes[out_index] = max_i;
                }
            }
        }
    }
}
//maxpool的反向传播
void backward_maxpool_layer(const maxpool_layer l, network net)
{
    int i;
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    //注意一下这个循环。。首先遍历的是maxpool的输出。。因为l.delta里面存储的是跟maxpool输出维度一样的梯度
    for(i = 0; i < h*w*c*l.batch; ++i){
        int index = l.indexes[i];
        //然后把梯度向上一层传递。。因为maxpool的输入中不是全部的值都对输出有影响。。所以只需要把部分梯度传递过去
        //也就是提取最大值那些位置的梯度传回去即可
        //这里的+=是为了将batch中多张图片的影响累加。。同时如果stride较小，也会出现重复的同一个位置
        net.delta[index] += l.delta[i];
    }
}

