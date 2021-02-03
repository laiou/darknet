#include "crop_layer.h"
#include "cuda.h"
#include <stdio.h>

image get_crop_image(crop_layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.out_c;
    return float_to_image(w,h,c,l.output);
}

void backward_crop_layer(const crop_layer l, network net){}
void backward_crop_layer_gpu(const crop_layer l, network net){}
//创建一个crop层
crop_layer make_crop_layer(int batch, int h, int w, int c, int crop_height, int crop_width, int flip, float angle, float saturation, float exposure)
{
    fprintf(stderr, "Crop Layer: %d x %d -> %d x %d x %d image\n", h,w,crop_height,crop_width,c);
    crop_layer l = {0};
    l.type = CROP;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    //crop的尺度
    l.scale = (float)crop_height / h;
    //是否翻转等等一些预处理参数
    l.flip = flip;
    l.angle = angle;
    l.saturation = saturation;
    l.exposure = exposure;
    l.out_w = crop_width;
    l.out_h = crop_height;
    l.out_c = c;
    l.inputs = l.w * l.h * l.c;
    l.outputs = l.out_w * l.out_h * l.out_c;
    l.output = calloc(l.outputs*batch, sizeof(float));
    //crop层前向传播，具体细节参考src/crop_layer.c
    l.forward = forward_crop_layer;
    //crop层的反向传播，具体参考src/crop_layer.c
    l.backward = backward_crop_layer;

    #ifdef GPU
    //gpu版本的crop层的前向传播，具体实现参考src/crop_layer_kernels.cu
    l.forward_gpu = forward_crop_layer_gpu;
    //gpu版本的crop层的反向传播，具体实现参考src/crop_layer_kernels.cu
    l.backward_gpu = backward_crop_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, l.outputs*batch);
    l.rand_gpu   = cuda_make_array(0, l.batch*8);
    #endif
    return l;
}

void resize_crop_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->out_w =  l->scale*w;
    l->out_h =  l->scale*h;

    l->inputs = l->w * l->h * l->c;
    l->outputs = l->out_h * l->out_w * l->out_c;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    #ifdef GPU
    cuda_free(l->output_gpu);
    l->output_gpu = cuda_make_array(l->output, l->outputs*l->batch);
    #endif
}

//crop的前向传播
void forward_crop_layer(const crop_layer l, network net)
{
    int i,j,c,b,row,col;
    int index;
    int count = 0;
    //l.flip的初始化值是0
    int flip = (l.flip && rand()%2);
    //产生随机数进行dw,dh的初始化
    int dh = rand()%(l.h - l.out_h + 1);
    int dw = rand()%(l.w - l.out_w + 1);
    float scale = 2;
    float trans = -1;
    if(l.noadjust){
        scale = 1;
        trans = 0;
    }
    if(!net.train){
        //如果不是在训练的话
        flip = 0;
        dh = (l.h - l.out_h)/2;
        dw = (l.w - l.out_w)/2;
    }
    //循环遍历batch中的每一张图片
    for(b = 0; b < l.batch; ++b){
        for(c = 0; c < l.c; ++c){
            //l.out_h和l.out_w是输出的尺度
            for(i = 0; i < l.out_h; ++i){
                for(j = 0; j < l.out_w; ++j){
                    //翻转的话，这里是在w的尺度上翻转的图像
                    if(flip){
                        
                        col = l.w - dw - j - 1;    
                    }else{
                        col = j + dw;
                    }
                    row = i + dh;
                    //提取对应图片像素点在input上的索引
                    index = col+l.w*(row+l.h*(c + l.c*b)); 
                    //提取对应像素值，根据scale和trans完成调整后赋值到output
                    l.output[count++] = net.input[index]*scale + trans;
                }
            }
        }
    }
}

