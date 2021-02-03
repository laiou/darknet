#include "iseg_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
//创建一个iseg层，分配内存和给相关参数赋值
//这里的iseg用于实例分割中
//相关内容也可以参考https://blog.csdn.net/artyze/article/details/82823445中的介绍
//总体上来讲实例分割不同于检测。。。这里实际上实现的作用是使得对同一个实例的预测向量尽可能的集中，不同实例的向量预测尽可能地远离
//这样就实现了分割的目的。。然后就是之所以embedding能有效。或者说之所以可以用更小的维度去表征这个类别。。是因为只要能区分开就行
//理论上假如一张图上最多90个类别，那么理论上最低能够区分这90个类的维度即可
layer make_iseg_layer(int batch, int w, int h, int classes, int ids)
{
    layer l = {0};
    l.type = ISEG;

    l.h = h;
    l.w = w;
    //输入通道数，classes表示类别数量，ids表示用几维取代表一个类别
    //这里的意思是比如说如果用onr-hot来表示类别，当类别数目很大的时候不是很方便
    //然后通过embedding来表示类别达到降为的目的
    //通常这里的class会设置成1..结合l.truth理解
    l.c = classes + ids;
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.batch = batch;
    l.extra = ids;
    l.cost = calloc(1, sizeof(float));
    l.outputs = h*w*l.c;
    l.inputs = l.outputs;
    //l.truths表示真值数量，90表示默认一张图上最多包含90个类别
    //这里的1表示类别id也就是说假如一共90个类别。。。这个1所代表的是这90个类别的id
    //在实例分割里面。可以看成是对每一个像素点进行分类，然后可以通过一个mask矩阵来实现分割
    //也就是说这个mask矩阵里面全部是0，1值。。来实现对一张图上的像素点进行分类
    //这里90个类别，每个类别有一个mask矩阵，然后一共90个mask矩阵，每个矩阵的大小就是图像的尺度
    //这里也就是输入特征图的w,h,这里的1表示类别id。。毕竟90个类别之间也需要区分一下的
    l.truths = 90*(l.w*l.h+1);
    l.delta = calloc(batch*l.outputs, sizeof(float));
    l.output = calloc(batch*l.outputs, sizeof(float));

    l.counts = calloc(90, sizeof(int));
    //从这里的l.sums和下面的if能看出ids的作用。。。
    l.sums = calloc(90, sizeof(float*));
    if(ids){
        int i;
        for(i = 0; i < 90; ++i){
            l.sums[i] = calloc(ids, sizeof(float));
        }
    }
    //iseg的前向传播。具体实现参考src/iseg_layer.c
    l.forward = forward_iseg_layer;
    //iseg的反向传播，具体实现参考src/iseg_layer.c
    l.backward = backward_iseg_layer;
#ifdef GPU
    //gpu版本的iseg_layer的前向传播，具体实现参考src/iseg_layer.c
    l.forward_gpu = forward_iseg_layer_gpu;
    //gpu版本的iseg_layer层的反向传播，具体实现参考src/iseg_layer.c
    l.backward_gpu = backward_iseg_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "iseg\n");
    srand(0);

    return l;
}

void resize_iseg_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->outputs = h*w*l->c;
    l->inputs = l->outputs;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta = realloc(l->delta, l->batch*l->outputs*sizeof(float));

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =     cuda_make_array(l->delta, l->batch*l->outputs);
    l->output_gpu =    cuda_make_array(l->output, l->batch*l->outputs);
#endif
}
//iseg_layer的前向传播，也算是loss层了
void forward_iseg_layer(const layer l, network net)
{
    //what_time_is_it_now获得当前的时间。。。。
    //具体是实现参考src/utils.c
    double time = what_time_is_it_now();
    int i,b,j,k;
    int ids = l.extra;
    //从net.input中赋值相关内容到l.output，loss层。。输入和输出一致的。。
    //memcpy完成数据的赋值
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));
    //通过memset将l.delta初始化成0
    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));

#ifndef GPU
    for (b = 0; b < l.batch; ++b){
        int index = b*l.outputs;
        activate_array(l.output + index, l.classes*l.w*l.h, LOGISTIC);
    }
#endif
    //遍历batch中每一张图片的处理结果
    for (b = 0; b < l.batch; ++b){
        // a priori, each pixel has no class
        //遍历类别id，比如说有90个类，每张图就生成了90个mask
        for(i = 0; i < l.classes; ++i){
            //接着就是特征图上每一个点的类别
            for(k = 0; k < l.w*l.h; ++k){
                //计算相应的索引，这里的索引是output中表示类别的mask的数据的索引
                //里面记录的是类别的id，但是还是通过mask呈现的，每一个classes通道上的mask表示了属于某一个类别的区域
                int index = b*l.outputs + i*l.w*l.h + k;
                //给l.delta中每一个像素点赋值
                //这里从后面能够看出计算的是mse loss。根据偏导数可以得到
                //这里的delta实际意义应该是类别的delta
                //可以看成是对l.delta的初始化
                l.delta[index] = 0 - l.output[index];
            }
        }
        //遍历表示每一个类别的enbedding向量，每一个embedding是ids维度
        // a priori, embedding should be small magnitude
        for(i = 0; i < ids; ++i){
            //同样是遍历一个特征图上的每一个点
            for(k = 0; k < l.w*l.h; ++k){
                //计算索引,从output中找到对应的类别的embedding向量的位置
                int index = b*l.outputs + (i+l.classes)*l.w*l.h + k;
                //将相应的delta进行赋值
                //这里也可以看成是对ids的delta的初始化
                l.delta[index] = .1 * (0 - l.output[index]);
            }
        }

        //将l.counts计数用0初始化
        //统计每个类别的数目
        //这里实际上是将一张图上某个类别的实例的预测值的和统计写入了l.sum，同时用这个代替真值参与loss计算。。参考后面的实现
        memset(l.counts, 0, 90*sizeof(int));
        for(i = 0; i < 90; ++i){
            //这里实现的是将将l.sums[i]初始化成0
            //初始化的长度是ids
            fill_cpu(ids, 0, l.sums[i], 1);
            //定位相应的真值的位置，提取相应的数值，也就是从真值的mask上提取对应的数值
            //这里的c表示的是类别id，也就是这个实例属于第几类
            int c = net.truth[b*l.truths + i*(l.w*l.h+1)];
            if(c < 0) break;
            // add up metric embeddings for each instance
            //遍历某一个类别的masks上的值
            for(k = 0; k < l.w*l.h; ++k){
                //根据真值中的类别索引计算对应坐标
                int index = b*l.outputs + c*l.w*l.h + k;
                //v表示真值中c类实例对应的mask上相应位置的值，不是0就是1
                float v = net.truth[b*l.truths + i*(l.w*l.h + 1) + 1 + k];
                if(v){
                    //对于v==1的值
                    //计算相应的delta，这里的delta是类别的delta。。
                    l.delta[index] = v - l.output[index];
                    //在l.output中将相应k位置上的ids个值加到l.sum[i]中
                    //因为上面对l.sum[i]初始话成0,实际上做了一个copy
                    //但是整个循环下来l.sum[i]中存储的是这张图是第i类实例在全部位置上预测值的和
                    axpy_cpu(ids, 1, l.output + b*l.outputs + l.classes*l.w*l.h + k, l.w*l.h, l.sums[i], 1);
                    //更新counts计数
                    ++l.counts[i];
                }
            }
        }
        //统计不同类实例的mse
        float *mse = calloc(90, sizeof(float));
        //遍历不通的类别
        for(i = 0; i < 90; ++i){
            //还是提取真值中相应的类别索引，即相关的类别id
            int c = net.truth[b*l.truths + i*(l.w*l.h+1)];
            if(c < 0) break;
            //遍历某个通道上的每一个数据
            for(k = 0; k < l.w*l.h; ++k){
                //v表示真值中c类实例对应的mask上相应位置的值，不是0就是1
                float v = net.truth[b*l.truths + i*(l.w*l.h + 1) + 1 + k];
                if(v){
                    //对于v=1的情况
                    int z;
                    float sum = 0;
                    //遍历类别embedding中的每一个值
                    for(z = 0; z < ids; ++z){
                        //计算相应的索引
                        int index = b*l.outputs + (l.classes + z)*l.w*l.h + k;
                        //根据mse的计算方式就能得到下面的计算过程
                        sum += pow(l.sums[i][z]/l.counts[i] - l.output[index], 2);
                    }
                    //将sum统计到mse中，得到每一类实例的结果
                    mse[i] += sum;
                }
            }
            //对每一类取均值
            mse[i] /= l.counts[i];
        }

        // Calculate average embedding
        //计算嵌入向量
        //遍历每一类实例
        for(i = 0; i < 90; ++i){
            if(!l.counts[i]) continue;
            //将l.sum[i]中的每一个值乘上1.f/l.counts[i],接着上面的操作，也就是将每个位置的预测取了均值
            scal_cpu(ids, 1.f/l.counts[i], l.sums[i], 1);
            if(b == 0 && net.gpu_index == 0){
                printf("%4d, %6.3f, ", l.counts[i], mse[i]);
                for(j = 0; j < ids; ++j){
                    printf("%6.3f,", l.sums[i][j]);
                }
                printf("\n");
            }
        }
        free(mse);

        // Calculate embedding loss
        //计算嵌入损失
        //还是遍历每一类实例
        for(i = 0; i < 90; ++i){
            if(!l.counts[i]) continue;
            //遍历特征图上的每一个位置
            for(k = 0; k < l.w*l.h; ++k){
                //v表示真值中i类实例对应的mask上相应位置的值，不是0就是1
                float v = net.truth[b*l.truths + i*(l.w*l.h + 1) + 1 + k];
                if(v){
                    //对于v=1
                    //这里的j<90是因为相对于真值而言，预测值在k这个位置的预测包括了全部类别的预测值
                    //所以还是要遍历每一个类别在k这个位置上的预测值
                    for(j = 0; j < 90; ++j){
                        if(!l.counts[j])continue;
                        int z;
                        for(z = 0; z < ids; ++z){
                            //计算相应的output中的索引
                            int index = b*l.outputs + (l.classes + z)*l.w*l.h + k;
                            //损失的计算。。就是使得某个位置上对相应类别的预测值尽量靠近一张图上全部位置对这个类别预测的均值
                            //最终是达到的效果就是，一张图上对某一类实例预测的结果都在某一均值附近，从而完成区分
                            //这里也类似一个k-means的聚类损失
                            float diff = l.sums[j][z] - l.output[index];
                            if (j == i) l.delta[index] +=   diff < 0? -.1 : .1;
                            else        l.delta[index] += -(diff < 0? -.1 : .1);
                        }
                    }
                }
            }
        }
        //遍历embedding中的每个值
        for(i = 0; i < ids; ++i){
            for(k = 0; k < l.w*l.h; ++k){
                //计算相应的索引，再一次调整delta
                int index = b*l.outputs + (i+l.classes)*l.w*l.h + k;
                l.delta[index] *= .01;
            }
        }
    }
    //统计损失
    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    printf("took %lf sec\n", what_time_is_it_now() - time);
}
//iseg层的反向传播
void backward_iseg_layer(const layer l, network net)
{   //实现的是将l.delta赋值到net.delta中
    axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
}

#ifdef GPU
//gpu版本的iseg层的前向传播
void forward_iseg_layer_gpu(const layer l, network net)
{
    copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
    int b;
    for (b = 0; b < l.batch; ++b){
        activate_array_gpu(l.output_gpu + b*l.outputs, l.classes*l.w*l.h, LOGISTIC);
        //if(l.extra) activate_array_gpu(l.output_gpu + b*l.outputs + l.classes*l.w*l.h, l.extra*l.w*l.h, LOGISTIC);
    }

    cuda_pull_array(l.output_gpu, net.input, l.batch*l.inputs);
    forward_iseg_layer(l, net);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
}
//gpu版本的iseg层的反向传播
void backward_iseg_layer_gpu(const layer l, network net)
{
    int b;
    for (b = 0; b < l.batch; ++b){
        //if(l.extra) gradient_array_gpu(l.output_gpu + b*l.outputs + l.classes*l.w*l.h, l.extra*l.w*l.h, LOGISTIC, l.delta_gpu + b*l.outputs + l.classes*l.w*l.h);
    }
    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif

