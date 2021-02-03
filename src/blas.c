#include "blas.h"

#include <math.h>
#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//reorg_cpu(net.input, l.w, l.h, l.c, l.batch, l.stride, 1, l.output);
//这里实际上是对输入数据进行了重新整合，维度还是不变的
//比如输入时[N,C,W,H],那么这里的输出就是[N,c/(stride*stride),w*stride,h*stride]
void reorg_cpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out)
{
    int b,i,j,k;
    //计算输出的通道数，虽然这里的out_c的通道比输入通道小，按时后面的W2和h2都变大了
    //总的输出参数还是一样的
    int out_c = c/(stride*stride);
    //遍历batch中每一张图片数据的处理结果
    for(b = 0; b < batch; ++b){
        //遍历相应图片数据处理结果的每一个通道
        for(k = 0; k < c; ++k){
            //遍历具体通道上每一个位置的数据
            for(j = 0; j < h; ++j){
                for(i = 0; i < w; ++i){
                    //计算相应的索引
                    //in_index表示的是input中的索引,或者说在这里的input中的索引
                    int in_index  = i + w*(j + h*(k + c*b));
                    //计算对应的点在整合之后的数据中的位置
                    int c2 = k % out_c;
                    int offset = k / out_c;
                    int w2 = i*stride + offset % stride;
                    int h2 = j*stride + offset / stride;
                    //out_index表示处理之后数据的索引，也就是output中的索引
                    int out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));
                    //根据forward的值决定相应的整合是前向还是还原。。
                    if(forward) out[out_index] = x[in_index];
                    else out[in_index] = x[out_index];
                }
            }
        }
    }
}
//flatten(l.output, l.w*l.h, l.c, l.batch, 1);
void flatten(float *x, int size, int layers, int batch, int forward)
{
    //声明一个交换空间并分配内存，大小是整个output中的数据尺度
    float *swap = calloc(size*layers*batch, sizeof(float));
    int i,c,b;
    //开始flatten操作
    //遍历batch中每一张图片的处理数据
    for(b = 0; b < batch; ++b){
        //遍历某一张图片处理数据中的每一个通道
        for(c = 0; c < layers; ++c){
            //遍历具体某一个通道下的每一个位置的数据
            for(i = 0; i < size; ++i){
                //计算相应的索引i1表示flatten前某一个数据在output中的索引
                int i1 = b*layers*size + c*size + i;
                //i2表示跟i1对应的数据在flatten之后的索引，当然。先放在swap里面
                int i2 = b*layers*size + i*layers + c;
                //如果forward是1
                //则swap[i2] =x[i1]
                //实际上这里实现的是一个flatten的过程。。具体解释就是说通常存储在l.output中的数据是根据batch也就是一张一张图片数据
                //的累加。。然后具体到某一张图片的数据存储是根据通道一个通道一个通道的存储，按照行优先或者说W维度优先。。这样写到一个一维数组中
                //而flatten的结果。。batch中还是按照图片顺序来的。但是具体到某一张图片的时候，就变成了按照位置来存储。也就是一个通道的数据上有w*h个位置
                //每个位置在通道维度上一共c个值。。还是行优先，但是会把这个位置上的c个值一起写入output。。
                if (forward) swap[i2] = x[i1];
                //这里如果forward是0,就是flatten的反过程了
                else swap[i1] = x[i2];
            }
        }
    }
    //将swap中的数据复制到x
    memcpy(x, swap, size*layers*batch*sizeof(float));
    //释放swap
    free(swap);
}
//weighted_sum_cpu(l.state, l.h_cpu, l.z_cpu, l.outputs*l.batch, l.output);
void weighted_sum_cpu(float *a, float *b, float *s, int n, float *c)
{
    int i;
    for(i = 0; i < n; ++i){
        c[i] = s[i]*a[i] + (1-s[i])*(b ? b[i] : 0);
    }
}

void weighted_delta_cpu(float *a, float *b, float *s, float *da, float *db, float *ds, int n, float *dc)
{
    int i;
    for(i = 0; i < n; ++i){
        if(da) da[i] += dc[i] * s[i];
        if(db) db[i] += dc[i] * (1-s[i]);
        ds[i] += dc[i] * (a[i] - b[i]);
    }
}
//shortcut_cpu(l.batch, l.w, l.h, l.c, net.layers[l.index].output, l.out_w, l.out_h, l.out_c, l.alpha, l.beta, l.output);
//shortcut_cpu(l.batch, l.out_w, l.out_h, l.out_c, l.delta, l.w, l.h, l.c, 1, l.beta, net.layers[l.index].delta);
//两个维度不一致的时候，会在跳连接的数据上提取两层全部维度中的最小值组成的一部分的数据进行叠加
void shortcut_cpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out)
{
    //计算原本数据维度和跳连接传入的数据维度的比例
    int stride = w1/w2;
    int sample = w2/w1;
    //同时判断h维度的比例是不是也正确
    assert(stride == h1/h2);
    assert(sample == h2/h1);
    //修正这个比例，从而这两个比例值一个是1，一个大于1...
    //也有可能两个都是1
    if(stride < 1) stride = 1;
    if(sample < 1) sample = 1;
    //提取跳连接数据维度和原本数据维度中各个维度上的较小值
    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;

    int i,j,k,b;
    //遍历batch中每张图片的处理数据
    //假如stride和sample都是1，后面及很容易理解了。。解释对应位置上的数据相加，结果写入l.output
    //起始其他的情况也类似，就是取了某一部分数据加上去了
    for(b = 0; b < batch; ++b){
        //遍历相应的通道
        for(k = 0; k < minc; ++k){
            for(j = 0; j < minh; ++j){
                for(i = 0; i < minw; ++i){
                    //计算相应的索引，根据sample和stride取调整相加数值的提取步长，使得数据传递的更平均一些。。。
                    int out_index = i*sample + w2*(j*sample + h2*(k + c2*b));
                    int add_index = i*stride + w1*(j*stride + h1*(k + c1*b));
                    //将对应的值加到output上去
                    out[out_index] = s1*out[out_index] + s2*add[add_index];
                }
            }
        }
    }
}
//mean_cpu(l.output, l.batch, l.out_c, l.out_h*l.out_w, l.mean)
//最终mean中存储的是整个batch中所有图片根据通道划分的均值
void mean_cpu(float *x, int batch, int filters, int spatial, float *mean)
{
    //这里的sacle算是一个取均值时候的总基数，一个batch上全部的某一个通道上的数据总量，batch*out_w*out_h
    float scale = 1./(batch * spatial);
    int i,j,k;
    //for循环遍历每一个通道，这里的通道是相对于输出数据的。
    //所以也可以看成一个卷积核一个卷积核来看
    for(i = 0; i < filters; ++i){
        //初始化当前通道的均值是0
        mean[i] = 0;
        //遍历batch中的每一个数据，因为一个batch中其实是多张图片的处理结果，这里相当于一张图一张图的来看
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                //取出对应的要计算均值的值在l.output中的索引
                int index = j*filters*spatial + i*spatial + k;
                //累计到mean中
                mean[i] += x[index];
            }
        }
        //计算均值，从上面的过程中可以看出这里的均值实际上是一个batch上所有图片中每一个通道上的均值
        // 最终mean中存储的是整个batch中所有图片根据通道划分的均值
        mean[i] *= scale;
    }
}
//variance_cpu(l.output, l.mean, l.batch, l.out_c, l.out_h*l.out_w, l.variance)
void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    //这里的sacle与上面计算均值的scale类似，是计算方差的基数
    // 为什么计算方差分母要减去1，参考：https://www.zhihu.com/question/20983193
    // 事实上，在统计学中，往往采用的方差计算公式都会让分母减1,这时因为所有数据的方差是基于均值这个固定点来计算的，
    // 对于有n个数据的样本，在均值固定的情况下，其采样自由度为n-1（只要n-1个数据固定，第n个可以由均值推出）
    float scale = 1./(batch * spatial - 1);
    int i,j,k;
    //还是for循环遍历每一个卷积核的处理结果，或者说l.output中不同图片处理结果的每一个通道
    for(i = 0; i < filters; ++i){
        //初始化方差为0
        variance[i] = 0;
        //遍历batch中的每一张图
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                //提取对应值在l.output中的索引，这里取索引的逻辑和上面计算均值的逻辑一致
                int index = j*filters*spatial + i*spatial + k;
                //这里的操作核具体的方差计算方式有关系
                //double pow(double x, double y)
                //返回x的y次幂。。。。
                variance[i] += pow((x[index] - mean[i]), 2);
            }
        }
        //计算最终的方差，具体逻辑参考上面的均值的计算，两者是对应的
        variance[i] *= scale;
    }
}
//l2normalize_cpu(l.output, l.scales, l.batch, l.out_c, l.out_w*l.out_h);
//实现l2正则化，在通道维度上的正则化，先选定某个位置，再将这个位置上全部通道的值提取出来进行正则化
void l2normalize_cpu(float *x, float *dx, int batch, int filters, int spatial)
{
    int b,f,i;
    //遍历batch中每一张图片产生的数据
    for(b = 0; b < batch; ++b){
        //遍历输出特征图上的每一个值
        for(i = 0; i < spatial; ++i){
            float sum = 0;
            //遍历每一个输出通道或者说每一个卷积核的处理
            for(f = 0; f < filters; ++f){
                //计算相应的索引，也就是output中某一张图片数据中某一个卷积核的处理结果中的每一个值
                int index = b*filters*spatial + f*spatial + i;
                //powf(x,y)即求x的y次方
                //这里是将全部卷积核在某一个位置上的处理结果的平方累加到sum中
                sum += powf(x[index], 2);
            }
            //sqrt计算平方根
            sum = sqrtf(sum);
            //遍历每一个卷积核的处理
            for(f = 0; f < filters; ++f){
                //计算相关的索引，还是提取全部卷积在某个位置上的处理值
                int index = b*filters*spatial + f*spatial + i;
                //更新对应的值
                x[index] /= sum;
                //同时更新dx的值
                dx[index] = (1 - x[index]) / sum;
            }
        }
    }
}

//normalize_cpu(l.output, l.mean, l.variance, l.batch, l.out_c, l.out_h*l.out_w); 
//normalize操作
void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    int b, f, i;
    //循环遍历batch中每一个数据的处理结果
    for(b = 0; b < batch; ++b){
        //遍历每一张图片输出的每一个通道
        for(f = 0; f < filters; ++f){
            for(i = 0; i < spatial; ++i){
                //提取对应数值在l.output中的索引
                int index = b*filters*spatial + f*spatial + i;
                //根据均值核方差调整对应的值，进行更新，具体就是相应的数值减去对应均值除以对应方差的平方根
                //后面那个0.00000if是一个极小数。。防止除数等于0。。。
                x[index] = (x[index] - mean[f])/(sqrt(variance[f]) + .000001f);
            }
        }
    }
}
//const_cpu(w*h, layer.kappa, norms, 1);
void const_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}
//mul_cpu(l.outputs*l.batch, l.r_cpu, 1, l.forgot_state, 1);
void mul_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] *= X[i*INCX];
}

//pow_cpu(w*h*c, 2, input, 1, squared, 1);
void pow_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] = pow(X[i*INCX], ALPHA);
}
//axpy_cpu(l.out_c, .01, l.mean, 1, l.rolling_mean, 1);
//axpy_cpu(l.n, learning_rate/batch, l.bias_updates, 1, l.biases, 1)
//axpy_cpu(l.nweights, -decay*batch, l.weights, 1, l.weight_updates, 1);
//axpy_cpu(l.nweights, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
//axpy_cpu(l.batch*l.inputs, l.scale, l.delta, 1, net.delta, 1);
void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] += ALPHA*X[i*INCX];
}
//scal_cpu(l.out_c, .99, l.rolling_mean, 1);
//scal_cpu(l.n, momentum, l.bias_updates, 1);
//scal_cpu(l.nweights, momentum, l.weight_updates, 1);
//scal_cpu(l.nweights, (float)l.out_w*l.out_h/(l.w*l.h), l.weights, 1);
void scal_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] *= ALPHA;
}
//fill_cpu(l.outputs*l.batch, 0, l.output, 1);
//fill_cpu(l.outputs*l.batch, 0, l.output, 1);
//fill_cpu(l.outputs * l.batch * l.steps, 0, output_layer.delta, 1);
//fill_cpu(l.hidden * l.batch * l.steps, 0, self_layer.delta, 1);
//fill_cpu(l.hidden * l.batch * l.steps, 0, input_layer.delta, 1);
//fill_cpu(l.hidden * l.batch, 0, l.state, 1);
//fill_cpu(l.outputs*l.batch, 0, l.output, 1);
void fill_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}

void deinter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT)
{
    int i, j;
    int index = 0;
    for(j = 0; j < B; ++j) {
        for(i = 0; i < NX; ++i){
            if(X) X[j*NX + i] += OUT[index];
            ++index;
        }
        for(i = 0; i < NY; ++i){
            if(Y) Y[j*NY + i] += OUT[index];
            ++index;
        }
    }
}

void inter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT)
{
    int i, j;
    int index = 0;
    for(j = 0; j < B; ++j) {
        for(i = 0; i < NX; ++i){
            OUT[index++] = X[j*NX + i];
        }
        for(i = 0; i < NY; ++i){
            OUT[index++] = Y[j*NY + i];
        }
    }
}
//copy_cpu(l.outputs*l.batch, l.output, 1, l.x, 1)
//copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
//copy_cpu(l.hidden * l.batch, input_layer.output, 1, l.state, 1);
//copy_cpu(l.outputs, l.biases, 1, l.output + i*l.outputs, 1);
//copy_cpu(l.outputs*l.batch, wf.output, 1, l.f_cpu, 1);
//copy_cpu(w*h, norms + w*h*(k-1), 1, norms + w*h*k, 1);
void copy_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] = X[i*INCX];
}

void mult_add_into_cpu(int N, float *X, float *Y, float *Z)
{
    int i;
    for(i = 0; i < N; ++i) Z[i] += X[i]*Y[i];
}
// smooth_l1_cpu(l.batch*l.inputs, net.input, net.truth, l.delta, l.output);
//smooth_l1 loss的实现
//smooth l1损失的计算：如果误差x的|X|<1则损失为0.5x^2否则损失为|x|-0.5
//这里的实现都乘了一个2....
void smooth_l1_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    //循环遍历input中的每一个值
    for(i = 0; i < n; ++i){
        //将真值中对应位置的值与预测值相减，这里比如说在yolo中实际上就是预测的4个框的坐标信息
        float diff = truth[i] - pred[i];
        //fabs返回diff的绝对值
        float abs_val = fabs(diff);
        //l.delta初始化，同时统计误差损失
        //预测值跟真值差距在（-1，1）以内
        if(abs_val < 1) {
            //更新对应的error和l.delta
            //根据smooth l1的导数更新l.delta
            error[i] = diff * diff;
            delta[i] = diff;
        }
        else {
            //误差超了。。。
            //另一套更新方式
            error[i] = 2*abs_val - 1;
            //这里跟smooth l1的导数有关系
            delta[i] = (diff < 0) ? 1 : -1;
        }
    }
}
//L1 loss的实现
void l1_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    //遍历输出的每一个值
    //处理的实际上是一个batch的数据
    for(i = 0; i < n; ++i){
        //计算真值与预测值的差值
        float diff = truth[i] - pred[i];
        //统计误差
        error[i] = fabs(diff);
        //根据L1的导数初始化更新l.delta
        delta[i] = diff > 0 ? 1 : -1;
    }
}
//softmax_x_ent_cpu(l.batch*l.inputs, l.output, net.truth, l.delta, l.loss);
void softmax_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    //遍历batch中每一个输入值
    for(i = 0; i < n; ++i){
        //提取对应的真值和预测值
        float t = truth[i];
        float p = pred[i];
        //计算损失和相应的梯度，统计真值类别对应的那个预测值的-log(p)loss
        //因为softmax值实际上是一个向量。。代表各种类别的概率，而真值是一个one-hot向量
        error[i] = (t) ? -log(p) : 0;
        //根据上面softmax的以及上面计算loss的方式，推导相应的梯度计算方式。。。
        //推导过程参考https://zhuanlan.zhihu.com/p/42040307
        delta[i] = t-p;
    }
}
//logistic_x_ent_cpu(l.batch*l.inputs, l.output, net.truth, l.delta, l.loss);
void logistic_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    //遍历输入数据
    for(i = 0; i < n; ++i){
        //提取对应的真值
        float t = truth[i];
        //提取对应的预测值
        float p = pred[i];
        //计算误差
        error[i] = -t*log(p) - (1-t)*log(1-p);
        //根据误差函数计算相应的偏导数更新l.delta
        delta[i] = t-p;
    }
}
//L2 loss的实现
void l2_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    //一样的遍历每一个输出
    for(i = 0; i < n; ++i){
        //计算真值和预测值的差异
        float diff = truth[i] - pred[i];
        //根据L2 loss的计算方式和导数统计误差和更新l.delta
        error[i] = diff * diff;
        delta[i] = diff;
    }
}

float dot_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    float dot = 0;
    for(i = 0; i < N; ++i) dot += X[i*INCX] * Y[i*INCY];
    return dot;
}
//softmax函数，跟最原始的softmax的公式有些许出入，但是大体上的实现过程是类似的，解决了计算softmax的上下溢出的问题和后续计算log(softmax)会出现log(0)的问题
//softmax(input + b*batch_offset + g*group_offset, n, temp, stride, output + b*batch_offset + g*group_offset);
void softmax(float *input, int n, float temp, int stride, float *output)
{
    int i;
    float sum = 0;
    float largest = -FLT_MAX;
    //遍历输入的每一个数据
    for(i = 0; i < n; ++i){
        //提取输入数据中的最大值
        if(input[i*stride] > largest) largest = input[i*stride];
    }
    //然后遍历每一个输入，实际上这里的stride是1，temp可以看成是相应的加权值
    for(i = 0; i < n; ++i){
        //计算softmax的时候将每一个值减去他们中的最大值，计算出的softmax和理论上的一致，但是能够解决上溢出和下溢出的问题
        //比如说e的x次方。。有可能会存在上下溢出的情况
        //然后temp是为了防止后面计算log(softmax)出现log(0)的情况，具体参考https://www.cnblogs.com/deepllz/p/9046157.html
        float e = exp(input[i*stride]/temp - largest/temp);
        sum += e;
        output[i*stride] = e;
    }
    for(i = 0; i < n; ++i){
        output[i*stride] /= sum;
    }
}

//softmax_cpu(net.input, l.inputs/l.groups, l.batch, l.inputs, l.groups, l.inputs/l.groups, 1, l.temperature, l.output);
void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
    int g, b;
    //遍历batch中每一张图片的处理数据
    for(b = 0; b < batch; ++b){
        //如果需要分组的话，就遍历每一个分组
        for(g = 0; g < groups; ++g){
            //softmax函数的具体实现参考src/blas.c
            softmax(input + b*batch_offset + g*group_offset, n, temp, stride, output + b*batch_offset + g*group_offset);
        }
    }
}
//upsample_cpu(net.input, l.w, l.h, l.c, l.batch, l.stride, 1, l.scale, l.output);
void upsample_cpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out)
{
    int i, j, k, b;
    //遍历batch中每张图片的处理数据
    for(b = 0; b < batch; ++b){
        //遍历相应的通道
        for(k = 0; k < c; ++k){
            //遍历上采样输出中某个通道上的每个数据
            for(j = 0; j < h*stride; ++j){
                for(i = 0; i < w*stride; ++i){
                    //计算相应的索引，in_index是输入的input中的索引
                    int in_index = b*w*h*c + k*w*h + (j/stride)*w + i/stride;
                    //out_index是输出的output中的索引
                    int out_index = b*w*h*c*stride*stride + k*w*h*stride*stride + j*w*stride + i;
                    //进行相应的赋值
                    //这里的scale是cfg中的值
                    if(forward) out[out_index] = scale*in[in_index];
                    //这里是+=是应为这里会在反向传播中使用，进行梯度累加
                    else in[in_index] += scale*out[out_index];
                }
            }
        }
    }
}


