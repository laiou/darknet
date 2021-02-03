#include "cost_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

COST_TYPE get_cost_type(char *s)
{
    if (strcmp(s, "seg")==0) return SEG;
    if (strcmp(s, "sse")==0) return SSE;
    if (strcmp(s, "masked")==0) return MASKED;
    if (strcmp(s, "smooth")==0) return SMOOTH;
    if (strcmp(s, "L1")==0) return L1;
    if (strcmp(s, "wgan")==0) return WGAN;
    fprintf(stderr, "Couldn't find cost type %s, going with SSE\n", s);
    return SSE;
}

char *get_cost_string(COST_TYPE a)
{
    switch(a){
        case SEG:
            return "seg";
        case SSE:
            return "sse";
        case MASKED:
            return "masked";
        case SMOOTH:
            return "smooth";
        case L1:
            return "L1";
        case WGAN:
            return "wgan";
    }
    return "sse";
}
//创建一个损失函数层
cost_layer make_cost_layer(int batch, int inputs, COST_TYPE cost_type, float scale)
{   
    //类似的分类内存，赋值操作
    fprintf(stderr, "cost                                           %4d\n",  inputs);
    cost_layer l = {0};
    l.type = COST;

    
    //这里的scale是从cfg中读取的相应层次里面的scale值
    l.scale = scale;
    //batch是cfg中的batch值
    l.batch = batch;
    l.inputs = inputs;
    l.outputs = inputs;
    l.cost_type = cost_type;
    l.delta = calloc(inputs*batch, sizeof(float));
    l.output = calloc(inputs*batch, sizeof(float));
    l.cost = calloc(1, sizeof(float));
    //损失函数的前向传播
    //具体细节参考src/cost_layer.c
    l.forward = forward_cost_layer;
    //损失函数的反向传播
    //具体是西安参考src/cost_layer.c
    l.backward = backward_cost_layer;
    #ifdef GPU
    //gpu版本的cost层的前向传播，具体实现参考src/cost_layer.c
    l.forward_gpu = forward_cost_layer_gpu;
    //gpu版本的cost层的反向传播，具体实现参考src/cost_layer.c
    l.backward_gpu = backward_cost_layer_gpu;

    l.delta_gpu = cuda_make_array(l.output, inputs*batch);
    l.output_gpu = cuda_make_array(l.delta, inputs*batch);
    #endif
    return l;
}

void resize_cost_layer(cost_layer *l, int inputs)
{
    l->inputs = inputs;
    l->outputs = inputs;
    l->delta = realloc(l->delta, inputs*l->batch*sizeof(float));
    l->output = realloc(l->output, inputs*l->batch*sizeof(float));
#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);
    l->delta_gpu = cuda_make_array(l->delta, inputs*l->batch);
    l->output_gpu = cuda_make_array(l->output, inputs*l->batch);
#endif
}
//损失函数的前向传播
void forward_cost_layer(cost_layer l, network net)
{   
    //如果不存在真值。。就不用计算损失了。。。
    if (!net.truth) return;
    //如果是MASKED类型的损失函数
    if(l.cost_type == MASKED){
        int i;
        //循环遍历输入的input，根据真值中的某些值筛选当前层的输入数据
        for(i = 0; i < l.batch*l.inputs; ++i){
            if(net.truth[i] == SECRET_NUM) net.input[i] = SECRET_NUM;
        }
    }
    //如果是smooth损失
    if(l.cost_type == SMOOTH){
        //smooth_l1_cpu的具体实现参考src/blas.c
        //具体就是实现了smooth l1 loss。。。。
        smooth_l1_cpu(l.batch*l.inputs, net.input, net.truth, l.delta, l.output);
    }else if(l.cost_type == L1){
        //L1 loss的实现具体参考src/blas.c
        l1_cpu(l.batch*l.inputs, net.input, net.truth, l.delta, l.output);
    } else {
        //L2 loss的实现，具体参考src/blas.c
        l2_cpu(l.batch*l.inputs, net.input, net.truth, l.delta, l.output);
    }
    //将l.output中全部的损失累加起来，记录到l.cost相应的位置，实际上是一个batch数据的全部预测的loss之和
    //从上面的过程就能看出。。
    l.cost[0] = sum_array(l.output, l.batch*l.inputs);
}
//损失函数的反向传播，也是多数反向传播的起点
void backward_cost_layer(const cost_layer l, network net)
{   //axpy_cpu的具体是实现参考src/blas.c
    //这里实现的操作是将这一层的l.delta传递到上一层去
    //具体操作就是将l.delta*l.scale加到net.delta上去，完成这一轮delta的累积传递
    axpy_cpu(l.batch*l.inputs, l.scale, l.delta, 1, net.delta, 1);
}

#ifdef GPU

void pull_cost_layer(cost_layer l)
{
    cuda_pull_array(l.delta_gpu, l.delta, l.batch*l.inputs);
}

void push_cost_layer(cost_layer l)
{
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.inputs);
}

//排序函数
int float_abs_compare (const void * a, const void * b)
{
    float fa = *(const float*) a;
    if(fa < 0) fa = -fa;
    float fb = *(const float*) b;
    if(fb < 0) fb = -fb;
    return (fa > fb) - (fa < fb);
}
//gpu版本的cost层前向传播
void forward_cost_layer_gpu(cost_layer l, network net)
{
    if (!net.truth) return;
    //如果需要smooth
    if(l.smooth){
        //scal_gpu的具体实现参考src/blas_kernels.cu
        //实现的是将net.truth_gpu中的每一个值都乘上1-l.smooth
        scal_gpu(l.batch*l.inputs, (1-l.smooth), net.truth_gpu, 1);
        //add_gpu的具体实现参考src/blas_kernels.cu
        //将l.smooth*1./l.inputs的值加到net.truth_gpu上
        add_gpu(l.batch*l.inputs, l.smooth * 1./l.inputs, net.truth_gpu, 1);
    }

    //如果是SMOOTH
    if(l.cost_type == SMOOTH){
        //smooth_l1_gpu的具体实现参考src/blas_kernels.cu
        //计算smooth_l1 loss,更新l.delta_gpu的值
        smooth_l1_gpu(l.batch*l.inputs, net.input_gpu, net.truth_gpu, l.delta_gpu, l.output_gpu);
    } else if (l.cost_type == L1){
        //如果是L1 loss
        //l1_gpu的具体实现参考src/blas_kernels.cu
        //计算l1 loss并更新相应的delta
        l1_gpu(l.batch*l.inputs, net.input_gpu, net.truth_gpu, l.delta_gpu, l.output_gpu);
    } else if (l.cost_type == WGAN){
        //如果是 WGAN loss
        //wgan_gpu的具体实现参考src/blas_kernels.cu
        //计算相应的损失并更新梯度
        wgan_gpu(l.batch*l.inputs, net.input_gpu, net.truth_gpu, l.delta_gpu, l.output_gpu);
    } else {
        //计算l2 损失
        //l2_gpu的具体实现参考src/blas_kernels.cu
        //计算l2损失并更新delta
        l2_gpu(l.batch*l.inputs, net.input_gpu, net.truth_gpu, l.delta_gpu, l.output_gpu);
    }

    if (l.cost_type == SEG && l.noobject_scale != 1) {
        //如果是SEGloss并且l.noobject_scale不等于1
        //scale_mask_gpu的具体实现参考src/blas_kernels.cu
        //将net.truth_gpu中是0的值对应位置上的l.delta_gpu乘上l.noobject_scale
        scale_mask_gpu(l.batch*l.inputs, l.delta_gpu, 0, net.truth_gpu, l.noobject_scale);
        //将net.truth_gpu中是0的值对应位置上的l.output_gpu乘上l.noobject_scale
        scale_mask_gpu(l.batch*l.inputs, l.output_gpu, 0, net.truth_gpu, l.noobject_scale);
    }
    if (l.cost_type == MASKED) {
        //mask_gpu的具体实现参考src/blas_kernels.cu
        //将跟SECRT_NUM上对应位置上的net.delta_gpu里面的值置0
        mask_gpu(l.batch*l.inputs, net.delta_gpu, SECRET_NUM, net.truth_gpu, 0);
    }
    //如果l.ratio不是0
    if(l.ratio){
        //将l.delta_gpu里面的值拉取到l.delta中
        //cuda_pull_array的具体实现参考src/cuda.c
        cuda_pull_array(l.delta_gpu, l.delta, l.batch*l.inputs);
        //将相应的值进行排序
        // 将l.delta中的值通过float_abs_compare函数进行排序
        //float_abc_compare函数的具体实现参考src/cost_layer.c
        //在这里的排序结果是按照delta中值的绝对值进行升序排序
        qsort(l.delta, l.batch*l.inputs, sizeof(float), float_abs_compare);
        int n = (1-l.ratio) * l.batch*l.inputs;
        float thresh = l.delta[n];
        thresh = 0;
        printf("%f\n", thresh);
        //supp_gpu的具体实现参考src/blas_kernels.cu
        //将l.delta_gpu中相应值的平方和thresh的平方比较，将l.delta_gpu中小的置0
        supp_gpu(l.batch*l.inputs, thresh, l.delta_gpu, 1);
    }

    if(l.thresh){
        //实现的功能逻辑同上
        supp_gpu(l.batch*l.inputs, l.thresh*1./l.inputs, l.delta_gpu, 1);
    }
    //将l.output_gpu的值拉取到主机上的l.output中
    //因为这里是损失层。。基本上是network的最后一层了。。将输出拉到主机
    cuda_pull_array(l.output_gpu, l.output, l.batch*l.inputs);
    //sum_array的具体实现参考src/utils.c
    //实现的是将l.output中的值累加
    l.cost[0] = sum_array(l.output, l.batch*l.inputs);
}
//gpu版本的cost层的反向传播
void backward_cost_layer_gpu(const cost_layer l, network net)
{
    //将l.delta_gpu中的值复制到net.delta_gpu中，完成delta的传递
    //axpy_gpu的具体实现参考src/blas_kernels.cu
    axpy_gpu(l.batch*l.inputs, l.scale, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif

