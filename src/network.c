#include <stdio.h>
#include <time.h>
#include <assert.h>
#include "network.h"
#include "image.h"
#include "data.h"
#include "utils.h"
#include "blas.h"

#include "crop_layer.h"
#include "connected_layer.h"
#include "gru_layer.h"
#include "rnn_layer.h"
#include "crnn_layer.h"
#include "local_layer.h"
#include "convolutional_layer.h"
#include "activation_layer.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "yolo_layer.h"
#include "normalization_layer.h"
#include "batchnorm_layer.h"
#include "maxpool_layer.h"
#include "reorg_layer.h"
#include "avgpool_layer.h"
#include "cost_layer.h"
#include "softmax_layer.h"
#include "dropout_layer.h"
#include "route_layer.h"
#include "upsample_layer.h"
#include "shortcut_layer.h"
#include "parser.h"
#include "data.h"

load_args get_base_args(network *net)
{
    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.size = net->w;

    args.min = net->min_crop;
    args.max = net->max_crop;
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.center = net->center;
    args.saturation = net->saturation;
    args.hue = net->hue;
    return args;
}
//从cfg和weights文件中加载模型数据，返回一个network类型的结构体
network *load_network(char *cfg, char *weights, int clear)
{
    //参数化cfg文件。具体实现参考src/parser.c。。前面已经注释过了。。。
    //分配需要的内存空间。。给部分参数赋值。。此时weights还没有加载进去。。
    //在gpu版本中也是在这个位置进行内存的分配和相应的gpu版本的相关参数的赋值
    network *net = parse_network_cfg(cfg);
    //到这里开始读取weights文件中的参数。。然后赋值给上面生成的network结构体
    if(weights && weights[0] != 0){
        //从weights文件加载weighrts到对应的network结构体中
        //load_weigths具体实现参照src/parser.c。。
        //在gpu版本中，在这里读取具体的weights将读取到的值推送到设备上对应的内存里
        load_weights(net, weights);
    }
    //根据clear标志选择是否将net->seen清零，暂时猜测net->seen表示当前读取的数据。。后面会更正********
    if(clear) (*net->seen) = 0;
    return net;
}
//获取当前处理的数据对应的batch_size id，也就是在第几个batch_size里面
size_t get_current_batch(network *net)
{
    size_t batch_num = (*net->seen)/(net->batch*net->subdivisions);
    return batch_num;
}

void reset_network_state(network *net, int b)
{
    int i;
    for (i = 0; i < net->n; ++i) {
        #ifdef GPU
        layer l = net->layers[i];
        if(l.state_gpu){
            fill_gpu(l.outputs, 0, l.state_gpu + l.outputs*b, 1);
        }
        if(l.h_gpu){
            fill_gpu(l.outputs, 0, l.h_gpu + l.outputs*b, 1);
        }
        #endif
    }
}

void reset_rnn(network *net)
{
    reset_network_state(net, 0);
}
//获得当前的学习率，根据相应的配置计算当前的学习率的值
float get_current_rate(network *net)
{
    size_t batch_num = get_current_batch(net);
    int i;
    float rate;
    if (batch_num < net->burn_in) return net->learning_rate * pow((float)batch_num / net->burn_in, net->power);
    switch (net->policy) {
        case CONSTANT:
            return net->learning_rate;
        case STEP:
            return net->learning_rate * pow(net->scale, batch_num/net->step);
        case STEPS:
            rate = net->learning_rate;
            for(i = 0; i < net->num_steps; ++i){
                if(net->steps[i] > batch_num) return rate;
                rate *= net->scales[i];
            }
            return rate;
        case EXP:
            return net->learning_rate * pow(net->gamma, batch_num);
        case POLY:
            return net->learning_rate * pow(1 - (float)batch_num / net->max_batches, net->power);
        case RANDOM:
            return net->learning_rate * pow(rand_uniform(0,1), net->power);
        case SIG:
            return net->learning_rate * (1./(1.+exp(net->gamma*(batch_num - net->step))));
        default:
            fprintf(stderr, "Policy is weird!\n");
            return net->learning_rate;
    }
}

char *get_layer_string(LAYER_TYPE a)
{
    switch(a){
        case CONVOLUTIONAL:
            return "convolutional";
        case ACTIVE:
            return "activation";
        case LOCAL:
            return "local";
        case DECONVOLUTIONAL:
            return "deconvolutional";
        case CONNECTED:
            return "connected";
        case RNN:
            return "rnn";
        case GRU:
            return "gru";
        case LSTM:
	    return "lstm";
        case CRNN:
            return "crnn";
        case MAXPOOL:
            return "maxpool";
        case REORG:
            return "reorg";
        case AVGPOOL:
            return "avgpool";
        case SOFTMAX:
            return "softmax";
        case DETECTION:
            return "detection";
        case REGION:
            return "region";
        case YOLO:
            return "yolo";
        case DROPOUT:
            return "dropout";
        case CROP:
            return "crop";
        case COST:
            return "cost";
        case ROUTE:
            return "route";
        case SHORTCUT:
            return "shortcut";
        case NORMALIZATION:
            return "normalization";
        case BATCHNORM:
            return "batchnorm";
        default:
            break;
    }
    return "none";
}
//接收的参数n是存储网络参数的链表的长度-1，减一是因为配置文件中第一组[net]开头。。这一组不算在具体网络结构里
network *make_network(int n)
{
    //malloc分配内存空间，并设置分配的内存为0，前面的参数表示要分配的元素的个数，后面的则是元素的size
    network *net = calloc(1, sizeof(network));
    //对相应的参数赋值。。
    //net->n存储网络层次的层数
    net->n = n;
    //layers结构体存储具体某一层的参数信息，定义参考darknet.h
    //这里分配了n个层的内存，并初始化
    //calloc分配的是n个连续的地址空间，返回的是起始地址的指针，从而完成连续的网络层次所需内存的分配
    net->layers = calloc(net->n, sizeof(layer));
    //这里的seen具体代表什么后面再看一下。。****
    //然后size_t用来表示对象的大小。。真实类型和操作系统有关系，size_t的取值range是目标平台下最大可能的数组尺寸
    net->seen = calloc(1, sizeof(size_t));
    //这里的t和cost也一样。。具体的意义后面补上******
    net->t    = calloc(1, sizeof(int));
    net->cost = calloc(1, sizeof(float));
    return net;
}
//进行网络的前向传播
void forward_network(network *netp)
{
//就是这里了。。gpu版本前向传播的入口
#ifdef GPU
    //如果前面的cudaSetDevices设置没出问题的话。。。
    if(netp->gpu_index >= 0){
        //forward_network_gpu的具体实现参考src/network.c
        forward_network_gpu(netp);   
        return;
    }
#endif
    //接收传入的network，记录当前相应层次的相关输入等信息
    network net = *netp;
    int i;
    //根据for循环实现network中每一个layer的前向传播
    for(i = 0; i < net.n; ++i){
        net.index = i;
        //就是这里传递的是当前层次的副本，所以原指针等不会变化。。但是改变的确实是其中指针所指向内存的值
        layer l = net.layers[i];
        //l.delta反向传播和计算损失的时候会用到。。。这个后面解释。。这里测试推理是不会更新模型权重的。所以把l.delta都置0了。。。
        //fill_cpu的具体实现参考src/blas.c
        if(l.delta){
            fill_cpu(l.outputs * l.batch, 0, l.delta, 1);
        }
        //进行某一层的forword计算。。
        //这里的l.forword函数具体的参考network创建的时候指定的函数。。
        //具体的forword到底是那一个函数。。要看当前的层次l是那一层。。比如说卷积。。池化。。等等。。
        //这些forword的具体算子实现在src目录下。。。不同层次实现在src目录下都能找到。。。
        l.forword(l, net);
        //更新net.input为当前层l的输出。。也就是下一层的输入，这里的更新net.input在具体forword的实现中会用到。。。
        //具体参考相关细节
        net.input = l.output;
        //这里的l.truth表示一张图片上包含的真实值的个数。。。。暂时是猜测。。后面会更新******
        if(l.truth) {
            net.truth = l.output;
        }
    }
    //计算当前network的损失。。统计每一层的损失。。相加之后返回平均损失。。
    //具体实现参考src/network.c
    calc_network_cost(netp);
}
//更新权重，在完成一个batch_size的前向和反向传播之后，对权重进行更新
void update_network(network *netp)
{
    //更新权重的gpu版本
#ifdef GPU
    if(netp->gpu_index >= 0){
        //update_network_gpu的具体实现参考src/network.c
        update_network_gpu(netp);   
        return;
    }
#endif
   
    network net = *netp;
    int i;
    update_args a = {0};
    //给相应的参数进行赋值，这里的a.batch就是一个batch_size
    a.batch = net.batch*net.subdivisions;
    //获得当前的学习率
    //get_current_rate的具体实现参考src/network.c
    a.learning_rate = get_current_rate(netp);
    //后面都是相关的一些超参数了，具体参考include/darknet.h
    a.momentum = net.momentum;
    a.decay = net.decay;
    a.adam = net.adam;
    a.B1 = net.B1;
    a.B2 = net.B2;
    a.eps = net.eps;
    ++*net.t;
    a.t = *net.t;

    for(i = 0; i < net.n; ++i){
        //注意这里的l，传入update实际上是net中当前层次的副本，虽是副本。但是还是能根据指针改变相应内存中的值
        layer l = net.layers[i];
        if(l.update){
            l.update(l, a);
        }
    }
}
//计算当前network的损失，统计存在损失的层的损失然后返回平均损失
void calc_network_cost(network *netp)
{
    network net = *netp;
    int i;
    float sum = 0;
    int count = 0;
    for(i = 0; i < net.n; ++i){
        if(net.layers[i].cost){
            sum += net.layers[i].cost[0];
            ++count;
        }
    }
    *net.cost = sum/count;
}

int get_predicted_class_network(network *net)
{
    return max_index(net->output, net->outputs);
}
//进行反向传播
void backward_network(network *netp)
{
    //这里进入GPU版本的反向传播
#ifdef GPU
    if(netp->gpu_index >= 0){
        //gpu版本的反向传播，具体实现参考src/network.c
        backward_network_gpu(netp);   
        return;
    }
#endif
    network net = *netp;
    int i;
    network orig = net;
    //循环反向遍历network中的层次
    for(i = net.n-1; i >= 0; --i){
        layer l = net.layers[i];
        if(l.stopbackward) break;
        //i==0表示到了反向传播的最后一层，这时候不在有前一层了，从而不在需要prev等操作
        if(i == 0){
            net = orig;
        }else{
            //定位当前层的上一层
            layer prev = net.layers[i-1];
            //定位当前net层次的input为前一层的输出
            net.input = prev.output;
            //等位当前net层次的delta为上一层的l.delta
            net.delta = prev.delta;
        }
        //更新索引
        net.index = i;
        //进入相应层次的反向传播
        l.backward(l, net);
    }
}
//根据输入的数据训练模型
float train_network_datum(network *net)
{
    //这里能看到net->seen记录的是当前训练已经处理的图片数量
    *net->seen += net->batch;
    //将训练的标志置为1
    net->train = 1;
    //进行前向传播。。。
    //具体实现参考src/network.c
    forward_network(net);
    //反向传播。。具体实现参考src/network.c
    backward_network(net);

    float error = *net->cost;
    //更新权重。。。
    //细节参考src/network.c
    if(((*net->seen)/net->batch)%net->subdivisions == 0) update_network(net);
    return error;
}

float train_network_sgd(network *net, data d, int n)
{
    int batch = net->batch;

    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        get_random_batch(d, batch, net->input, net->truth);
        float err = train_network_datum(net);
        sum += err;
    }
    return (float)sum/(n*batch);
}
//训练网络模型。。计算损失
float train_network(network *net, data d)
{
    //先判断一下输入数据的数量是不是net->batch的整数倍。。。
    //或者说是不是一个batch_size..用rows是因为一张图的数据写成了一个行向量。。从而这里的行数就是图片数
    assert(d.X.rows % net->batch == 0);
    //将通常说的batch_size拆分成子batch。。。
    int batch = net->batch;
    int n = d.X.rows / batch;

    int i;
    float sum = 0;

    for(i = 0; i < n; ++i){
        //取出一个子batch的数据
        //具体实现参考src/data.c
        //取出一个batch的数据信息。。送到net->input和net->truth
        get_next_batch(d, batch, i*batch, net->input, net->truth);
        //进行训练。。。。
        //具体实现参考src/network.c
        float err = train_network_datum(net);
        //统计这个batch的损失
        sum += err;
    }
    //返回整个batch_size的平均损失
    return (float)sum/(n*batch);
}

void set_temp_network(network *net, float t)
{
    int i;
    for(i = 0; i < net->n; ++i){
        net->layers[i].temperature = t;
    }
}

//设置整个网络的batch参数。。将batch设置成这里的b
void set_batch_network(network *net, int b)
{   //这个network结构中的batch
    net->batch = b;
    int i;
    //每个层次中的batch参数
    for(i = 0; i < net->n; ++i){
        net->layers[i].batch = b;
#ifdef CUDNN
        if(net->layers[i].type == CONVOLUTIONAL){
            cudnn_convolutional_setup(net->layers + i);
        }
        if(net->layers[i].type == DECONVOLUTIONAL){
            layer *l = net->layers + i;
            cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, l->out_h, l->out_w);
            cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1); 
        }
#endif
    }
}

int resize_network(network *net, int w, int h)
{
#ifdef GPU
    cuda_set_device(net->gpu_index);
    cuda_free(net->workspace);
#endif
    int i;
    //if(w == net->w && h == net->h) return 0;
    net->w = w;
    net->h = h;
    int inputs = 0;
    size_t workspace_size = 0;
    //fprintf(stderr, "Resizing to %d x %d...\n", w, h);
    //fflush(stderr);
    for (i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == CONVOLUTIONAL){
            resize_convolutional_layer(&l, w, h);
        }else if(l.type == CROP){
            resize_crop_layer(&l, w, h);
        }else if(l.type == MAXPOOL){
            resize_maxpool_layer(&l, w, h);
        }else if(l.type == REGION){
            resize_region_layer(&l, w, h);
        }else if(l.type == YOLO){
            resize_yolo_layer(&l, w, h);
        }else if(l.type == ROUTE){
            resize_route_layer(&l, net);
        }else if(l.type == SHORTCUT){
            resize_shortcut_layer(&l, w, h);
        }else if(l.type == UPSAMPLE){
            resize_upsample_layer(&l, w, h);
        }else if(l.type == REORG){
            resize_reorg_layer(&l, w, h);
        }else if(l.type == AVGPOOL){
            resize_avgpool_layer(&l, w, h);
        }else if(l.type == NORMALIZATION){
            resize_normalization_layer(&l, w, h);
        }else if(l.type == COST){
            resize_cost_layer(&l, inputs);
        }else{
            error("Cannot resize this type of layer");
        }
        if(l.workspace_size > workspace_size) workspace_size = l.workspace_size;
        if(l.workspace_size > 2000000000) assert(0);
        inputs = l.outputs;
        net->layers[i] = l;
        w = l.out_w;
        h = l.out_h;
        if(l.type == AVGPOOL) break;
    }
    layer out = get_network_output_layer(net);
    net->inputs = net->layers[0].inputs;
    net->outputs = out.outputs;
    net->truths = out.outputs;
    if(net->layers[net->n-1].truths) net->truths = net->layers[net->n-1].truths;
    net->output = out.output;
    free(net->input);
    free(net->truth);
    net->input = calloc(net->inputs*net->batch, sizeof(float));
    net->truth = calloc(net->truths*net->batch, sizeof(float));
#ifdef GPU
    if(gpu_index >= 0){
        cuda_free(net->input_gpu);
        cuda_free(net->truth_gpu);
        net->input_gpu = cuda_make_array(net->input, net->inputs*net->batch);
        net->truth_gpu = cuda_make_array(net->truth, net->truths*net->batch);
        if(workspace_size){
            net->workspace = cuda_make_array(0, (workspace_size-1)/sizeof(float)+1);
        }
    }else {
        free(net->workspace);
        net->workspace = calloc(1, workspace_size);
    }
#else
    free(net->workspace);
    net->workspace = calloc(1, workspace_size);
#endif
    //fprintf(stderr, " Done!\n");
    return 0;
}

layer get_network_detection_layer(network *net)
{
    int i;
    for(i = 0; i < net->n; ++i){
        if(net->layers[i].type == DETECTION){
            return net->layers[i];
        }
    }
    fprintf(stderr, "Detection layer not found!!\n");
    layer l = {0};
    return l;
}

image get_network_image_layer(network *net, int i)
{
    layer l = net->layers[i];
#ifdef GPU
    //cuda_pull_array(l.output_gpu, l.output, l.outputs);
#endif
    if (l.out_w && l.out_h && l.out_c){
        return float_to_image(l.out_w, l.out_h, l.out_c, l.output);
    }
    image def = {0};
    return def;
}

image get_network_image(network *net)
{
    int i;
    for(i = net->n-1; i >= 0; --i){
        image m = get_network_image_layer(net, i);
        if(m.h != 0) return m;
    }
    image def = {0};
    return def;
}

void visualize_network(network *net)
{
    image *prev = 0;
    int i;
    char buff[256];
    for(i = 0; i < net->n; ++i){
        sprintf(buff, "Layer %d", i);
        layer l = net->layers[i];
        if(l.type == CONVOLUTIONAL){
            prev = visualize_convolutional_layer(l, buff, prev);
        }
    } 
}

void top_predictions(network *net, int k, int *index)
{
    top_k(net->output, net->outputs, k, index);
}

//具体的推理操作的实现。。返回推理结果
float *network_predict(network *net, float *input)
{
    //接收输入的network..和input（也就是经过resize的图片数据）。。。
    network orig = *net;
    net->input = input;
    //因为是推理。。将无关参数置0
    net->truth = 0;
    net->train = 0;
    net->delta = 0;
    //进行前向传播
    //具体实现参考src/network.c
    forward_network(net);
    //提取前向传播的结果
    float *out = net->output;
    //更新原来的network，就是将推理前的network数据再次赋值给net。。。
    *net = orig;
    return out;
}
//对boxes计数。。根据thresh阈值过滤。。计数过滤之后的检测框，这里计数的是全部的框。比如yolov3中三个尺度的[yolo]层产生的检测框经过thresh过滤之后的和
int num_detections(network *net, float thresh)
{
    int i;
    int s = 0;
    //循环遍历net中所有的层次。。在相应的层次统计数量
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO){
            //如果存在YOLO层。。根据yolo_num_detection函数得到相关数量。。其实这里的YOLO对应cfg中的[yolo]组。。。所以这里的YOLO和DETECTION等其实是在读取cfg
            //的时候赋值的。。关于yolo_num_detections的具体实现参考src/yolo_layer.c
            s += yolo_num_detections(l, thresh);
        }
        //这里的REGION层的计数方式l.n在REGION中表示一个cell可以预测多少个框。。，这里REGION层的l.n实际上是在cfg中读取出来的
        //从而计数是通过l.w*l.h*l.n来实现的
        if(l.type == DETECTION || l.type == REGION){
            s += l.w*l.h*l.n;
        }
    }
    return s;
}
//这里实际上只是根据thresh阈值筛选boxes的数量。。然后给detedtion分配对应数量的内存。。
//并且给相关参数赋值。。但是并没有存储相关boxes的信息
detection *make_network_boxes(network *net, float thresh, int *num)
{
    //定位到network的最后一层
    layer l = net->layers[net->n - 1];
    int i;
    //获取boxes的数量。。具体实现参考src/network.c，根据置信度阈值过滤。。计数过滤之后的数量
    int nboxes = num_detections(net, thresh);
    if(num) *num = nboxes;
    //detection结构体存储了一个框相关的信息，具体声明参考include/darknet.h
    //这里根据上面的得到的框的数量分配内存
    detection *dets = calloc(nboxes, sizeof(detection));
    //循环为上面分配的detection里面的相关参数分配内存
    for(i = 0; i < nboxes; ++i){
        //根据当前层次中l.classees给detection中的prob分配内存。。。l.classes表示检测的类别数
        dets[i].prob = calloc(l.classes, sizeof(float));
        //l.coords表示当前预测的坐标数量。。。
        if(l.coords > 4){
            dets[i].mask = calloc(l.coords-4, sizeof(float));
        }
    }
    return dets;
}
//将相关符合thresh的框的信息写入detection结构体...
void fill_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, detection *dets)
{
    int j;
    //for循环遍历整个network结构。。寻找对应的层次结构
    for(j = 0; j < net->n; ++j){
        layer l = net->layers[j];
        if(l.type == YOLO){
            //将YOLO层框的信息提取到detection结构中
            //get_yolo_detections具体实现参考src/yolo_layer.c
            //这个函数结束。。dets中存储的框的坐标已经转换成了相对于原图的坐标了。。。
            int count = get_yolo_detections(l, w, h, net->w, net->h, thresh, map, relative, dets);
            //这里dets实际上是一个指针。。从而在上面操作完成后。。将dets代表的指针往后移动相应的位置。。
            //给下一次填充做定位
            dets += count;
        }
        if(l.type == REGION){
            //获得region层的框的坐标信息。。。大体逻辑和上面YOLO层的类似
            //具体实现参考src/region_layer.c
            get_region_detections(l, w, h, net->w, net->h, thresh, map, hier, relative, dets);
            dets += l.w*l.h*l.n;
        }
        if(l.type == DETECTION){
            //提取detection层的坐标，修正到原图片尺度
            //具体实现参考src/detection_layer.c
            get_detection_detections(l, w, h, thresh, dets);
            dets += l.w*l.h*l.n;
        }
    }
}
//获取network预测的检测框。。get_network_boxes(net, 1, 1, thresh, 0, 0, 0, &nboxes)
detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num)
{
    //给具体存储boxes的detection结构体分配内存。。给相关参数赋值。。具体细节参考src/network.c
    //根据置信度阈值筛选boxes。。然后统计数量。。分配内存
    detection *dets = make_network_boxes(net, thresh, num);
    //fill_network_boxes的具体实现参考src/network.c
    //把具体的预测的框的坐标转换成真实的b_x,b_y等坐标。。然后在修正到原图尺度下的坐标并写入detection结构体中..
    fill_network_boxes(net, w, h, thresh, hier, map, relative, dets);
    return dets;
}

void free_detections(detection *dets, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        free(dets[i].prob);
        if(dets[i].mask) free(dets[i].mask);
    }
    free(dets);
}

float *network_predict_image(network *net, image im)
{
    image imr = letterbox_image(im, net->w, net->h);
    set_batch_network(net, 1);
    float *p = network_predict(net, imr.data);
    free_image(imr);
    return p;
}

int network_width(network *net){return net->w;}
int network_height(network *net){return net->h;}

matrix network_predict_data_multi(network *net, data test, int n)
{
    int i,j,b,m;
    int k = net->outputs;
    matrix pred = make_matrix(test.X.rows, k);
    float *X = calloc(net->batch*test.X.rows, sizeof(float));
    for(i = 0; i < test.X.rows; i += net->batch){
        for(b = 0; b < net->batch; ++b){
            if(i+b == test.X.rows) break;
            memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
        }
        for(m = 0; m < n; ++m){
            float *out = network_predict(net, X);
            for(b = 0; b < net->batch; ++b){
                if(i+b == test.X.rows) break;
                for(j = 0; j < k; ++j){
                    pred.vals[i+b][j] += out[j+b*k]/n;
                }
            }
        }
    }
    free(X);
    return pred;   
}

matrix network_predict_data(network *net, data test)
{
    int i,j,b;
    int k = net->outputs;
    matrix pred = make_matrix(test.X.rows, k);
    float *X = calloc(net->batch*test.X.cols, sizeof(float));
    for(i = 0; i < test.X.rows; i += net->batch){
        for(b = 0; b < net->batch; ++b){
            if(i+b == test.X.rows) break;
            memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
        }
        float *out = network_predict(net, X);
        for(b = 0; b < net->batch; ++b){
            if(i+b == test.X.rows) break;
            for(j = 0; j < k; ++j){
                pred.vals[i+b][j] = out[j+b*k];
            }
        }
    }
    free(X);
    return pred;   
}

void print_network(network *net)
{
    int i,j;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        float *output = l.output;
        int n = l.outputs;
        float mean = mean_array(output, n);
        float vari = variance_array(output, n);
        fprintf(stderr, "Layer %d - Mean: %f, Variance: %f\n",i,mean, vari);
        if(n > 100) n = 100;
        for(j = 0; j < n; ++j) fprintf(stderr, "%f, ", output[j]);
        if(n == 100)fprintf(stderr,".....\n");
        fprintf(stderr, "\n");
    }
}

void compare_networks(network *n1, network *n2, data test)
{
    matrix g1 = network_predict_data(n1, test);
    matrix g2 = network_predict_data(n2, test);
    int i;
    int a,b,c,d;
    a = b = c = d = 0;
    for(i = 0; i < g1.rows; ++i){
        int truth = max_index(test.y.vals[i], test.y.cols);
        int p1 = max_index(g1.vals[i], g1.cols);
        int p2 = max_index(g2.vals[i], g2.cols);
        if(p1 == truth){
            if(p2 == truth) ++d;
            else ++c;
        }else{
            if(p2 == truth) ++b;
            else ++a;
        }
    }
    printf("%5d %5d\n%5d %5d\n", a, b, c, d);
    float num = pow((abs(b - c) - 1.), 2.);
    float den = b + c;
    printf("%f\n", num/den); 
}

float network_accuracy(network *net, data d)
{
    matrix guess = network_predict_data(net, d);
    float acc = matrix_topk_accuracy(d.y, guess,1);
    free_matrix(guess);
    return acc;
}

float *network_accuracies(network *net, data d, int n)
{
    static float acc[2];
    matrix guess = network_predict_data(net, d);
    acc[0] = matrix_topk_accuracy(d.y, guess, 1);
    acc[1] = matrix_topk_accuracy(d.y, guess, n);
    free_matrix(guess);
    return acc;
}
//获取当前network的output_layer
layer get_network_output_layer(network *net)
{
    int i;
    //从后往前遍历每一个层次，寻找计算损失函数的COST层
    //但是这里只能取到从后往前的第一个cost层。。。跟cpu版本里面统计损失有细微差别。。。
    for(i = net->n - 1; i >= 0; --i){
        if(net->layers[i].type != COST) break;
    }
    //将COST层作为网络的输出层返回
    return net->layers[i];
}

float network_accuracy_multi(network *net, data d, int n)
{
    matrix guess = network_predict_data_multi(net, d, n);
    float acc = matrix_topk_accuracy(d.y, guess,1);
    free_matrix(guess);
    return acc;
}

void free_network(network *net)
{
    int i;
    for(i = 0; i < net->n; ++i){
        free_layer(net->layers[i]);
    }
    free(net->layers);
    if(net->input) free(net->input);
    if(net->truth) free(net->truth);
#ifdef GPU
    if(net->input_gpu) cuda_free(net->input_gpu);
    if(net->truth_gpu) cuda_free(net->truth_gpu);
#endif
    free(net);
}

// Some day...
// ^ What the hell is this comment for?


layer network_output_layer(network *net)
{
    int i;
    for(i = net->n - 1; i >= 0; --i){
        if(net->layers[i].type != COST) break;
    }
    return net->layers[i];
}

int network_inputs(network *net)
{
    return net->layers[0].inputs;
}

int network_outputs(network *net)
{
    return network_output_layer(net).outputs;
}

float *network_output(network *net)
{
    return network_output_layer(net).output;
}

#ifdef GPU
//gpu版本的前向传播函数
void forward_network_gpu(network *netp)
{
    network net = *netp;
    //设置相应执行设备为net中指定的设备
    cuda_set_device(net.gpu_index);
    //这里实现的是将网络的输入数据复制到gpu上，也就是输入的图片数据
    //cuda_push_array的实现参考src/cuda.c
    cuda_push_array(net.input_gpu, net.input, net.inputs*net.batch);
    //如果是在测试的话。。还有真值也需要传入
    if(net.truth){
        cuda_push_array(net.truth_gpu, net.truth, net.truths*net.batch);
    }

    int i;
    //开始遍历network中的每一个layer。。进行前向传播。。
    for(i = 0; i < net.n; ++i){
        net.index = i;
        //还是和cpu版本一致，传入forward中的是当前层layer的一个副本
        layer l = net.layers[i];
        //如果l.delta_gpu存在
        if(l.delta_gpu){
            //fill_gpu的具体实现参考src/blas_kernels.cu
            //实现的是将l.delta_gpu用0初始化
            fill_gpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
        }
        //进行相应层次的gpu版本的前向传播
        l.forward_gpu(l, net);
        //将net.input_gpu的值替换成l.output_gpu，作为下一层layer进行gpu计算的输入
        net.input_gpu = l.output_gpu;
        //这里跟cpu版本的一致，把当前层的输出用作下一层的输入
        net.input = l.output;
        //如果有真值存在，将真值也传递下去
        if(l.truth) {
            net.truth_gpu = l.output_gpu;
            net.truth = l.output;
        }
    }
    //pull_network_output的具体实现参考src/network.c
    //将network的输出层的输出拉去一份到本地主机上，拉到了cpu版本的相应的层次的内存里，比如果COST层在cpu上也是给l.output分配了内存的
    //这个cost层的l.output_gpu的值被复制一份到了l.output中
    pull_network_output(netp);
    //计算当前network的损失，统计存在损失的层的损失然后返回平均损失
    //具体实现参考src/network.c
    calc_network_cost(netp);
}
//gpu版本的反向传播，
void backward_network_gpu(network *netp)
{
    int i;
    network net = *netp;
    network orig = net;
    //设置相应的执行设备为相应的指定设备
    //具体实现参考src/cuda.c
    cuda_set_device(net.gpu_index);
    //从后往前遍历网络中的层次
    for(i = net.n-1; i >= 0; --i){
        //和前面的以及cpu版本一致，传入的是一个当前层次的副本
        layer l = net.layers[i];
        if(l.stopbackward) break;
        //当反向传播到第一层，也就是网络的起始层的时候。。对应的就是上面存储的network的初始状态，也就是说这时候的输入就是相应的图片数据，也不需要在前传delta了
        //因为前面已经其他的层次了
        if(i == 0){
            net = orig;
        }else{
            //还是和cpu版本的类似，也是先指定相应的参数的指针
            layer prev = net.layers[i-1];
            net.input = prev.output;
            net.delta = prev.delta;
            net.input_gpu = prev.output_gpu;
            net.delta_gpu = prev.delta_gpu;
        }
        //更新网络的索引
        net.index = i;
        //然后进入相应层次的反向传播过程
        l.backward_gpu(l, net);
    }
}
//更新权重的gpu版本
void update_network_gpu(network *netp)
{
    network net = *netp;
    //设置运行代码的设备
    cuda_set_device(net.gpu_index);
    int i;
    //后面的流程和cpu版本的也基本一致。。。。
    update_args a = {0};
    a.batch = net.batch*net.subdivisions;
    a.learning_rate = get_current_rate(netp);
    a.momentum = net.momentum;
    a.decay = net.decay;
    a.adam = net.adam;
    a.B1 = net.B1;
    a.B2 = net.B2;
    a.eps = net.eps;
    ++*net.t;
    a.t = (*net.t);

    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.update_gpu){
            l.update_gpu(l, a);
        }
    }
}

void harmless_update_network_gpu(network *netp)
{
    network net = *netp;
    cuda_set_device(net.gpu_index);
    int i;
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.weight_updates_gpu) fill_gpu(l.nweights, 0, l.weight_updates_gpu, 1);
        if(l.bias_updates_gpu) fill_gpu(l.nbiases, 0, l.bias_updates_gpu, 1);
        if(l.scale_updates_gpu) fill_gpu(l.nbiases, 0, l.scale_updates_gpu, 1);
    }
}

typedef struct {
    network *net;
    data d;
    float *err;
} train_args;

void *train_thread(void *ptr)
{
    train_args args = *(train_args*)ptr;
    free(ptr);
    cuda_set_device(args.net->gpu_index);
    *args.err = train_network(args.net, args.d);
    return 0;
}

pthread_t train_network_in_thread(network *net, data d, float *err)
{
    pthread_t thread;
    train_args *ptr = (train_args *)calloc(1, sizeof(train_args));
    ptr->net = net;
    ptr->d = d;
    ptr->err = err;
    if(pthread_create(&thread, 0, train_thread, ptr)) error("Thread creation failed");
    return thread;
}

void merge_weights(layer l, layer base)
{
    if (l.type == CONVOLUTIONAL) {
        axpy_cpu(l.n, 1, l.bias_updates, 1, base.biases, 1);
        axpy_cpu(l.nweights, 1, l.weight_updates, 1, base.weights, 1);
        if (l.scales) {
            axpy_cpu(l.n, 1, l.scale_updates, 1, base.scales, 1);
        }
    } else if(l.type == CONNECTED) {
        axpy_cpu(l.outputs, 1, l.bias_updates, 1, base.biases, 1);
        axpy_cpu(l.outputs*l.inputs, 1, l.weight_updates, 1, base.weights, 1);
    }
}

void scale_weights(layer l, float s)
{
    if (l.type == CONVOLUTIONAL) {
        scal_cpu(l.n, s, l.biases, 1);
        scal_cpu(l.nweights, s, l.weights, 1);
        if (l.scales) {
            scal_cpu(l.n, s, l.scales, 1);
        }
    } else if(l.type == CONNECTED) {
        scal_cpu(l.outputs, s, l.biases, 1);
        scal_cpu(l.outputs*l.inputs, s, l.weights, 1);
    }
}


void pull_weights(layer l)
{
    if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
        cuda_pull_array(l.biases_gpu, l.bias_updates, l.n);
        cuda_pull_array(l.weights_gpu, l.weight_updates, l.nweights);
        if(l.scales) cuda_pull_array(l.scales_gpu, l.scale_updates, l.n);
    } else if(l.type == CONNECTED){
        cuda_pull_array(l.biases_gpu, l.bias_updates, l.outputs);
        cuda_pull_array(l.weights_gpu, l.weight_updates, l.outputs*l.inputs);
    }
}

void push_weights(layer l)
{
    if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
        cuda_push_array(l.biases_gpu, l.biases, l.n);
        cuda_push_array(l.weights_gpu, l.weights, l.nweights);
        if(l.scales) cuda_push_array(l.scales_gpu, l.scales, l.n);
    } else if(l.type == CONNECTED){
        cuda_push_array(l.biases_gpu, l.biases, l.outputs);
        cuda_push_array(l.weights_gpu, l.weights, l.outputs*l.inputs);
    }
}

void distribute_weights(layer l, layer base)
{
    if (l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL) {
        cuda_push_array(l.biases_gpu, base.biases, l.n);
        cuda_push_array(l.weights_gpu, base.weights, l.nweights);
        if (base.scales) cuda_push_array(l.scales_gpu, base.scales, l.n);
    } else if (l.type == CONNECTED) {
        cuda_push_array(l.biases_gpu, base.biases, l.outputs);
        cuda_push_array(l.weights_gpu, base.weights, l.outputs*l.inputs);
    }
}


/*

   void pull_updates(layer l)
   {
   if(l.type == CONVOLUTIONAL){
   cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
   cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
   if(l.scale_updates) cuda_pull_array(l.scale_updates_gpu, l.scale_updates, l.n);
   } else if(l.type == CONNECTED){
   cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
   cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.outputs*l.inputs);
   }
   }

   void push_updates(layer l)
   {
   if(l.type == CONVOLUTIONAL){
   cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
   cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
   if(l.scale_updates) cuda_push_array(l.scale_updates_gpu, l.scale_updates, l.n);
   } else if(l.type == CONNECTED){
   cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
   cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.outputs*l.inputs);
   }
   }

   void update_layer(layer l, network net)
   {
   int update_batch = net.batch*net.subdivisions;
   float rate = get_current_rate(net);
   l.t = get_current_batch(net);
   if(l.update_gpu){
   l.update_gpu(l, update_batch, rate*l.learning_rate_scale, net.momentum, net.decay);
   }
   }
   void merge_updates(layer l, layer base)
   {
   if (l.type == CONVOLUTIONAL) {
   axpy_cpu(l.n, 1, l.bias_updates, 1, base.bias_updates, 1);
   axpy_cpu(l.nweights, 1, l.weight_updates, 1, base.weight_updates, 1);
   if (l.scale_updates) {
   axpy_cpu(l.n, 1, l.scale_updates, 1, base.scale_updates, 1);
   }
   } else if(l.type == CONNECTED) {
   axpy_cpu(l.outputs, 1, l.bias_updates, 1, base.bias_updates, 1);
   axpy_cpu(l.outputs*l.inputs, 1, l.weight_updates, 1, base.weight_updates, 1);
   }
   }

   void distribute_updates(layer l, layer base)
   {
   if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
   cuda_push_array(l.bias_updates_gpu, base.bias_updates, l.n);
   cuda_push_array(l.weight_updates_gpu, base.weight_updates, l.nweights);
   if(base.scale_updates) cuda_push_array(l.scale_updates_gpu, base.scale_updates, l.n);
   } else if(l.type == CONNECTED){
   cuda_push_array(l.bias_updates_gpu, base.bias_updates, l.outputs);
   cuda_push_array(l.weight_updates_gpu, base.weight_updates, l.outputs*l.inputs);
   }
   }
 */

/*
   void sync_layer(network *nets, int n, int j)
   {
   int i;
   network net = nets[0];
   layer base = net.layers[j];
   scale_weights(base, 0);
   for (i = 0; i < n; ++i) {
   cuda_set_device(nets[i].gpu_index);
   layer l = nets[i].layers[j];
   pull_weights(l);
   merge_weights(l, base);
   }
   scale_weights(base, 1./n);
   for (i = 0; i < n; ++i) {
   cuda_set_device(nets[i].gpu_index);
   layer l = nets[i].layers[j];
   distribute_weights(l, base);
   }
   }
 */

void sync_layer(network **nets, int n, int j)
{
    int i;
    network *net = nets[0];
    layer base = net->layers[j];
    scale_weights(base, 0);
    for (i = 0; i < n; ++i) {
        cuda_set_device(nets[i]->gpu_index);
        layer l = nets[i]->layers[j];
        pull_weights(l);
        merge_weights(l, base);
    }
    scale_weights(base, 1./n);
    for (i = 0; i < n; ++i) {
        cuda_set_device(nets[i]->gpu_index);
        layer l = nets[i]->layers[j];
        distribute_weights(l, base);
    }
}

typedef struct{
    network **nets;
    int n;
    int j;
} sync_args;

void *sync_layer_thread(void *ptr)
{
    sync_args args = *(sync_args*)ptr;
    sync_layer(args.nets, args.n, args.j);
    free(ptr);
    return 0;
}

pthread_t sync_layer_in_thread(network **nets, int n, int j)
{
    pthread_t thread;
    sync_args *ptr = (sync_args *)calloc(1, sizeof(sync_args));
    ptr->nets = nets;
    ptr->n = n;
    ptr->j = j;
    if(pthread_create(&thread, 0, sync_layer_thread, ptr)) error("Thread creation failed");
    return thread;
}

void sync_nets(network **nets, int n, int interval)
{
    int j;
    int layers = nets[0]->n;
    pthread_t *threads = (pthread_t *) calloc(layers, sizeof(pthread_t));

    *(nets[0]->seen) += interval * (n-1) * nets[0]->batch * nets[0]->subdivisions;
    for (j = 0; j < n; ++j){
        *(nets[j]->seen) = *(nets[0]->seen);
    }
    for (j = 0; j < layers; ++j) {
        threads[j] = sync_layer_in_thread(nets, n, j);
    }
    for (j = 0; j < layers; ++j) {
        pthread_join(threads[j], 0);
    }
    free(threads);
}

float train_networks(network **nets, int n, data d, int interval)
{
    int i;
    int batch = nets[0]->batch;
    int subdivisions = nets[0]->subdivisions;
    assert(batch * subdivisions * n == d.X.rows);
    pthread_t *threads = (pthread_t *) calloc(n, sizeof(pthread_t));
    float *errors = (float *) calloc(n, sizeof(float));

    float sum = 0;
    for(i = 0; i < n; ++i){
        data p = get_data_part(d, i, n);
        threads[i] = train_network_in_thread(nets[i], p, errors + i);
    }
    for(i = 0; i < n; ++i){
        pthread_join(threads[i], 0);
        //printf("%f\n", errors[i]);
        sum += errors[i];
    }
    //cudaDeviceSynchronize();
    if (get_current_batch(nets[0]) % interval == 0) {
        printf("Syncing... ");
        fflush(stdout);
        sync_nets(nets, n, interval);
        printf("Done!\n");
    }
    //cudaDeviceSynchronize();
    free(threads);
    free(errors);
    return (float)sum/(n);
}
//将gpu上network的输出拉到主机上来
void pull_network_output(network *net)
{
    //get_network_output_layer的具体实现参考src/network.c
    //实现的是找到当前network的输出层
    layer l = get_network_output_layer(net);
    //cuda_pull_arry的具体实现参考src/cuda.c
    cuda_pull_array(l.output_gpu, l.output, l.outputs*l.batch);
}

#endif
