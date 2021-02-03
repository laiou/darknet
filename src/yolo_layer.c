#include "yolo_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
//创建一个yolo层
layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes)
{
    int i;
    layer l = {0};
    l.type = YOLO;
    //这里的n表示一个cell预测多少个框
    l.n = n;
    //l.total表示一共预设了多少组anchor框的宽高。。
    //然后通过mask来选取使用那一组预设的anchor框
    l.total = total;
    l.batch = batch;
    l.h = h;
    l.w = w;
    //输入通道数
    l.c = n*(classes + 4 + 1);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.cost = calloc(1, sizeof(float));
    //这里的bias里面存储的是预设的那些anchor的值
    l.biases = calloc(total*2, sizeof(float));
    //如果存在mask的话。。。这里的mask的值是从cfg中读取的
    //表示当前层次预测什么规模的框，比如0，1，2表示预测小物体
    if(mask) l.mask = mask;
    else{
        l.mask = calloc(n, sizeof(int));
        //从这里看得出。。l.mask的长度跟l.n是对应的
        for(i = 0; i < n; ++i){
            l.mask[i] = i;
        }
    }
    l.bias_updates = calloc(n*2, sizeof(float));
    l.outputs = h*w*n*(classes + 4 + 1);
    l.inputs = l.outputs;
    l.truths = 90*(4 + 1);
    l.delta = calloc(batch*l.outputs, sizeof(float));
    l.output = calloc(batch*l.outputs, sizeof(float));
    for(i = 0; i < total*2; ++i){
        l.biases[i] = .5;
    }
    //yolo层的前向传播，具体实现参考src/yolo_layer.c
    l.forward = forward_yolo_layer;
    //yolo层的反向传播，具体实现参考src/yolo_layer.c
    l.backward = backward_yolo_layer;
#ifdef GPU
    //yolo层的前向传播的gpu实现，具体细节参考src/yolo_layer.c
    l.forward_gpu = forward_yolo_layer_gpu;
    //yolo层的反向传播的gpu的实现，具体细节参考src/yolo_layer.c
    l.backward_gpu = backward_yolo_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "yolo\n");
    srand(0);

    return l;
}

void resize_yolo_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->outputs = h*w*l->n*(l->classes + 4 + 1);
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
//根据某一个检测框坐标的索引等信息。。提取最终的框坐标，返回一个存储框具体信息的box结构体,box结构体里面就是一个检测框的4个坐标值
//get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h)
//顺便再写一下。。。l.mask[]的长度其实就是代表当前层预测多少个框。。跟yolo层的l.n是对应的。。然后就是预定义了l.total组anchor的宽高。。。通过mask来选择用那一个比例
//这样子来实现的t_w,t_h，到b_w,b_h的转换。。这里的操作就是把预测值根据预先设定的anchor等转换成相对于网络输入图像的坐标值。。。
box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{   
    //解析一个框具体坐标的过程
    //对于一个yolo层,l.biases在创建的时候提取了cfg中anchor的值。。具体参考src/parser.c然后就是
    //在yolo层中l.biases存储的是预定义的anchor的宽和高。。。
    //这里要注意一下预测的l.output中存储的坐标比如t_x,t_y,t_w,t_h等。。这里的t_x,t_y表示的是框中心点相对于cell中心的偏移。。并不是通常的左上角坐标
    //然后就是这里实现的是将预测的t_x,t_y等转换成真实的b_x,b_y等坐标点
    box b;
    //对于b.x其实就是1/l.w+t_x/l.w。。表示了cell中心点加上了预测的偏移量的转换
    b.x = (i + x[index + 0*stride]) / lw;
    //b.y这里跟b.x一样
    b.y = (j + x[index + 1*stride]) / lh;
    //b.w这里的话就是根据预定义的anchor的宽或者高跟整个网络输入尺度的比值乘上e的t_w或者t_h次方进行转换得到....
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}
//delta_yolo_box(truth, l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, net.w, net.h, l.delta, (2-truth.w*truth.h), l.w*l.h);
//delta_yolo_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, net.w, net.h, l.delta, (2-truth.w*truth.h), l.w*l.h);
//计算相应预测框坐标的delta。。。
float delta_yolo_box(box truth, float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, float *delta, float scale, int stride)
{
    //获得预测框的坐标值，具体实现细节参考src/yolo_layer.c
    box pred = get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride);
    //计算预测框和相应真值框的iou
    float iou = box_iou(pred, truth);
    //这里对真值框坐标的转换结合get_yolo_box里面的操作一起看。。实际上就是说
    //不管是计算iou还是delta,都要将相应左边转换到同一个标准下。。这个过程和上面get_yolo_box中的转换刚好是反过来的
    float tx = (truth.x*lw - i);
    float ty = (truth.y*lh - j);
    float tw = log(truth.w*w / biases[2*n]);
    float th = log(truth.h*h / biases[2*n + 1]);
    //计算相应坐标值的delta，根据损失函数的导数计算。。。
    delta[index + 0*stride] = scale * (tx - x[index + 0*stride]);
    delta[index + 1*stride] = scale * (ty - x[index + 1*stride]);
    delta[index + 2*stride] = scale * (tw - x[index + 2*stride]);
    delta[index + 3*stride] = scale * (th - x[index + 3*stride]);
    return iou;
}

//delta_yolo_class(l.output, l.delta, class_index, class, l.classes, l.w*l.h, 0);
//class表示真值的class类别信息
void delta_yolo_class(float *output, float *delta, int index, int class, int classes, int stride, float *avg_cat)
{
    int n;
    if (delta[index]){
        delta[index + stride*class] = 1 - output[index + stride*class];
        if(avg_cat) *avg_cat += output[index + stride*class];
        return;
    }
    //遍历每一个预测的类别
    for(n = 0; n < classes; ++n){
        //计算相应每种类别预测的delta
        delta[index + stride*n] = ((n == class)?1 : 0) - output[index + stride*n];
        //然后将预测类别为真值类别的预测值累加到avg_cat中
        if(n == class && avg_cat) *avg_cat += output[index + stride*n];
    }
}
//获得某一个检测框的类别索引entry_index(l, 0, m*l.w*l.h + i, 4);
//看一下返回值 batch*l.outputs + n*l.w*l.h*(4+l.classes+1) + entry*l.w*l.h + loc
//以entry_index(l, 0, n*l.w*l.h + i, 4)为例。。。
//首先batch*l.outputs...batch表示当前处理的图片在整个batch里的位置。。。。l.outputs表示当前层输出的feature的参数量，假如输出维度是w,h,c。。。则l.outputs等于w*h*c
//所以前面的batch*l.outputs表示之前图片处理结果的偏移。。。
//n*l.w*l.h(4+l.classes+1)因于n =   location / (l.w*l.h)，所以实际上等价于（m*l.w*l.h + i）*(4+l.classes+1)
//4表示一个框的四个坐标，1表示存储的置信度，l.classes表示当前网络一共能够预测多少个类别，对于某一个框，预测产生的参数是（x,y,w,h,c,C1,C2,.....）
//m表示当前图片中某一个cell的第m个框，i表示当前图片的第i个cell，然后就是这里其实需要注意的是int n = location/(l.w*l.h)...
//因为是int...所以实际上n=m。。。。取整数了。。。。。
//然后就是说一下l.output的存储逻辑吧。。。简单说就是按照w,h,c的维度存储到一个一维数组里面。。形象一点的表达是：
//假如这时候w=h=2,l.classes=2,框数量=2。。。。那么l.output里面的情况就是：
//x_0.0,x_0.1,x_1.0,x_1.1,y_0.0,y_0.1,y_1.0,y_1.1,w_0.0,w_0.1,w_1.0,w_1.1,h_0.0,h_0.1,h_1.0,h_1.1,c_0.0,c_0.1,c_1.0,c_1.1,C1_0.0,C1_0.1,C1_1.0,C1_1.1,C2_0.0,C2_0.1,C2_1.0,C2_1.1这些是所有
//cell中第一个框的信息。然后接着同样的格式存储第二个框。。
//从而n*l.w*l.h*(4+l.classes+1)是找到第n个框的位置。。后面的entry*l.w*l.h是为了定位到具体某一个框的x,y,w,h等信息的开头。。然后通过loc在找到具体的位置。。。。
static int entry_index(layer l, int batch, int location, int entry)
{
    int n =   location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    return batch*l.outputs + n*l.w*l.h*(4+l.classes+1) + entry*l.w*l.h + loc;
}
//yolo层的前向传播，注意一下怎么划分正负样本的逻辑就行了
void forward_yolo_layer(const layer l, network net)
{
    int i,j,b,t,n;
    //将net.input中的数据赋值到l.output中。。应为实际上也是计算loss。。。输入和输出一致
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));

#ifndef GPU
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array(l.output + index, 2*l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array(l.output + index, (1+l.classes)*l.w*l.h, LOGISTIC);
        }
    }
#endif
    //将l.delta置0
    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    if(!net.train) return;
    float avg_iou = 0;
    float recall = 0;
    float recall75 = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    int class_count = 0;
    *(l.cost) = 0;
    //遍历batch中每一张图的处理数据
    for (b = 0; b < l.batch; ++b) {
        //遍历某一个通道上全部的位置，或者说遍历每一个cell
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                //遍历某一个cell上预测的全部的预测框
                //这里是计算除了真值点之外，其他点预测的损失或者说delta..只不过这里直接将全部的点都处理了一遍。。反正后面在计算真值点的损失的时候可以重写
                //具体来讲就是，一张图的预测框是很多的，但是往往真值点也就是实际上存在真值的就几个。。这时候正负样本不均匀。。所以要其他非真值点进行处理，
                //将某些预测的框和真值框比较接近的框也算成正样本。。。大概就是这么一个流程
                for (n = 0; n < l.n; ++n) {
                    //计算相应的索引，具体实现参考src/yolo_layer.c，这里的box_index是output中的索引
                    int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                    //根据相应的索引提取对应的box的预测框
                    box pred = get_yolo_box(l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, net.w, net.h, l.w*l.h);
                    float best_iou = 0;
                    int best_t = 0;
                    //提取一张图片上的真值框
                    for(t = 0; t < l.max_boxes; ++t){
                        //提取相应的真值框，float_to_box的具体实现参考src/box.c
                        box truth = float_to_box(net.truth + t*(4 + 1) + b*l.truths, 1);
                        if(!truth.x) break;
                        //计算当前预测框和提取出的真值框的iou
                        float iou = box_iou(pred, truth);
                        //记录当前预测框和这张图片上的真值框的最大的iou,然后就是记录产生这个最大iou的真值框的id
                        if (iou > best_iou) {
                            best_iou = iou;
                            best_t = t;
                        }
                    }
                    //计算当前预测框产生的prob的预测值的索引
                    int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4);
                    //将相应的prob的预测值加到avg_anyobj里面
                    avg_anyobj += l.output[obj_index];
                    //同时计算相应的prob的梯度，这里是先假设这个cell下的框的prob的真值是0，也就是不包含目标，最好结合后面的部分一起理解这个逻辑吧
                    //这里是先假设全部都是负样本
                    l.delta[obj_index] = 0 - l.output[obj_index];
                    //如果best_iou超过阈值，可以留下，这里的ignore_thresh
                    //表示如果best_iou大于这个值，就直接认为这个框里面存在目标
                    //将上面计算的不存在目标的梯度清掉，best_iou > l.ignore_thresh表示这个框不适合做负样本
                    //但是能不能直接做正样本还要再看。。所以先把原来的假设是负样本的梯度清除
                    if (best_iou > l.ignore_thresh) {
                        //将上面已经计算过的不包含目标的梯度清除
                        l.delta[obj_index] = 0;
                    }
                    //然后考虑best_iou是否超过thruth_thresh阈值，如果超过了，则认为这个框内包含目标，真值的prob是1
                    //best_iou > l.truth_thresh表示将这个预测框作为正样本
                    if (best_iou > l.truth_thresh) {
                        //然后重写梯度，毕竟跟上面一开始假设的是不包含目标
                        //这样几个操作下来，就会全部的梯度delta大体上就是三种了：一种相差甚远的delta直接作为负样本，然后跟真值框重合度很高的用label为1计算梯度
                        //然后剩下的就是重合度在中间层次的，梯度先全部清0
                        l.delta[obj_index] = 1 - l.output[obj_index];
                        //提取相应iou最大的真值框的类别label
                        int class = net.truth[best_t*(4 + 1) + b*l.truths + 4];
                        if (l.map) class = l.map[class];
                        //计算对应预测框的类别预测的索引
                        int class_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4 + 1);
                        //计算相应的类别预测的delta，统计预测类别为真值类别的预测值到avg_cat
                        //delta_yolo_class的具体实现参考src/yolo_layer.c
                        delta_yolo_class(l.output, l.delta, class_index, class, l.classes, l.w*l.h, 0);
                        //提取相应的真值框的坐标
                        box truth = float_to_box(net.truth + best_t*(4 + 1) + b*l.truths, 1);
                        //计算相应的坐标的delta，具体实现参考src/yolo_layer.c
                        delta_yolo_box(truth, l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, net.w, net.h, l.delta, (2-truth.w*truth.h), l.w*l.h);
                    }
                }
            }
        }
        //到这里就相当于是对batch中某一张图的产生的而预测数据进行了上述处理
        //接着还是先遍历全部的真值框
        //这里实际上就是计算真值点上预测的相应的损失。。。
        for(t = 0; t < l.max_boxes; ++t){
            //提取真值框的坐标
            box truth = float_to_box(net.truth + t*(4 + 1) + b*l.truths, 1);

            if(!truth.x) break;
            float best_iou = 0;
            int best_n = 0;
            //这里的l.w实际上就是cell的分割数，比如分割乘7x7的cell，那么这里的l.w就是7
            //然后就是上面获得的truth里面实际上是相对于对应cell左上角的偏移量，然后这里是利用i,j来讲这个真值框
            //分配到对应的cell里面
            i = (truth.x * l.w);
            j = (truth.y * l.h);
            box truth_shift = truth;

            truth_shift.x = truth_shift.y = 0;
            //遍历预设的每一组的anchor宽高比例
            //这里是实现的就是将预设的anchor作为预测值，计算跟对应真值的iou
            //就是将真值和预设的anchor对应起来，根据iou确定配对关系
            for(n = 0; n < l.total; ++n){
                box pred = {0};
                //将anchor的宽高按照网络的输入归一化
                pred.w = l.biases[2*n]/net.w;
                pred.h = l.biases[2*n+1]/net.h;
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou){
                    best_iou = iou;
                    best_n = n;
                }
            }
            //到这里的时候best_n就记录了当前真值框配对下的anchor的id
            //提取相应的mask的值，int_index的具体实现参考src/utils.c
            //也就是根据上面得到的best_n的id去得到对应的mask的id
            //或者说判断上面的best_n是否是mask中指定的anchor
            int mask_n = int_index(l.mask, best_n, l.n);
            if(mask_n >= 0){
                //计算相应的索引，这个索引是在output中的索引，表示跟mask_n对应的那个预测框的坐标
                //注意一下，一个cell预测多少个框和mask中的值的数量是对应起来的
                int box_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 0);
                //计算相应的坐标的delta。。。
                float iou = delta_yolo_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, net.w, net.h, l.delta, (2-truth.w*truth.h), l.w*l.h);
                //计算相应预测框的prob的坐标
                int obj_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4);
                //统计prob的预测值
                avg_obj += l.output[obj_index];
                //由上述过程能看到，这个框的真值的prob是1，因为有真值框划分过来了
                l.delta[obj_index] = 1 - l.output[obj_index];
                //提取真值框的class类别id
                int class = net.truth[t*(4 + 1) + b*l.truths + 4];
                if (l.map) class = l.map[class];
                //计算预测框类别预测值的索引
                int class_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4 + 1);
                //计算相应的类别预测的delta
                delta_yolo_class(l.output, l.delta, class_index, class, l.classes, l.w*l.h, &avg_cat);

                //将目标个数加一，也就是真值的目标数量
                ++count;
                //统计目标class的个数
                ++class_count;
                //如果真值框和预测框的iou超过相应的值，则进行相应的计数，表示真值点检测出了目标，用于计算召回率
                if(iou > .5) recall += 1;
                if(iou > .75) recall75 += 1;
                //统计真值点上预测的iou值
                avg_iou += iou;
            }
        }
    }
    //计算损失。这里结合损失函数就能看出来了
    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    printf("Region %d Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f,  count: %d\n", net.index, avg_iou/count, avg_cat/class_count, avg_obj/count, avg_anyobj/(l.w*l.h*l.n*l.batch), recall/count, recall75/count, count);
}
//yolo层的反向传播
void backward_yolo_layer(const layer l, network net)
{
    //将l.delta赋值到net.delta完成delta的传递
   axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
}
//对取到的boxes的坐标进行修正。。修正的是b_x,b_y,b_h,b_w
void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w=0;
    int new_h=0;
    //这里的逻辑就是看两个修正参数，w,和h
    //根据网络输入尺度跟修正参数的比值来对box的坐标进行修正
    //具体修正细节参考下面的实现。。实际上就是把得到的相对于网络输入尺寸图片的左边再次修正到真正的原图上的坐标
    if (((float)netw/w) < ((float)neth/h)) {
        new_w = netw;
        new_h = (h * netw)/w;
    } else {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (i = 0; i < n; ++i){
        box b = dets[i].bbox;
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw); 
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth); 
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}
//统计cfg中配置的[yolo]层产生的检测框的数量，根据thresh过滤，这里的thresh表示每一个框的置信度阈值，test_yolo模式会输入这个阈值。。。
//根据这个阈值筛选去掉置信度低于这个阈值的框。。统计高于这个阈值框的数量
int yolo_num_detections(layer l, float thresh)
{
    int i, n;
    int count = 0;
    //这里YOLO层的l.n实际上也是从cfg中读取并赋值的。。表示一个cell预测多少个框
    //这里的两层循环的逻辑是先遍历每一个cell...在遍历每一个cell下面的每一个boxes
    for (i = 0; i < l.w*l.h; ++i){
        for(n = 0; n < l.n; ++n){
            //获得类别索引
            //entry_index的具体实现参考src/yolo_layer.c
            //这里首先l.output存储了当前层的输出值。。。在这里对entry_index的调用就是为了得到l.output中的某个框的置信度的索引
            //关于l.output的存储方式等等参考entry_index的注释
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            if(l.output[obj_index] > thresh){
                ++count;
            }
        }
    }
    return count;
}

void avg_flipped_yolo(layer l)
{
    int i,j,n,z;
    //对于yolo层l.n表示一个cell检测的框的个数
    //这个时候调用avg_flipped_yolo的时候batch等于2。。。也就是l.output里面存储了两张图的预测数据
    //但是l.outputs表示的是l层输出的数据量，对于yolo层也就是l.outputs=l.n*l.w*l.h(l.classes+4+1)
    //从而这里实际上是将filp指向了l.output中两张图片预测数据的分隔处
    float *flip = l.output + l.outputs;
    //这里实现的是把第二张图片的x,y信息前面加一个负号。。。w,h,以及置信度类别预测等信息根据l.w维度进行翻转
    for (j = 0; j < l.h; ++j){
        for (i = 0; i < l.w/2; ++i){
            for (n = 0; n < l.n; ++n){
                for(z = 0; z < l.classes + 4 + 1; ++z){
                    int i1 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + i;
                    int i2 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + (l.w - i - 1);
                    float swap = flip[i1];
                    flip[i1] = flip[i2];
                    flip[i2] = swap;
                    if(z == 0){
                        flip[i1] = -flip[i1];
                        flip[i2] = -flip[i2];
                    }
                }
            }
        }
    }
    //这里再把第一张图的相关数据跟第二张图处理以后的相关数据相加在平均。。从而最终得到一张图的数据
    for(i = 0; i < l.outputs; ++i){
        l.output[i] = (l.output[i] + flip[i])/2.;
    }
}
//提取yolo层检测框到detection结构体
int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets)
{
    int i,j,n;
    //predictions代表当前层l的预测输出
    float *predictions = l.output;
    //当l.batch等于2的时候。。进行avg_flipped_yolo(l)
    //具体细节参考src/yolo_layer.c
    //通常test的时候batch都是1...
    if (l.batch == 2) avg_flipped_yolo(l);
    int count = 0;
    //根据循环去遍历数据。。提取相关信息的坐标。。然后写入detection结构体。。
    for (i = 0; i < l.w*l.h; ++i){
        //获得行列信息，这里的row和col实际上是cell的中心点位置。。具体看下面的get_yolo_box中的内容。。也就是在将预测左边转换成真实坐标的时候
        //会用到的cell中心点的位置
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            //根据对应索引提取坐标，entry_index实现细节参考src/yolo_layer.c
            //包括如何根据索引提取相关内容。。在entry_index的注释里面都有
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            //根据相关索引从l.output中取出对应值
            float objectness = predictions[obj_index];
            //根据置信度阈值thresh进行筛选
            if(objectness <= thresh) continue;
            //同样的获取索引的操作。。。
            int box_index  = entry_index(l, 0, n*l.w*l.h + i, 0);
            //然后上面去到了具体坐标的索引。。这里根据索引提起框的坐标信息，顺便一起把detection结构体中其他的相关内容进行赋值
            //get_yolo_box的具体实现参考src/yolo_layer.c
            //这个函数结束取到的坐标值是相对于网络输入尺寸的坐标。。
            dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h);
            dets[count].objectness = objectness;
            dets[count].classes = l.classes;
            //这里提取每个类别预测的索引。。然后根据索引提取类别预测结果并赋值。。
            for(j = 0; j < l.classes; ++j){
                int class_index = entry_index(l, 0, n*l.w*l.h + i, 4 + 1 + j);
                //这里也看到了detection结构体中的prob存储了根据阈过滤后的类被预测信息。。低于阈值的类别预测被置0了。。
                //从上面能看到。。objectness表示当前框的置信度预测。。而这里的prob则是置信度乘上类别预测。。。
                //实际上就是预测的置信度乘上预测的类别概率作为最终的预测的结果。。写入了prob中。。。
                //当然。这个结果小于thresh的会被置0
                float prob = objectness*predictions[class_index];
                dets[count].prob[j] = (prob > thresh) ? prob : 0;
            }
            //一个框的内容提取完毕。。更新计数。。为下一次提取定位
            ++count;
        }
    }
    //correct_yolo_boxes的具体实现参考src/yolo_layer.c
    //最后对上面得到的b_x,b_y,b_h,b_w进行修正。。
    //在test_yolo模式里面。。这里的参数，w=h=1....具体参考example/yolo.c中的test_yolo函数里面的get_network_boxes(net, 1, 1, thresh, 0, 0, 0, &nboxes);
    //具体的意义是上面的box坐标其实目前还是在网络输入图片上的坐标，并不是真实图片上的坐标。。这个修正，。。就是把相对于网络输入尺寸上的框坐标修正到真正的输入图片上的坐标
    correct_yolo_boxes(dets, count, w, h, netw, neth, relative);
    return count;
}

#ifdef GPU
//yolo层前向传播的gpu实现
void forward_yolo_layer_gpu(const layer l, network net)
{
    copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
    int b, n;
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array_gpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array_gpu(l.output_gpu + index, (1+l.classes)*l.w*l.h, LOGISTIC);
        }
    }
    if(!net.train || l.onlyforward){
        cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        return;
    }

    cuda_pull_array(l.output_gpu, net.input, l.batch*l.inputs);
    forward_yolo_layer(l, net);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
}
//yolo层反向传播的gpu实现
void backward_yolo_layer_gpu(const layer l, network net)
{
    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif

