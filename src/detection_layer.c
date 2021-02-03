#include "detection_layer.h"
#include "activations.h"
#include "softmax_layer.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
//创建一个detection层
//detection层其实处理检测结果的那一层，类似loss计算那一层，用于yolov1中
detection_layer make_detection_layer(int batch, int inputs, int n, int side, int classes, int coords, int rescore)
{
    detection_layer l = {0};
    l.type = DETECTION;
    //这里的l.n表示每个cell预测的检测框的个数
    l.n = n;
    l.batch = batch;
    l.inputs = inputs;
    l.classes = classes;
    l.coords = coords;
    l.rescore = rescore;
    //side就是比如说yolo中将图片划分成7x7的cell里面的7
    l.side = side;
    //因为是出结果的那一层，所以最终尺度的w和h，应该和side一致
    l.w = side;
    l.h = side;
    //再次判断维度是否正确
    assert(side*side*((1 + l.coords)*l.n + l.classes) == inputs);
    l.cost = calloc(1, sizeof(float));
    //只是计算损失，输出维度和原来的inputs一致
    l.outputs = l.inputs;
    //真实label的参数量
    l.truths = l.side*l.side*(1+l.coords+l.classes);
    l.output = calloc(batch*l.outputs, sizeof(float));
    l.delta = calloc(batch*l.outputs, sizeof(float));
    //detection层的前向传播.具体实现参考src/detection_layer.c
    l.forward = forward_detection_layer;
    //detection层的反向传播，具体实现参考src/detection_layer.c
    l.backward = backward_detection_layer;
#ifdef GPU
    //gpu版本的detection层
    //detection层的gpu版本的前向传播，具体实现参考src/detection_layer.c
    l.forward_gpu = forward_detection_layer_gpu;
    //gpu版本的detection层的反向传播，具体实现参考src/detection_layer.c
    l.backward_gpu = backward_detection_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "Detection Layer\n");
    srand(0);

    return l;
}
//detection层的前向传播
void forward_detection_layer(const detection_layer l, network net)
{   
    //locations表示cell数量
    int locations = l.side*l.side;
    int i,j;
    //从net.input中复制相关数据到l.output中,这一层只是计算损失。。。处理检测结果。。输出和输入一样的。。。
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));
    //if(l.reorg) reorg(l.output, l.w*l.h, size*l.n, l.batch, 1);
    int b;
    //如果需要softmax操作
    if (l.softmax){
        //遍历batch中每一张图的相关数据
        for(b = 0; b < l.batch; ++b){
            //index计算每张图相关数据起点的索引
            int index = b*l.inputs;
            //遍历具体的数据，以cell为单位遍历
            for (i = 0; i < locations; ++i) {
                //这里和yolov1输出数据的存储方式有关系，具体参考yolov1输出张量格式
                //这里存储的格式是类别，prob，框坐标，比如cell个数是4个，每一个cell预测两个框，预测两类的话：
                //class_1_cell0,class_2_cell_0,class_1_cell_1,class_2_cell_1,class_1_cell_2,class_2_cell_2,class_1_cell_3,class_2_cell_3,
                //prob0_cell0,prob1_cell0,prob0_cell1,prob1_cell1,prob0_cell2,prob1_cell2,prob0_cell3,prob1_cell3,
                //然后就是x,y,w,h等等坐标了，也是根据cell的顺序来的
                //从而得到了下面的索引
                int offset = i*l.classes;
                //softmax的具体实现参考src/blas.c
                softmax(l.output + index + offset, l.classes, 1, 1,
                        l.output + index + offset);
            }
        }
    }
    //训练过程中
    if(net.train){
        float avg_iou = 0;
        float avg_cat = 0;
        float avg_allcat = 0;
        float avg_obj = 0;
        float avg_anyobj = 0;
        int count = 0;
        *(l.cost) = 0;
        int size = l.inputs * l.batch;
        //初始化l.delta为0,这里将l.delta清0,上面初始化的时候已经是0了，而且从后面可以看出
        //对l.delta的操作还都是重写里面的值。。。
        memset(l.delta, 0, size * sizeof(float));
        //遍历batch中每张图的结果
        for (b = 0; b < l.batch; ++b){
            //index表示每张图数据起点的索引
            int index = b*l.inputs;
            //locations表示cell个数
            for (i = 0; i < locations; ++i) {
                //提取相应类别置信度真值的索引，和真值的存储方式有关系
                int truth_index = (b*locations + i)*(1+l.coords+l.classes);
                int is_obj = net.truth[truth_index];
                for (j = 0; j < l.n; ++j) {
                    int p_index = index + locations*l.classes + i*l.n + j;
                    //初始化l.delta中的相关数值，这里实际上是obj误差的计算
                    //这里从l.output中取出的实际上是预测出来的prob的值
                    //更新noobj的梯度，当前cell没有真值框的话。。真值prob就是0....
                    l.delta[p_index] = l.noobject_scale*(0 - l.output[p_index]);
                    //这里是先把全部框的objloss都按照noobj来处理的损失。。后面会根据实际情况调整
                    *(l.cost) += l.noobject_scale*pow(l.output[p_index], 2);
                    //累积全部prob的预测值
                    avg_anyobj += l.output[p_index];
                }

                int best_index = -1;
                float best_iou = 0;
                float best_rmse = 20;

                if (!is_obj){
                    //如果当前cell没有框。。。
                    continue;
                }
                //继续计算类别预测的索引
                int class_index = index + i*l.classes;
                for(j = 0; j < l.classes; ++j) {
                    //根据分类误差的损失计算梯度，统计误差
                    l.delta[class_index+j] = l.class_scale * (net.truth[truth_index+1+j] - l.output[class_index+j]);
                    *(l.cost) += l.class_scale * pow(net.truth[truth_index+1+j] - l.output[class_index+j], 2);
                    //分别统计正确类别下的预测值和全部类别的预测值的和
                    if(net.truth[truth_index + 1 + j]) avg_cat += l.output[class_index+j];
                    avg_allcat += l.output[class_index+j];
                }
                //提取当前cell下真值框坐标
                box truth = float_to_box(net.truth + truth_index + 1 + l.classes, 1);
                //转换成相应比例下的尺度
                truth.x /= l.side;
                truth.y /= l.side;
                //最终从当前cell预测的多个框里面选出一个best_box作为当前的预测真值。。就是相当于是矮子里面选个高的
                //认为他就是预测值，参与objloss的计算
                //毕竟训练过程中，总是要有正负样本来参与loss的计算的。。这里相当于是一种筛选预测值作为正负样本的方式
                for(j = 0; j < l.n; ++j){
                    //计算box坐标的索引
                    int box_index = index + locations*(l.classes + l.n) + (i*l.n + j) * l.coords;
                    //提取预测box的坐标
                    //具体实现参考src/box.c
                    box out = float_to_box(l.output + box_index, 1);
                    out.x /= l.side;
                    out.y /= l.side;

                    if (l.sqrt){
                        out.w = out.w*out.w;
                        out.h = out.h*out.h;
                    }
                    //计算预测框和真实框的iou
                    //具体实现参考src/box.c
                    float iou  = box_iou(out, truth);
                    //iou = 0;
                    //计算预测和真值框的均方误差。。
                    //具体实现参考src/box.c
                    float rmse = box_rmse(out, truth);

                    if(best_iou > 0 || iou > 0){
                        //定位预测框和真值框iou最大的那个框
                        if(iou > best_iou){
                            best_iou = iou;
                            best_index = j;
                        }
                    }else{
                        //如果预测和真值不相交，直接计算误差损失
                        if(rmse < best_rmse){
                            //更新最小损失的值，记录最小均方误差的框
                            best_rmse = rmse;
                            best_index = j;
                        }
                    }
                }

                if(l.forced){
                    if(truth.w*truth.h < .1){
                        best_index = 1;
                    }else{
                        best_index = 0;
                    }
                }
                if(l.random && *(net.seen) < 64000){
                    best_index = rand()%l.n;
                }
                //计算上面的best_box的索引和对应真值的索引
                int box_index = index + locations*(l.classes + l.n) + (i*l.n + best_index) * l.coords;
                int tbox_index = truth_index + 1 + l.classes;
                //提取best_box框坐标
                box out = float_to_box(l.output + box_index, 1);
                out.x /= l.side;
                out.y /= l.side;
                if (l.sqrt) {
                    out.w = out.w*out.w;
                    out.h = out.h*out.h;
                }
                //计算best_box和真值框的iou
                float iou  = box_iou(out, truth);

                //printf("%d,", best_index);
                //计算best_box的obj的损失
                int p_index = index + locations*l.classes + i*l.n + best_index;
                //因为上面把全部预测狂都当成是noobj来计算损失并累加起来了，现在best_box是有框的
                //所以先减去之前的noonj loss，然后重新加上objloss
                *(l.cost) -= l.noobject_scale * pow(l.output[p_index], 2);
                *(l.cost) += l.object_scale * pow(1-l.output[p_index], 2);
                //累积认为是真值框的prob预测值
                avg_obj += l.output[p_index];
                //计算best_box的obj的梯度。。这里是存在框的，所以根据相关导数就能推导
                l.delta[p_index] = l.object_scale * (1.-l.output[p_index]);
                //需要rescore的话。重新调整一下best_box的梯度
                if(l.rescore){
                    l.delta[p_index] = l.object_scale * (iou - l.output[p_index]);
                }
                //更新具体的坐标值的梯度
                l.delta[box_index+0] = l.coord_scale*(net.truth[tbox_index + 0] - l.output[box_index + 0]);
                l.delta[box_index+1] = l.coord_scale*(net.truth[tbox_index + 1] - l.output[box_index + 1]);
                l.delta[box_index+2] = l.coord_scale*(net.truth[tbox_index + 2] - l.output[box_index + 2]);
                l.delta[box_index+3] = l.coord_scale*(net.truth[tbox_index + 3] - l.output[box_index + 3]);
                if(l.sqrt){
                    l.delta[box_index+2] = l.coord_scale*(sqrt(net.truth[tbox_index + 2]) - l.output[box_index + 2]);
                    l.delta[box_index+3] = l.coord_scale*(sqrt(net.truth[tbox_index + 3]) - l.output[box_index + 3]);
                }
                //统计坐标损失，加到总的损失里面去
                *(l.cost) += pow(1-iou, 2);
                //统计best_box和真值的iou，相当于是统计了预测值为真值的框的损失
                avg_iou += iou;
                //更新一下计数，可以看成是计数了预测为真值框的个数
                ++count;
            }
        }

        if(0){
            float *costs = calloc(l.batch*locations*l.n, sizeof(float));
            for (b = 0; b < l.batch; ++b) {
                int index = b*l.inputs;
                for (i = 0; i < locations; ++i) {
                    for (j = 0; j < l.n; ++j) {
                        int p_index = index + locations*l.classes + i*l.n + j;
                        costs[b*locations*l.n + i*l.n + j] = l.delta[p_index]*l.delta[p_index];
                    }
                }
            }
            int indexes[100];
            top_k(costs, l.batch*locations*l.n, 100, indexes);
            float cutoff = costs[indexes[99]];
            for (b = 0; b < l.batch; ++b) {
                int index = b*l.inputs;
                for (i = 0; i < locations; ++i) {
                    for (j = 0; j < l.n; ++j) {
                        int p_index = index + locations*l.classes + i*l.n + j;
                        if (l.delta[p_index]*l.delta[p_index] < cutoff) l.delta[p_index] = 0;
                    }
                }
            }
            free(costs);
        }

        //mag_array的具体实现参考src/utils.c，具体实现的是将l.delta中每一个值的平方相加求和，返回这个和的平方根。。
        //这里已经处理完了一个batch的数据，然后更新之一层的cost，看一下上面的l.delta的计算方式。。实际上l.cost的累加和这里的算法差别不大
        //主要是因为都是均方误差。。。
        *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);


        printf("Detection Avg IOU: %f, Pos Cat: %f, All Cat: %f, Pos Obj: %f, Any Obj: %f, count: %d\n", avg_iou/count, avg_cat/count, avg_allcat/(count*l.classes), avg_obj/count, avg_anyobj/(l.batch*locations*l.n), count);
        //if(l.reorg) reorg(l.delta, l.w*l.h, size*l.n, l.batch, 0);
    }
}
//detection_layer的反向传播
void backward_detection_layer(const detection_layer l, network net)
{   //具体是实现参考src/blas.c
    //这里实现的是将l.delta复制到net.delta中。向上一层传递
    axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
}
//提取detection层的坐标信息。。相对更直接了当。。。。。直接修正到相对于原图的尺度。。
void get_detection_detections(layer l, int w, int h, float thresh, detection *dets)
{
    int i,j,n;
    float *predictions = l.output;
    //int per_cell = 5*num+classes;
    for (i = 0; i < l.side*l.side; ++i){
        int row = i / l.side;
        int col = i % l.side;
        for(n = 0; n < l.n; ++n){
            int index = i*l.n + n;
            int p_index = l.side*l.side*l.classes + i*l.n + n;
            float scale = predictions[p_index];
            int box_index = l.side*l.side*(l.classes + l.n) + (i*l.n + n)*4;
            box b;
            b.x = (predictions[box_index + 0] + col) / l.side * w;
            b.y = (predictions[box_index + 1] + row) / l.side * h;
            b.w = pow(predictions[box_index + 2], (l.sqrt?2:1)) * w;
            b.h = pow(predictions[box_index + 3], (l.sqrt?2:1)) * h;
            dets[index].bbox = b;
            dets[index].objectness = scale;
            for(j = 0; j < l.classes; ++j){
                int class_index = i*l.classes;
                float prob = scale*predictions[class_index+j];
                dets[index].prob[j] = (prob > thresh) ? prob : 0;
            }
        }
    }
}

#ifdef GPU
//gpu版本的detection层的前向传播
void forward_detection_layer_gpu(const detection_layer l, network net)
{   
    //如果不在训练的话
    if(!net.train){
        //copy_gpu的具体实现参考src/blas_kernels.cu
        //这里实现的是将net.intput_gpu中的值赋值到l.output_gpu中去
        copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
        return;
    }
    //将一个矩阵从gpu设备拉取到主机，将数据从net.input_gpu拉取到net.input
    cuda_pull_array(net.input_gpu, net.input, l.batch*l.inputs);
    //进行detection层cpu版本的前向传播。。
    //具体实现参考src/detection_layer.c
    forward_detection_layer(l, net);
    //实现的是将l.output推送到gpu上的l.output_gpu
    cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);
    //这里实现的是将l.delta推送到l.delta_gpu上
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.inputs);
}
//gpu版本的detection层的反向传播
void backward_detection_layer_gpu(detection_layer l, network net)
{
    //这里实现的是将l.delta_gpu中的值加到net.delta_gpu中去
    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
    //copy_gpu(l.batch*l.inputs, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif

