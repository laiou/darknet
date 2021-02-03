#include "region_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
//创建一个region层，用于检测框的回归，在yolov2中有用到
layer make_region_layer(int batch, int w, int h, int n, int classes, int coords)
{
    layer l = {0};
    l.type = REGION;
    //这里的n表示每个位置预测多少个框
    l.n = n;
    l.batch = batch;
    l.h = h;
    l.w = w;
    //输入的通道数也就是n*(classes + coords + 1);
    //classes表示预测的类别数量，coords表示预测的坐标
    l.c = n*(classes + coords + 1);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.coords = coords;
    l.cost = calloc(1, sizeof(float));
    l.biases = calloc(n*2, sizeof(float));
    l.bias_updates = calloc(n*2, sizeof(float));
    l.outputs = h*w*n*(classes + coords + 1);
    l.inputs = l.outputs;
    //l.truths表示真值的长度，这里的30是假设一张图片上最多有30个真值框。。
    l.truths = 30*(l.coords + 1);
    l.delta = calloc(batch*l.outputs, sizeof(float));
    l.output = calloc(batch*l.outputs, sizeof(float));
    int i;
    //初始化偏置为0.5
    for(i = 0; i < n*2; ++i){
        l.biases[i] = .5;
    }
    //region层的前向传播，具体实现参考src/region_layer.c
    l.forward = forward_region_layer;
    //region层的反向传播，具体实现参考src/region_layer.c
    l.backward = backward_region_layer;
#ifdef GPU
    //gpu版本的region层的前向传播，具体实现参考src/region_layer.c
    l.forward_gpu = forward_region_layer_gpu;
    //gpu版本的region层的反向传播，具体实现参考src/region_layer.c
    l.backward_gpu = backward_region_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "detection\n");
    srand(0);

    return l;
}

void resize_region_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->outputs = h*w*l->n*(l->classes + l->coords + 1);
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
//提取region层的box信息。。转换到相对于网络输尺度的坐标。。将预测的t_x,t_y等变成b_x,b_y等
box get_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0*stride]) / w;
    b.y = (j + x[index + 1*stride]) / h;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}
//delta_region_box(truth, l.output, l.biases, n, box_index, i, j, l.w, l.h, l.delta, .01, l.w*l.h);
// truth.w = l.biases[2*n]/l.w;
//truth.h = l.biases[2*n+1]/l.h;
//delta_region_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, l.delta, l.coord_scale *  (2 - truth.w*truth.h), l.w*l.h);
float delta_region_box(box truth, float *x, float *biases, int n, int index, int i, int j, int w, int h, float *delta, float scale, int stride)
{   
    //提取对应的预测框坐标
    box pred = get_region_box(x, biases, n, index, i, j, w, h, stride);
    //计算预测框和真值框的iou
    float iou = box_iou(pred, truth);
    //这里将得到的坐标转化成相对于当前cell左上角的偏移
    float tx = (truth.x*w - i);
    float ty = (truth.y*h - j);
    //关于tw和th。。。需要注意的是从output中直接俄得到的值眼睛过转换才能算是框坐标里面的w,h，具体参考上面的get_region_box函数
    //而这里的truth.w表示的是当前真值框的width相对于当前特征图尺度w的比例，实际上算是box中的坐标值了。。
    //所以要进行box坐标值到prev值的转换，结合上面get_region_box函数中的内容，就能方便的看出
    float tw = log(truth.w*w / biases[2*n]);
    float th = log(truth.h*h / biases[2*n + 1]);
    //利用预测坐标和相关的真值坐标更新l.delta。根据坐标的损失函数，求导即可
    delta[index + 0*stride] = scale * (tx - x[index + 0*stride]);
    delta[index + 1*stride] = scale * (ty - x[index + 1*stride]);
    delta[index + 2*stride] = scale * (tw - x[index + 2*stride]);
    delta[index + 3*stride] = scale * (th - x[index + 3*stride]);
    //同时返回iou
    return iou;
}
//delta_region_mask(net.truth + t*(l.coords + 1) + b*l.truths + 5, l.output, l.coords - 4, mask_index, l.delta, l.w*l.h, l.mask_scale);
void delta_region_mask(float *truth, float *x, int n, int index, float *delta, int stride, int scale)
{
    int i;
    for(i = 0; i < n; ++i){
        delta[index + i*stride] = scale*(truth[i] - x[index + i*stride]);
    }
}

//delta_region_class(l.output, l.delta, class_index, class, l.classes, l.softmax_tree, l.class_scale, l.w*l.h, &avg_cat, !l.softmax);
void delta_region_class(float *output, float *delta, int index, int class, int classes, tree *hier, float scale, int stride, float *avg_cat, int tag)
{
    int i, n;
    //l.softmax_tree一般不使用
    if(hier){
        float pred = 1;
        while(class >= 0){
            pred *= output[index + stride*class];
            int g = hier->group[class];
            int offset = hier->group_offset[g];
            for(i = 0; i < hier->group_size[g]; ++i){
                delta[index + stride*(offset + i)] = scale * (0 - output[index + stride*(offset + i)]);
            }
            delta[index + stride*class] = scale * (1 - output[index + stride*class]);

            class = hier->parent[class];
        }
        *avg_cat += pred;
    } else {
        //通常都是进入这里了
        //如果delta[index]存在而且不进行softmax的话
        if (delta[index] && tag){
            //计算相应的类别预测的delta，index + stride*class是因为class算是一个类别id吧
            //然后真值的label是1...
            delta[index + stride*class] = scale * (1 - output[index + stride*class]);
            return;
        }
        //遍历全部的类别
        for(n = 0; n < classes; ++n){
            //更新其他类别的delta,负样本的label是0
            delta[index + stride*n] = scale * (((n == class)?1 : 0) - output[index + stride*n]);
            //将预测为真值的预测框的类别的预测值累加到avg_cat
            if(n == class) *avg_cat += output[index + stride*n];
        }
    }
}

float logit(float x)
{
    return log(x/(1.-x));
}

float tisnan(float x)
{
    return (x != x);
}

int entry_index(layer l, int batch, int location, int entry)
{
    int n =   location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    return batch*l.outputs + n*l.w*l.h*(l.coords+l.classes+1) + entry*l.w*l.h + loc;
}
//region层的前向传播
void forward_region_layer(const layer l, network net)
{
    int i,j,b,t,n;
    //将net.input中的输入复制到l.output中。。region相当于是一个loss计算层，所以输入和输出一致
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));

#ifndef GPU
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array(l.output + index, 2*l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, l.coords);
            if(!l.background) activate_array(l.output + index, l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, l.coords + 1);
            if(!l.softmax && !l.softmax_tree) activate_array(l.output + index, l.classes*l.w*l.h, LOGISTIC);
        }
    }
    if (l.softmax_tree){
        int i;
        int count = l.coords + 1;
        for (i = 0; i < l.softmax_tree->groups; ++i) {
            int group_size = l.softmax_tree->group_size[i];
            softmax_cpu(net.input + count, group_size, l.batch, l.inputs, l.n*l.w*l.h, 1, l.n*l.w*l.h, l.temperature, l.output + count);
            count += group_size;
        }
    } else if (l.softmax){
        int index = entry_index(l, 0, 0, l.coords + !l.background);
        softmax_cpu(net.input + index, l.classes + l.background, l.batch*l.n, l.inputs/l.n, l.w*l.h, 1, l.w*l.h, 1, l.output + index);
    }
#endif
    //将l.delta中的值设置成0
    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    if(!net.train) return;
    float avg_iou = 0;
    float recall = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    int class_count = 0;
    *(l.cost) = 0;
    //遍历batch中每一张图片的处理数据
    for (b = 0; b < l.batch; ++b) {
        //一般这里不执行,从parser.c中的parse_region函数能看出这里的l.softmax_tree需要在cfg中指定一个tree_file路径。。
        //通常情况都没有用到这个参数。。
        if(l.softmax_tree){
            int onlyclass = 0;
            for(t = 0; t < 30; ++t){
                box truth = float_to_box(net.truth + t*(l.coords + 1) + b*l.truths, 1);
                if(!truth.x) break;
                int class = net.truth[t*(l.coords + 1) + b*l.truths + l.coords];
                float maxp = 0;
                int maxi = 0;
                if(truth.x > 100000 && truth.y > 100000){
                    for(n = 0; n < l.n*l.w*l.h; ++n){
                        int class_index = entry_index(l, b, n, l.coords + 1);
                        int obj_index = entry_index(l, b, n, l.coords);
                        float scale =  l.output[obj_index];
                        l.delta[obj_index] = l.noobject_scale * (0 - l.output[obj_index]);
                        float p = scale*get_hierarchy_probability(l.output + class_index, l.softmax_tree, class, l.w*l.h);
                        if(p > maxp){
                            maxp = p;
                            maxi = n;
                        }
                    }
                    int class_index = entry_index(l, b, maxi, l.coords + 1);
                    int obj_index = entry_index(l, b, maxi, l.coords);
                    delta_region_class(l.output, l.delta, class_index, class, l.classes, l.softmax_tree, l.class_scale, l.w*l.h, &avg_cat, !l.softmax);
                    if(l.output[obj_index] < .3) l.delta[obj_index] = l.object_scale * (.3 - l.output[obj_index]);
                    else  l.delta[obj_index] = 0;
                    l.delta[obj_index] = 0;
                    ++class_count;
                    onlyclass = 1;
                    break;
                }
            }
            if(onlyclass) continue;
        }
        //这里就是遍历某张图全部的预测结果
        //这里主要完成对相应框坐标梯度的更新和对不包含目标的框的梯度的更新，也就是noobj梯度
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {
                    //box_index提取出相应位置上的框坐标的起始处的索引，这个索引是在output中的索引或者说是在预测值里面的索引
                    //entry_index的具体实现参考src/region_layer.c
                    int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                    //根据索引得到相应框的坐标
                    //get_region_box的具体实现参考src/region_layer.c
                    box pred = get_region_box(l.output, l.biases, n, box_index, i, j, l.w, l.h, l.w*l.h);
                    //给best_iou初始化成0
                    float best_iou = 0;
                    //遍历30个真值框。。因为预设留下了30个真值框的位置
                    for(t = 0; t < 30; ++t){
                        //float_to_box的具体实现参考src/box.c，也是一个从记录真值信息的net.truth中提取框坐标的操作
                        //提取相应的真值框的坐标
                        box truth = float_to_box(net.truth + t*(l.coords + 1) + b*l.truths, 1);
                        //如果没有提取到坐标。。break。。。
                        if(!truth.x) break;
                        //计算当前取到的预测框和读取的真值框的iou
                        float iou = box_iou(pred, truth);
                        //比较更新best_iou，best_iou始终保持iou的最大值
                        if (iou > best_iou) {
                            best_iou = iou;
                        }
                    }
                    //到了这里的话。。best_iou里面的值实际上表示的就是当前预测框跟这张图片上所有真值框的iou的最大值
                    //这里是计算prob的坐标，也是output中的坐标值
                    int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, l.coords);
                    //将相应预测框对应的预测的prob值累加到avg_anyobj中
                    avg_anyobj += l.output[obj_index];
                    //更新obj的l.delta，这里参考一下具体损失函数的公式，求一下导数就能看出来了，0是因为noobj的真值是0.。
                    //如果这个框不包括目标，那么他的真值的prob就是0，应为l.delta里面包括了obj和noobj的梯度，所以这里先直接把noobj的梯度更新进去
                    //先假设这个框没有目标。。具体逻辑结合后面的内容
                    l.delta[obj_index] = l.noobject_scale * (0 - l.output[obj_index]);
                    //这里的l.background表示如果考虑背景的话。。也就是把背景也当成一个检测的类别的话
                    //那么这里的框就相当于是有目标了，所以真值就是1了
                    if(l.background) l.delta[obj_index] = l.noobject_scale * (1 - l.output[obj_index]);
                    //如果best_iou的值大于l.thresh也就是当前层的阈值
                    if (best_iou > l.thresh) {
                        //将相应预测框的prob的梯度置0
                        //因为超过了阈值意味着这个框可以留下来，里面是有目标的。。所以将上面已经写进去的noobj的梯度清除
                        l.delta[obj_index] = 0;
                    }
                    //如果已处理的图片数量小于12800
                    if(*(net.seen) < 12800){
                        box truth = {0};
                        //直接用当前cell的中心点坐标作为真值框的x,y
                        //l.biases的初始值都是0.5
                        //这里的x,y是相对于输入特征图左上角的偏移量
                        truth.x = (i + .5)/l.w;
                        truth.y = (j + .5)/l.h;
                        truth.w = l.biases[2*n]/l.w;
                        truth.h = l.biases[2*n+1]/l.h;
                        //delta_region_layer的具体实现参考src/region_layer.c
                        //根据相应的真值更新Box的坐标的梯度
                        delta_region_box(truth, l.output, l.biases, n, box_index, i, j, l.w, l.h, l.delta, .01, l.w*l.h);
                    }
                }
            }
        }
        //接着还是遍历真值里面全部的真值框
        for(t = 0; t < 30; ++t){
            //提取对应的真值框的坐标
            box truth = float_to_box(net.truth + t*(l.coords + 1) + b*l.truths, 1);

            if(!truth.x) break;
            float best_iou = 0;
            int best_n = 0;
            //结合get_region_box函数中的操作
            //可以看到这里将truth.x原来是相对于对应cell左上角的偏移，现在变成了相对于输入特征图左上角的偏移量
            //因为i,j都是int类型。。。。所以都会被取整。。
            //通过i，j将相应的真值分配到对应的cell里面。。从而下面就只需要遍历这个cell下预测出的框了
            i = (truth.x * l.w);
            //跟上面的thruth.y一样
            j = (truth.y * l.h);
            //生成一个truth的备份
            box truth_shift = truth;
            //将备份框中的x,y置0
            truth_shift.x = 0;
            truth_shift.y = 0;
            //遍历某一个cell下预测的每一个框
            for(n = 0; n < l.n; ++n){
                //还是提取相应box的坐标吧，output里面的坐标，这里通过i,j来确定当前提取到的真值框分配到哪一个cell
                int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                //根据box_index提取对应的box坐标的预测值
                box pred = get_region_box(l.output, l.biases, n, box_index, i, j, l.w, l.h, l.w*l.h);
                //如果需要bias_match
                if(l.bias_match){
                    pred.w = l.biases[2*n]/l.w;
                    pred.h = l.biases[2*n+1]/l.h;
                }
                //将上面提取的预测框中的x,y置0
                pred.x = 0;
                pred.y = 0;
                //计算此时pred和truth_shift的iou。。关于这里的置0，还要再想想*****************
                float iou = box_iou(pred, truth_shift);
                //记录当前cell预测框和当前cell下真值框的最大iou和对应预测框的id
                if (iou > best_iou){
                    best_iou = iou;
                    best_n = n;
                }
            }
            //根据box坐标的索引，也就是上面得到的最大iou的框的坐标在output中的索引
            int box_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, 0);
            //计算真值框和上面best_iou的框的实际上的iou，同时计算相应的坐标的梯度
            //虽然在上一个循环里面更新了坐标的梯度。。但是那里面并不是利用标注的真值框更新的，在这里将和真值框iou比较大的框的梯度用标注框进行重写
            float iou = delta_region_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, l.delta, l.coord_scale *  (2 - truth.w*truth.h), l.w*l.h);
            //如果lcoords大于4
            if(l.coords > 4){
                //则计算一个mask索引，也就是说这时候预测输出的坐标中多出来的部分看作mask
                //这里对多出来的mask部分计算梯度
                int mask_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, 4);
                //delta_region_mask的实现参考src/region_layer.c
                delta_region_mask(net.truth + t*(l.coords + 1) + b*l.truths + 5, l.output, l.coords - 4, mask_index, l.delta, l.w*l.h, l.mask_scale);
            }
            //如果iou>0.5，则recall计数加1
            //也就是真值框相应cell下最匹配的预测框的iou
            if(iou > .5) recall += 1;
            //将这个iou记录到avg_iou中，这里记录的是预测框包括目标的iou的值
            avg_iou += iou;
            //提取prob预测的索引
            int obj_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, l.coords);
            //将对应的预测值加到avg_obj里面，这里统计的是预测框包含目标的prob值
            avg_obj += l.output[obj_index];
            //因为检测框包含目标，将对应的l.delta重写
            l.delta[obj_index] = l.object_scale * (1 - l.output[obj_index]);
            //如果需要rescore
            if (l.rescore) {
                //用iou重新更新梯度
                l.delta[obj_index] = l.object_scale * (iou - l.output[obj_index]);
            }
            //如果把背景当成一个检测类别
            if(l.background){
                //那么对于当前的包含目标的框来说的话，它相对于背景的label就应该是0
                l.delta[obj_index] = l.object_scale * (0 - l.output[obj_index]);
            }
            //提取对应的真值框的预测类别
            int class = net.truth[t*(l.coords + 1) + b*l.truths + l.coords];
            //如果需要map
            if (l.map) class = l.map[class];
            //计算对应的预测框预测的类别信息的索引
            int class_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, l.coords + 1);
            //delta_region_class的具体实现参考src/region_layer.c
            //计算类别的delta同时将预测值为真值的类别的预测值累加到avg_cat
            delta_region_class(l.output, l.delta, class_index, class, l.classes, l.softmax_tree, l.class_scale, l.w*l.h, &avg_cat, !l.softmax);
            //记录一张图上预测框包含目标的框的个数
            ++count;
            //计数一张图上预测的包含目标的框的类别也预测正确的个数
            ++class_count;
        }
    }
    //一个batch处理完，计算loss，结合损失函数和此时l.delta里面的值很容易看出来。。。
    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    printf("Region Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, Avg Recall: %f,  count: %d\n", avg_iou/count, avg_cat/class_count, avg_obj/count, avg_anyobj/(l.w*l.h*l.n*l.batch), recall/count, count);
}
//region层的反向传播，在前向传播中l.delta已经计算好了。。所以反向传播将其传递到net.delta即可
void backward_region_layer(const layer l, network net)
{
    /*
       int b;
       int size = l.coords + l.classes + 1;
       for (b = 0; b < l.batch*l.n; ++b){
       int index = (b*size + 4)*l.w*l.h;
       gradient_array(l.output + index, l.w*l.h, LOGISTIC, l.delta + index);
       }
       axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
     */
    //这里原来是个空函数。。但是这一句应该还是要的。。。
    axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
}

void correct_region_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w=0;
    int new_h=0;
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
//将region层的坐标框信息提取到detection结构体中。。。这个函数结束之后。。其中的detection结构体存储的是最终相对于原图片的坐标
void get_region_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, float tree_thresh, int relative, detection *dets)
{
    int i,j,n,z;
    float *predictions = l.output;
    //这里的过程跟yolo层中的类似。。细节就不在写了
    if (l.batch == 2) {
        float *flip = l.output + l.outputs;
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w/2; ++i) {
                for (n = 0; n < l.n; ++n) {
                    for(z = 0; z < l.classes + l.coords + 1; ++z){
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
        for(i = 0; i < l.outputs; ++i){
            l.output[i] = (l.output[i] + flip[i])/2.;
        }
    }
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int index = n*l.w*l.h + i;
            for(j = 0; j < l.classes; ++j){
                dets[index].prob[j] = 0;
            }
            //抽取各种索引。。。详细的参考src/yolo_layer.c中的注释
            //这里的entry_index引用的是本文件下的entry_index。。但是逻辑上实现上和yolo中的几乎完全一致
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, l.coords);
            int box_index  = entry_index(l, 0, n*l.w*l.h + i, 0);
            int mask_index = entry_index(l, 0, n*l.w*l.h + i, 4);
            float scale = l.background ? 1 : predictions[obj_index];
            //提取具体的box坐标到dets。。。这里结束得到的还是相对于网络输入尺寸的坐标。。
            //具体实现参考src/region_layer.c
            dets[index].bbox = get_region_box(predictions, l.biases, n, box_index, col, row, l.w, l.h, l.w*l.h);
            dets[index].objectness = scale > thresh ? scale : 0;
            if(dets[index].mask){
                for(j = 0; j < l.coords - 4; ++j){
                    dets[index].mask[j] = l.output[mask_index + j*l.w*l.h];
                }
            }

            int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + !l.background);
            //这里是跟yolo层不同的部分。。算是region层独有的了。。后面再补**********
            if(l.softmax_tree){

                hierarchy_predictions(predictions + class_index, l.classes, l.softmax_tree, 0, l.w*l.h);
                if(map){
                    for(j = 0; j < 200; ++j){
                        int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + 1 + map[j]);
                        float prob = scale*predictions[class_index];
                        dets[index].prob[j] = (prob > thresh) ? prob : 0;
                    }
                } else {
                    int j =  hierarchy_top_prediction(predictions + class_index, l.softmax_tree, tree_thresh, l.w*l.h);
                    dets[index].prob[j] = (scale > thresh) ? scale : 0;
                }
            } else {
                if(dets[index].objectness){
                    for(j = 0; j < l.classes; ++j){
                        int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + 1 + j);
                        float prob = scale*predictions[class_index];
                        dets[index].prob[j] = (prob > thresh) ? prob : 0;
                    }
                }
            }
        }
    }
    //修正上面的坐标到原图片的尺度
    correct_region_boxes(dets, l.w*l.h*l.n, w, h, netw, neth, relative);
}

#ifdef GPU
//region层前向传播的gpu实现。。对照着cpu版本的实现来看
void forward_region_layer_gpu(const layer l, network net)
{
    copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
    int b, n;
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array_gpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC);
            if(l.coords > 4){
                index = entry_index(l, b, n*l.w*l.h, 4);
                activate_array_gpu(l.output_gpu + index, (l.coords - 4)*l.w*l.h, LOGISTIC);
            }
            index = entry_index(l, b, n*l.w*l.h, l.coords);
            if(!l.background) activate_array_gpu(l.output_gpu + index,   l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, l.coords + 1);
            if(!l.softmax && !l.softmax_tree) activate_array_gpu(l.output_gpu + index, l.classes*l.w*l.h, LOGISTIC);
        }
    }
    if (l.softmax_tree){
        int index = entry_index(l, 0, 0, l.coords + 1);
        softmax_tree(net.input_gpu + index, l.w*l.h, l.batch*l.n, l.inputs/l.n, 1, l.output_gpu + index, *l.softmax_tree);
    } else if (l.softmax) {
        int index = entry_index(l, 0, 0, l.coords + !l.background);
        softmax_gpu(net.input_gpu + index, l.classes + l.background, l.batch*l.n, l.inputs/l.n, l.w*l.h, 1, l.w*l.h, 1, l.output_gpu + index);
    }
    if(!net.train || l.onlyforward){
        cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        return;
    }

    cuda_pull_array(l.output_gpu, net.input, l.batch*l.inputs);
    forward_region_layer(l, net);
    //cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);
    if(!net.train) return;
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
}

//region层反向传播的gpu版本的实现。。对照着cpu版本的实现来看。。基本过程一致
void backward_region_layer_gpu(const layer l, network net)
{
    int b, n;
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            gradient_array_gpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC, l.delta_gpu + index);
            if(l.coords > 4){
                index = entry_index(l, b, n*l.w*l.h, 4);
                gradient_array_gpu(l.output_gpu + index, (l.coords - 4)*l.w*l.h, LOGISTIC, l.delta_gpu + index);
            }
            index = entry_index(l, b, n*l.w*l.h, l.coords);
            if(!l.background) gradient_array_gpu(l.output_gpu + index,   l.w*l.h, LOGISTIC, l.delta_gpu + index);
        }
    }
    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif

void zero_objectness(layer l)
{
    int i, n;
    for (i = 0; i < l.w*l.h; ++i){
        for(n = 0; n < l.n; ++n){
            int obj_index = entry_index(l, 0, n*l.w*l.h + i, l.coords);
            l.output[obj_index] = 0;
        }
    }
}

