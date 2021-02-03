#include "darknet.h"

char *voc_names[] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};
//训练yolo模型
void train_yolo(char *cfgfile, char *weightfile)
{
    //取到保存训练图片路径的txt
    char *train_images = "/data/voc/train.txt";
    //取到保存训练模型的文件路径
    char *backup_directory = "/home/pjreddie/backup/";
    //以当前系统时间作为随机种子
    srand(time(0));
    //basecfg的具体实现参考src/utils.c
    //取到具体训练的网络的名字。。从cfgfile的字符串中提取。。。。
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    //初始化avg_loss。。。。
    float avg_loss = -1;
    //从具体的cfg和weights文件中加载并船舰network。。。返回一个存储了相关参数的network结构
    //具体参考src/network.c
    //这里做的就是把预训练的模型参数加载进来。。给network结构相关权重等完成初始化。。。
    network *net = load_network(cfgfile, weightfile, 0);
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    //接下来就是跟数据相关的了
    //这里的imgs表示图片数量。。然后net->batch和net->subdivisions其实就是cfg文件中[net]组里面的batch和subdivisoons两个参数
    //这里imgs实际上网络一次加载的图片总数。。也就是通常我们所说的batch_size。。。
    //这里把通常所说的batch_size分成了subdivisions个子batch。。。
    int imgs = net->batch*net->subdivisions;
    //这里的net->seen是一个size_t类型的指针。。具体参考include/network.h
    //这里可以看成一个int类型
    //net->seen表示当前处理的图片的数量。。。
    //这个i表示当前处理的数据是第几个batch。。。
    //具体的参考后面的训练
    int i = *net->seen/imgs;
    //data的声明参考include/darknet.h
    //具体存储了图片数据。。label信息以及图片的w,h等等。。
    data train, buffer;

    //定位到预先加载的network的最后一层
    layer l = net->layers[net->n - 1];
    //l.side代表的是yolo系列算法中的cell的划分。。比如将一张图片划分成7x7的网格。。。这样的话。。l.side其实就是等于7
    //这个参数其实在[yolo]层通过l.w或者l.h也能取得到
    int side = l.side;
    //检测的类别数量
    int classes = l.classes;
    //这里的l.jitter的作用是对图像的宽高进行一些微小的变化。。。以yolov3.cfg为例。。[yolo]组就有这个jitter参数
    //具体的使用参考下面的代码
    float jitter = l.jitter;
    //从train_images指向的txt文件中读取具体图片的路径
    //get_paths的实现参考src/data.c
    //将每张图片的路径存储到一个list结构中
    list *plist = get_paths(train_images);
    //int N = plist->size;
    //list_to_array具体实现参考src/list.c
    //实现的功能是将plist里面存储的图片路径的字符串的指针存到一个指针数组中
    //嵌套逻辑就是plist->node->val。。这里的val也是一个指针。。指向了存储图片路径那个具体字符串的内存首地址。。
    //然后这里就是把全部node->val的指针提取到一个指针数组里面比如a[]。。然后返回这个指针数组第一个元素的指针。。。
    char **paths = (char **)list_to_array(plist);

    //laod_args结构体定义了一些相关参数。。具体声明参考include/darknet.h
    load_args args = {0};
    //给上面的load_args结构里面相关参数赋值
    args.w = net->w;
    args.h = net->h;
    args.paths = paths;
    //这里imgs表示实际上网络一次加载的图片数量
    args.n = imgs;
    //plist->size表示参与训练的图片总数
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    //从这里就看出side 的作用了。。。nu_boxes。。。。cell的划分。。。[yolo]层的l.w,l.h.。。。
    args.num_boxes = side;
    //buffer的作用后面你具体再看***********
    args.d = &buffer;
    //这里的args.type是一个data_type类型。。具体参考include/darknet.h的声明
    args.type = REGION_DATA;
    //angle参数。。图片预处理的旋转角度。。。
    args.angle = net->angle;
    //下面的这三个都是对图像亮度。色度等进行调节的参数了。。。
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;

    //pthread_t是一个线程的标识符
    //pthread_t在头文件中定义为typedef unsigned long int pthread_t;
    //load_data_in_thread的具体实现参考src/data.c
    //利用线程加载数据。。返回对应线程的标识符
    pthread_t load_thread = load_data_in_thread(args);
    //计时器声明
    clock_t time;
    //while(i*imgs < N*120){
    //接着就是训练的部分了。。。。
    //get_current_batch的具体实现参考src/network.c。。。 
    //前面也能看得到。。网络一次加载的图片数量就是net->batch*net->subdivisions
    //获得当前处理的数据在第几个batch_size里面。。判断有没有超出范围
    while(get_current_batch(net) < net->max_batches){
        i += 1;
        //取当前时间。。。
        time=clock();
        //利用pthread_join等待一个线程结束。。线程间的同步操作
        //就是等待load_thread指定的线程结束
        // int pthread_join(pthread_t thread, void **retval);
        //以阻塞的方式等待thread指定的线程结束。当函数返回时，被等待线程的资源被收回。如果线程已经结束，那么该函数会立即返回。
        //retval: 用户定义的指针，用来存储被等待线程的返回值
        //返回值 ： 0代表成功。 失败，返回的则是错误号。
        pthread_join(load_thread, 0);
        //将train指向前面线程加载到的数据
        train = buffer;
        //加载新的线程。。取取下一个batch_size的数据
        load_thread = load_data_in_thread(args);

        printf("Loaded: %lf seconds\n", sec(clock()-time));
        //刷新计时
        time=clock();
        //训练网络。。计算损失
        //train_neteork的具体实现参考src/network.c
        float loss = train_network(net, train);
        if (avg_loss < 0) avg_loss = loss;
        //这里就能看到avg_loss是一个加权值。。每一个batch_size的loss乘上0.1和原来的avg_loss的0.9加到一起
        //通过指数加权平均的方式得到
        avg_loss = avg_loss*.9 + loss*.1;
        //get_current_rate获取当前的学习率...
        //具体实现参考src/network.c************这个还没看。。。。
        printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n", i, loss, avg_loss, get_current_rate(net), sec(clock()-time), i*imgs);
        if(i%1000==0 || (i < 1000 && i%100 == 0)){
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            //保存权重，具体参考src/parser.c
            save_weights(net, buff);
        }
        //释放数据内存。。具体参考src/data.c
        free_data(train);
    }
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}

void print_yolo_detections(FILE **fps, char *id, int total, int classes, int w, int h, detection *dets)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            if (dets[i].prob[j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, dets[i].prob[j],
                    xmin, ymin, xmax, ymax);
        }
    }
}

void validate_yolo(char *cfg, char *weights)
{
    network *net = load_network(cfg, weights, 0);
    set_batch_network(net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

    char *base = "results/comp4_det_test_";
    //list *plist = get_paths("data/voc.2007.test");
    list *plist = get_paths("/home/pjreddie/data/voc/2007_test.txt");
    //list *plist = get_paths("data/voc.2012.test");
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n-1];
    int classes = l.classes;

    int j;
    FILE **fps = calloc(classes, sizeof(FILE *));
    for(j = 0; j < classes; ++j){
        char buff[1024];
        snprintf(buff, 1024, "%s%s.txt", base, voc_names[j]);
        fps[j] = fopen(buff, "w");
    }

    int m = plist->size;
    int i=0;
    int t;

    float thresh = .001;
    int nms = 1;
    float iou_thresh = .5;

    int nthreads = 8;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.type = IMAGE_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    time_t start = time(0);
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
            int nboxes = 0;
            detection *dets = get_network_boxes(net, w, h, thresh, 0, 0, 0, &nboxes);
            if (nms) do_nms_sort(dets, l.side*l.side*l.n, classes, iou_thresh);
            print_yolo_detections(fps, id, l.side*l.side*l.n, classes, w, h, dets);
            free_detections(dets, nboxes);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", (double)(time(0) - start));
}

void validate_yolo_recall(char *cfg, char *weights)
{
    network *net = load_network(cfg, weights, 0);
    set_batch_network(net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

    char *base = "results/comp4_det_test_";
    list *plist = get_paths("data/voc.2007.test");
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n-1];
    int classes = l.classes;
    int side = l.side;

    int j, k;
    FILE **fps = calloc(classes, sizeof(FILE *));
    for(j = 0; j < classes; ++j){
        char buff[1024];
        snprintf(buff, 1024, "%s%s.txt", base, voc_names[j]);
        fps[j] = fopen(buff, "w");
    }

    int m = plist->size;
    int i=0;

    float thresh = .001;
    float iou_thresh = .5;
    float nms = 0;

    int total = 0;
    int correct = 0;
    int proposals = 0;
    float avg_iou = 0;

    for(i = 0; i < m; ++i){
        char *path = paths[i];
        image orig = load_image_color(path, 0, 0);
        image sized = resize_image(orig, net->w, net->h);
        char *id = basecfg(path);
        network_predict(net, sized.data);

        int nboxes = 0;
        detection *dets = get_network_boxes(net, orig.w, orig.h, thresh, 0, 0, 1, &nboxes);
        if (nms) do_nms_obj(dets, side*side*l.n, 1, nms);

        char labelpath[4096];
        find_replace(path, "images", "labels", labelpath);
        find_replace(labelpath, "JPEGImages", "labels", labelpath);
        find_replace(labelpath, ".jpg", ".txt", labelpath);
        find_replace(labelpath, ".JPEG", ".txt", labelpath);

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);
        for(k = 0; k < side*side*l.n; ++k){
            if(dets[k].objectness > thresh){
                ++proposals;
            }
        }
        for (j = 0; j < num_labels; ++j) {
            ++total;
            box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
            float best_iou = 0;
            for(k = 0; k < side*side*l.n; ++k){
                float iou = box_iou(dets[k].bbox, t);
                if(dets[k].objectness > thresh && iou > best_iou){
                    best_iou = iou;
                }
            }
            avg_iou += best_iou;
            if(best_iou > iou_thresh){
                ++correct;
            }
        }

        fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total, (float)proposals/(i+1), avg_iou*100/total, 100.*correct/total);
        free_detections(dets, nboxes);
        free(id);
        free_image(orig);
        free_image(sized);
    }
}
//命令行输入yolo中的test_yolo模式
//yolo的推理测试模式
//这里接收的几个参数cfgfile是网络的cfg配置文件，weightfile是权重文件，filename是测试图片的文件夹，thresh是检测的阈值，就是得到boundingbox的时候需要的阈值
void test_yolo(char *cfgfile, char *weightfile, char *filename, float thresh)
{
    //image结构体存储图片数据具体声明参考darknet.h
    //load_alphabet实现参考src/image.c
    //这里只是分配相关内存，同时加载了部分/data/labels里面的图片，用这里面的图片作为test的label信息。。
    //alphabet的具体用法参考下面的draw_detections中的相关内容
    image **alphabet = load_alphabet();
    //加载模型。。。
    //load_network具体实现参考src/network.c
    network *net = load_network(cfgfile, weightfile, 0);
    //定位到整个ntwork结构的最后一个layer层。。也就是上面参数化生成的结构network里面存储的最后一个网络层次
    layer l = net->layers[net->n-1];
    //具体实现参考src/network.c
    //具体实现的操作就是把当前整个网络的batch参数设置成1。。
    set_batch_network(net, 1);
    //srand通常用来设置rand产生随机数的种子
    srand(2222222);
    //clock_t。。计时函数。。。
    clock_t time;
    char buff[256];
    char *input = buff;
    float nms=.4;
    while(1){
        //这里的逻辑就是先判断是否提供了测试数据的路径。。如果前面没有提供。。这里可以再一次根据提示提供测试数据的路径
        if(filename){
            //strncpy实现字符赋值。。从filename中将字符串复制到input。。。
            //filename表示测试图片的路径
            strncpy(input, filename, 256);
        } else {

            printf("Enter Image Path: ");
            //刷新标准输出缓冲区
            fflush(stdout);
            //从标准输入读取一行到input
            input = fgets(input, 256, stdin);
            if(!input) return;
            //strtok分解字符串。。这里就是用换行符"\n"分解input，返回被分解的第一个字符串。。。
            strtok(input, "\n");
        }
        //通过load_image_color加载具体的图像数据。。
        //具体实现参考src/image.c
        //因为这里的传入参数是inout,0,0所以不会触发load_image_color里面内置的resize操做。。具体细节参考相关实现
        image im = load_image_color(input,0,0);
        //将读取到的图片resize到网络的输入尺寸
        //具体实现参考src/image.c
        //image结构体存储了一张图片的像素信息。。具体声明参考include/darknet.h
        image sized = resize_image(im, net->w, net->h);
        //取到读取并resize之后的像素信息，赋值给这里的X作为神经网络的输入。。。
        float *X = sized.data;
        //计时开始。。。。
        time=clock();
        //这里就是具体的推理过程了
        //具体的细节参考src/network.c
        network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));

        int nboxes = 0;
        //推理完成之后。。获取相关的检测框。。。这里是网络预测的全部的检测框。。还没有经过NMS筛选。。。
        //get_network_boxes具体实现参考src/network.c
        //detection结构体存储了检测框的相关内容。。具体细节参考include/darknet.h
        //此处得到的坐标已经被修正到了原图片尺度下。。。
        detection *dets = get_network_boxes(net, 1, 1, thresh, 0, 0, 0, &nboxes);
        //根据相关参数判断是都进行nms
        //do_nms_sort的具体实现参考src/box.c
        //这里的l.side指的是划分的cell个数。。划分的cell个数是l.side*l.side。。关于l.side的赋值。。l.side指的是yolo系列中的网格划分
        //比如说把一张图划分成7x7个cell等这时候的l.side就是7
        if (nms) do_nms_sort(dets, l.side*l.side*l.n, l.classes, nms);
        //把框画到图片上去。。。。具体实现参考src/image.c。。。细枝末节。。。后面有时间再补上。。。*********
        draw_detections(im, dets, l.side*l.side*l.n, thresh, voc_names, alphabet, 20);
        save_image(im, "predictions");
        show_image(im, "predictions", 0);
        free_detections(dets, nboxes);
        free_image(im);
        free_image(sized);
        if (filename) break;
    }
}
//命令行的yolo模式。。。。
void run_yolo(int argc, char **argv)
{
    //find_char_arg。。找到输入命令的相关参数。。并取出。。取出后会把这个参数以及数值从输入中删掉
    //具体实现参考src/utils.c。。。。。
    //这几个find的操作是类似的。。 
    char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    float thresh = find_float_arg(argc, argv, "-thresh", .2);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
    //检查最少的参数个数是否正确
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }
    
    int avg = find_int_arg(argc, argv, "-avg", 1);
    char *cfg = argv[3];
    //这里还是再一次判断参数量对不对。。然后选择给weights文件赋值。。。
    //filename代表的是测试图片的路径。。
    char *weights = (argc > 4) ? argv[4] : 0;
    char *filename = (argc > 5) ? argv[5]: 0;
    //根据相应的不同参数进入不同的模式。。。
    //test_yolo是进行yolo的推理测试。。
    //test_yolo具体实现参考examples/yolo.c
    if(0==strcmp(argv[2], "test")) test_yolo(cfg, weights, filename, thresh);
    //根据参数train进入训练模式。。train_yolo具体参考example/yolo.c
    else if(0==strcmp(argv[2], "train")) train_yolo(cfg, weights);
    else if(0==strcmp(argv[2], "valid")) validate_yolo(cfg, weights);
    else if(0==strcmp(argv[2], "recall")) validate_yolo_recall(cfg, weights);
    else if(0==strcmp(argv[2], "demo")) demo(cfg, weights, thresh, cam_index, filename, voc_names, 20, frame_skip, prefix, avg, .5, 0,0,0,0);
}
