#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "crop_layer.h"
#include "utils.h"
#include "cuda.h"
#include "image.h"
}
//获取相应位置上的像素值
__device__ float get_pixel_kernel(float *image, int w, int h, int x, int y, int c)
{
    if(x < 0 || x >= w || y < 0 || y >= h) return 0;
    return image[x + w*(y + c*h)];
}

//rgb_to_hsv_kernel(rgb)
//将rgb空间的值转换到hsv空间
__device__ float3 rgb_to_hsv_kernel(float3 rgb)
{
    float r = rgb.x;
    float g = rgb.y; 
    float b = rgb.z;

    float h, s, v;
    //取rgb中的最大和最小值
    float max = (r > g) ? ( (r > b) ? r : b) : ( (g > b) ? g : b);
    float min = (r < g) ? ( (r < b) ? r : b) : ( (g < b) ? g : b);
    float delta = max - min;
    v = max;
    if(max == 0){
        s = 0;
        h = -1;
    }else{
        s = delta/max;
        if(r == max){
            h = (g - b) / delta;
        } else if (g == max) {
            h = 2 + (b - r) / delta;
        } else {
            h = 4 + (r - g) / delta;
        }
        if (h < 0) h += 6;
    }
    return make_float3(h, s, v);
}

//hsv_to_rgb_kernel(hsv)
//将hsv空间的值转换成rgb
__device__ float3 hsv_to_rgb_kernel(float3 hsv)
{
    float h = hsv.x;
    float s = hsv.y; 
    float v = hsv.z;

    float r, g, b;
    float f, p, q, t;

    if (s == 0) {
        r = g = b = v;
    } else {
        int index = (int) floorf(h);
        f = h - index;
        p = v*(1-s);
        q = v*(1-s*f);
        t = v*(1-s*(1-f));
        if(index == 0){
            r = v; g = t; b = p;
        } else if(index == 1){
            r = q; g = v; b = p;
        } else if(index == 2){
            r = p; g = v; b = t;
        } else if(index == 3){
            r = p; g = q; b = v;
        } else if(index == 4){
            r = t; g = p; b = v;
        } else {
            r = v; g = p; b = q;
        }
    }
    r = (r < 0) ? 0 : ((r > 1) ? 1 : r);
    g = (g < 0) ? 0 : ((g > 1) ? 1 : g);
    b = (b < 0) ? 0 : ((b > 1) ? 1 : b);
    return make_float3(r, g, b);
}

//双线性插值的实现，具体参考双线性插值的计算方式
__device__ float bilinear_interpolate_kernel(float *image, int w, int h, float x, float y, int c)
{
    //floorf表示向下取整
    int ix = (int) floorf(x);
    int iy = (int) floorf(y);

    float dx = x - ix;
    float dy = y - iy;
    //get_pixel_kernel的具体实现参考src/crop_layer_kernels.cu，获取相应位置上的像素值
    float val = (1-dy) * (1-dx) * get_pixel_kernel(image, w, h, ix, iy, c) + 
        dy     * (1-dx) * get_pixel_kernel(image, w, h, ix, iy+1, c) + 
        (1-dy) *   dx   * get_pixel_kernel(image, w, h, ix+1, iy, c) +
        dy     *   dx   * get_pixel_kernel(image, w, h, ix+1, iy+1, c);
    return val;
}

//levels_image_kernel<<<cuda_gridsize(size), BLOCK>>>(net.input_gpu, layer.rand_gpu, layer.batch, layer.w, layer.h, net.train, layer.saturation, layer.exposure, translate, scale, layer.shift)
//实现的是对图像上的色值进行变换，实现相关的预处理功能
__global__ void levels_image_kernel(float *image, float *rand, int batch, int w, int h, int train, float saturation, float exposure, float translate, float scale, float shift)
{
    //size表示最大的block索引，实际上的总数据量应该是batch*w*h*c
    int size = batch * w * h;
    //计算相应的线程id
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    //如果线程id超过了size表明数据处理完毕，这里为什么是id>=size而不是id>=size*c
    //结合后面的实现来看。。准确的来讲就是这里的grid被划分成b*w*h个block，每个block固定512个线程
    //id >=size保证的是最大的线程id就是size，如果要处理完全部数据的话，从这里来看每个线程就要处理c个数值
    if(id >= size) return;
    //某个通道上w维度的偏移
    int x = id % w;
    //这里把id更新成了处理过的w维度数据的个数
    id /= w;
    //得到某个通道上h维度的偏移
    int y = id % h;
    //再次把id更新成处理过的w*h维度的个数。。可以看成处理过的通道数来理解
    //实际上这里的id就表示处理到第几张图片了,因为通过上面的id>=size等来限制之后假如要处理完全部的数据，只能是一个线程处理c个值
    //通过最大的线程id来推理的话，这里的id就表示处理到了第几张图
    id /= h;
    //利用随机数来进行预处理的相关参数
    float rshift = rand[0];
    float gshift = rand[1];
    float bshift = rand[2];
    float r0 = rand[8*id + 0];
    float r1 = rand[8*id + 1];
    float r2 = rand[8*id + 2];
    float r3 = rand[8*id + 3];

    saturation = r0*(saturation - 1) + 1;
    saturation = (r1 > .5f) ? 1.f/saturation : saturation;
    exposure = r2*(exposure - 1) + 1;
    exposure = (r3 > .5f) ? 1.f/exposure : exposure;

    //这里就表示处理过的图片数量的偏移
    size_t offset = id * h * w * 3;
    //将image定位到相应位置
    image += offset;
    //rgb图片数据的三通道起始，也就是实际上一个线程处理3个值
    //实际上这里就取到了某张图片某个位置上的3个值，应为是rgb三通道的图片
    float r = image[x + w*(y + h*0)];
    float g = image[x + w*(y + h*1)];
    float b = image[x + w*(y + h*2)];、
    //make_float3实现的是将rgb三个值拼成一个向量或者说这里的r,g,b变成rgb.r,rgb.g,rgb.b
    float3 rgb = make_float3(r,g,b);
    //如果是在训练的话
    if(train){
        //将rgb转换成hsv,rgb_to_hsv_kernel的具体实现参考src/crop_layer_kernels.cu
        float3 hsv = rgb_to_hsv_kernel(rgb);
        //进行相应的预处理
        hsv.y *= saturation;
        hsv.z *= exposure;
        //hsv_to_rgb_kernel的具体实现参考src/crop_layer_kernels.cu
        //将hsv空间的值转换成rgb空间
        rgb = hsv_to_rgb_kernel(hsv);
    } else {
        shift = 0;
    }
    //对相应的三个值进行预处理
    image[x + w*(y + h*0)] = rgb.x*scale + translate + (rshift - .5f)*shift;
    image[x + w*(y + h*1)] = rgb.y*scale + translate + (gshift - .5f)*shift;
    image[x + w*(y + h*2)] = rgb.z*scale + translate + (bshift - .5f)*shift;
}

//forward_crop_layer_kernel<<<cuda_gridsize(size), BLOCK>>>(net.input_gpu, layer.rand_gpu, size, layer.c, layer.h, layer.w, layer.out_h, layer.out_w, net.train, layer.flip, radians, layer.output_gpu)
//实现crop的核函数
__global__ void forward_crop_layer_gpu_crop_layer_kernel(float *input, float *rand, int size, int c, int h, int w, int crop_height, int crop_width, int train, int flip, float angle, float *output)
{
    //计算相应的线程id
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    //id>=size表示数据处理完毕
    if(id >= size) return;

    float cx = w/2.f;
    float cy = h/2.f;
    //计数处理到哪一个位置了
    int count = id;
    //得到输出在w维度的偏移
    int j = id % crop_width;
    //将id更新成处理过的crop_width维度的个数
    id /= crop_width;
    //得到输出在h维度的偏移
    int i = id % crop_height;
    //将id更新成处理的通道数量
    id /= crop_height;
    //k表示某张图片的第几个通道
    int k = id % c;
    //将id更新成处理到了第几张图片
    id /= c;
    int b = id;
    //取随机值
    float r4 = rand[8*b + 4];
    float r5 = rand[8*b + 5];
    float r6 = rand[8*b + 6];
    float r7 = rand[8*b + 7];

    //进行dw和dh的初始化
    float dw = (w - crop_width)*r4;
    float dh = (h - crop_height)*r5;
    flip = (flip && (r6 > .5f));
    angle = 2*angle*r7 - angle;
    //如果是训练的话
    if(!train){
        dw = (w - crop_width)/2.f;
        dh = (h - crop_height)/2.f;
        flip = 0;
        angle = 0;
    }
    //定位相应图像的位置
    input += w*h*c*b;
    //如果filp的话，在w维度上进行翻转
    float x = (flip) ? w - dw - j - 1 : j + dw;    
    float y = i + dh;

    float rx = cosf(angle)*(x-cx) - sinf(angle)*(y-cy) + cx;
    float ry = sinf(angle)*(x-cx) + cosf(angle)*(y-cy) + cy;
    //得到最终的crop的输出，bilinear_interpolate_kernel的具体实现参考src/crop_layer_kernels.cu
    //双线性插值函数的实现，结合双线性插值的计算方式来看
    output[count] = bilinear_interpolate_kernel(input, w, h, rx, ry, k);
}
//gpu版本的crop层的前向传播
extern "C" void forward_crop_layer_gpu(crop_layer layer, network net)
{
    //cuda_random的具体实现参考src/cuda.c
    //cuda_random生成0.0到1.0之间的浮点值，存储到rand_gpu中
    cuda_random(layer.rand_gpu, layer.batch*8);
    //将相应的角度转换成弧度
    float radians = layer.angle*3.14159265f/180.f;

    float scale = 2;
    float translate = -1;
    if(layer.noadjust){
        scale = 1;
        translate = 0;
    }
    //这里的size表示输入的数据量，不统计channel。。只计算了batch和相应的单通道数据量
    //从后面的levels_image_kernel中能看到这里size是用于线程的划分。。。将grid划分成size个block
    int size = layer.batch * layer.w * layer.h;

    //levels_image_kernel的具体实现参考src/corp_layer_kernels.cu
    //实现的是图片每个通道上的色值变换，也就是相关的预处理操作
    levels_image_kernel<<<cuda_gridsize(size), BLOCK>>>(net.input_gpu, layer.rand_gpu, layer.batch, layer.w, layer.h, net.train, layer.saturation, layer.exposure, translate, scale, layer.shift); 
    check_error(cudaPeekAtLastError());

    //这里的size表示的是输出的数据量，同样用来划分grid
    //从这里的划分方式来看一个block负责计算一个输出值。。实际上也就只需要一个线程即可，实际过程并不一定是一个block只计算一个值
    //这里方便理解这个划分。。
    size = layer.batch*layer.c*layer.out_w*layer.out_h;

    //crop的操作，forward_crop_layer_kernel的具体实现参考src/crop_layer_kernels.cu
    //根据双线性插值进行crop和filp的操作
    forward_crop_layer_kernel<<<cuda_gridsize(size), BLOCK>>>(net.input_gpu, layer.rand_gpu, size, layer.c, layer.h, layer.w, layer.out_h, layer.out_w, net.train, layer.flip, radians, layer.output_gpu);
    check_error(cudaPeekAtLastError());

/*
       cuda_pull_array(layer.output_gpu, layer.output, size);
       image im = float_to_image(layer.crop_width, layer.crop_height, layer.c, layer.output + 0*(size/layer.batch));
       image im2 = float_to_image(layer.crop_width, layer.crop_height, layer.c, layer.output + 1*(size/layer.batch));
       image im3 = float_to_image(layer.crop_width, layer.crop_height, layer.c, layer.output + 2*(size/layer.batch));

       translate_image(im, -translate);
       scale_image(im, 1/scale);
       translate_image(im2, -translate);
       scale_image(im2, 1/scale);
       translate_image(im3, -translate);
       scale_image(im3, 1/scale);
       
       show_image(im, "cropped");
       show_image(im2, "cropped2");
       show_image(im3, "cropped3");
       cvWaitKey(0);
       */
}

