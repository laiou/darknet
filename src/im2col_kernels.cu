#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "im2col.h"
#include "cuda.h"
}

// src: https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu
// You may also want to read: https://github.com/BVLC/caffe/blob/master/LICENSE

//完成相应卷积数据从多通道展开成一个2为矩阵的功能
//每个线程完成的是输入特征图某一个通道上参与某一次卷积计算的数据的展开
__global__ void im2col_gpu_kernel(const int n, const float* data_im,
        const int height, const int width, const int ksize,
        const int pad,
        const int stride,
        const int height_col, const int width_col,
        float *data_col) {
        //计算线程id
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    //这个for循环实际上要不要没啥差别?
    for(; index < n; index += blockDim.x*gridDim.x){
    //这里面内部完成了一个某个通道上一次卷积操作的数据的转换
    //w_out表示的是特征图上某一次卷积操作在w维度上的次数
        int w_out = index % width_col;
        int h_index = index / width_col;
        //h_out表示的是特征图上某一次卷积操作在h维度上的次数
        int h_out = h_index % height_col;
        //channel_in表示的是某一次卷积操作在输入特征图上的通道数
        int channel_in = h_index / height_col;
        //channel_out表示在输入特征图channel_in通道上的展开数据在输出特征的上的通道数
        //这个通道数怎么说呢。。其实输出就是一个二维矩阵。。强行通道有点尴尬。。。
        //实际上就是输出矩阵的行索引，这和行索引channel_out代表的是channel_in通道上的值展开后的起始行位置
        int channel_out = channel_in * ksize * ksize;
        //h_in和w_in表示像素值在输入特征图上w,和h维度的坐标
        int h_in = h_out * stride - pad;
        int w_in = w_out * stride - pad;
        //data_col_ptr指向存储结果的位置
        float* data_col_ptr = data_col;
        //定位相应的输出存储的位置
        data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
        //data_im_ptr指向输入特征图
        const float* data_im_ptr = data_im;
        //定位输入特征图上像素的位置，也就是某次卷积操作的数据的起始位置
        data_im_ptr += (channel_in * height + h_in) * width + w_in;
        //遍历一个卷积核尺寸的数据
        for (int i = 0; i < ksize; ++i) {
            for (int j = 0; j < ksize; ++j) {
                int h = h_in + i;
                int w = w_in + j;
                //完成数据的赋值
                *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
                    data_im_ptr[i * width + j] : 0;

                //*data_col_ptr = data_im_ptr[ii * width + jj];
                //存储结果的指针按照移动到下一个位置
                data_col_ptr += height_col * width_col;
            }
        }
    }
}
//im2col_gpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
//将参与卷积的计算数据从一个多通道数据展开成一个大矩阵。。实现的功能和src/im2col_cpu.c中的功能一致
void im2col_gpu(float *im,
         int channels, int height, int width,
         int ksize, int stride, int pad, float *data_col){
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    //这里的channels表示参与卷积的输入特征图的通道数
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * height_col * width_col;
    //可以理解成一共需要在输入特征图上进行channels*height_col*width_col次卷积
    //从这里能看出。。一个线程完成的是某一个通道上参与某一次卷积的数据的展开
    //这里是为了保证线程总数足够用，同时取整数做grid的划分参数
    //im2col_gpu_kernel的实现参考src/im2col_kernels.cu
    im2col_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK,
        BLOCK>>>(
                num_kernels, im, height, width, ksize, pad,
                stride, height_col,
                width_col, data_col);
}
