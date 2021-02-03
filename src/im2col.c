#include "im2col.h"
#include <stdio.h>
//根据相应的索引提取原特征图对应位置的像素
float im2col_get_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    //因为这里传入的row,col都是填充pad之后的位置，对应回原图要先减去pad
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
        //返回相应位置的像素值
    return im[col + width*(row + height*channel)];
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
//im2col_cpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
//实现的是将输入的多尺度特征图转换成矩阵，便于卷积计算
void im2col_cpu(float* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col) 
{
    int c,h,w;
    //这里就是计算卷积层输出的特征图尺度，也就是卷积计算之后输出的w,h的尺度
    //实际上也是用这两个参数来控制卷积核跟图片数据展开的维度
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;
    //假如groups=1
    //这里的channels_col实际上是一个卷积核的参数量，多通道的卷积核是跟输入特征图拥有一样的channel
    //如果要将一个多维度卷积核展开成一个矩阵A的一行，这就是A的列数
    int channels_col = channels * ksize * ksize;
    //这里的循环对应这最终展开的图像矩阵的行数，两个矩阵点积，卷积核展开的列数恰好对应相应图片展开的行数
    //而卷积核矩阵展开的行数就是卷积核的个数
    for (c = 0; c < channels_col; ++c) {
        //因为卷积核展开是按照一行一行展开的，从而c%ksize得到的是当前数据是在展开前卷积核的第几列。。同时应注意
        //这里的卷积核是多维度的，为了便于理解，这里实际上是c % kernel_width，就是说比如一个二维卷积核，一行一行展开成一个一位数组
        //通过这样的操作就能定位当前数据是原二维卷积核的第几列
        int w_offset = c % ksize;
        //h_offset定位的就是多少行了，同样的二维卷积核一行一行展开成一位数组的时候，这里可以看成是(c/kernel_height)%kernel_width
        int h_offset = (c / ksize) % ksize;
        //c_im定位的是通道数，就是当前的卷积参数在卷积核的第几个通道上，通过int类型直接取整了
        //同时通过c_im，w_offset和h_offset就能够定位卷积核展开之后的每一个参数对应在原来卷积核中的位置
        int c_im = c / ksize / ksize;
        //接着这里的height_col跟width_col就是控制展开图像矩阵的列数了，实际上最终展开的列数就是height_col*width_col
        //这里之所以是height_col*width_col是因为图像对应位置的数据也跟卷积核展开的逻辑一样，对应位置的数据一行一行展开成一个一维数组
        //一个卷积核卷积占用的位置上的数据形成最终展开矩阵的一列，而height_col*width_col恰好就是整个特征图上卷积的次数，其实就是卷积输出的元素个数

        //所以这里也可以看成是通过输出特征图反推原图片上的数据位置
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                //h*stride就是当前卷积相应的行偏移量，h_offset表示当前卷积参数是卷积核的第几行
                int im_row = h_offset + h * stride;
                //同样的w*stride表示当前卷积相应的列偏移量，w_offset表示当前卷积参数是卷积核的第几列
                int im_col = w_offset + w * stride;
                //col_index就是最终展开的图像矩阵存储在一个一维数组中的索引了，最终展开的图像矩阵channels_col*（height_col*width_col）同样是一行一行存储成了一个一维数组
                //c*height_col*width_col+h*width_col+w,代表的是当前卷积核里面某一个参数，卷积时对应的数据在展开以后的图像矩阵中的索引，这里展开的图像矩阵，也是一行一行存储的
                //c*height_col*width_col控制的就是具体展开矩阵的第几行前面的参数量，然后h*width_col+w控制在这一行的第几列，也就是具体在这一行的第几个数，毕竟卷积操作会存在
                //多次移动卷积核，h*width_col+w定位的就是第几次卷积核的移动。。。
                int col_index = (c * height_col + h) * width_col + w;
                //然后根据索引，从原图片中取出相应像素值，写入最终转置数据存储的相应位置
                //im2col_get_pixel的具体实现参考src/im2col.c
                //这里的pad是原图填充0的长度，c_im定位的是相应图象数据的通道
                //im_row,im_col都是填充pad之后相应的位置
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
            }
        }
    }
}

