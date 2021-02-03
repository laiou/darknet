#include <stdio.h>
#include <math.h>
void col2im_add_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad, float val)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return;
        //注意这里是累加，不是重写，也就是意味着一个个batch产生的l.delta都累加存了起来
        //够了一个batch_size之后进行参数更新
        //这里的+=是应为batch中有多张图，梯度的影响把多张图的作用放到一起，还有就是stride较小的话。。一个位置会参与多次计算
        //是有重复的梯度的
    im[col + width*(row + height*channel)] += val;
}
//This one might be too, can't remember.
//将有特征图展开的矩阵还原成特征图的形态
//类似于im2col.c中的反向操作。。。。
void col2im_cpu(float* data_col,
         int channels,  int height,  int width,
         int ksize,  int stride, int pad, float* data_im) 
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                double val = data_col[col_index];
                col2im_add_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad, val);
            }
        }
    }
}

