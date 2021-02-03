#ifndef OPTION_LIST_H
#define OPTION_LIST_H
#include "list.h"
//最终存储cfg配置的结构体，其中的used是一个标志位。用来标记这个结构体的数据是否已经被使用
//关于used，的作用可以参考src/option_list.c中的option_find()函数
typedef struct{
    char *key;
    char *val;
    int used;
} kvp;


int read_option(char *s, list *options);
void option_insert(list *l, char *key, char *val);
char *option_find(list *l, char *key);
float option_find_float(list *l, char *key, float def);
float option_find_float_quiet(list *l, char *key, float def);
void option_unused(list *l);

#endif
