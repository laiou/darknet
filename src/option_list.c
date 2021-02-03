#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "option_list.h"
#include "utils.h"

list *read_data_cfg(char *filename)
{
    FILE *file = fopen(filename, "r");
    if(file == 0) file_error(filename);
    char *line;
    int nu = 0;
    list *options = make_list();
    while((line=fgetl(file)) != 0){
        ++ nu;
        strip(line);
        switch(line[0]){
            case '\0':
            case '#':
            case ';':
                free(line);
                break;
            default:
                if(!read_option(line, options)){
                    fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
                    free(line);
                }
                break;
        }
    }
    fclose(file);
    return options;
}

metadata get_metadata(char *file)
{
    metadata m = {0};
    list *options = read_data_cfg(file);

    char *name_list = option_find_str(options, "names", 0);
    if(!name_list) name_list = option_find_str(options, "labels", 0);
    if(!name_list) {
        fprintf(stderr, "No names or labels found\n");
    } else {
        m.names = get_labels(name_list);
    }
    m.classes = option_find_int(options, "classes", 2);
    free_list(options);
    return m;
}
//读取具体的参数信息，也就是读取cfg文件中的batch，filter等等信息
int read_option(char *s, list *options)
{
    size_t i;
    size_t len = strlen(s);
    char *val = 0;
    //取到"="后面的值的指针赋给val
    for(i = 0; i < len; ++i){
        if(s[i] == '='){
            //这里的"\0"将原来的s一分为二...
            s[i] = '\0';
            val = s+i+1;
            break;
        }
    }
    //再次判断i，是否正确，，假如"="是最后一个。则没有数值，配置文件书写错误，假如没有读取到"=",则读取错误
    if(i == len-1) return 0;
    char *key = s;
    //将具体的信息写进对应的options中，option_insert具体实现参考option_list.c
    option_insert(options, key, val);
    return 1;
}
//将读取的具体的配置比如说batch，filter等写进对应的list结构
void option_insert(list *l, char *key, char *val)
{
    //kvp结构体，用来存储具体的参数信息，具体声明参考src/option_list.h
    kvp *p = malloc(sizeof(kvp));
    p->key = key;
    p->val = val;
    p->used = 0;
    //将节点插入链表中
    list_insert(l, p);
}

void option_unused(list *l)
{
    node *n = l->front;
    while(n){
        kvp *p = (kvp *)n->val;
        if(!p->used){
            fprintf(stderr, "Unused field: '%s = %s'\n", p->key, p->val);
        }
        n = n->next;
    }
}
//根据相应的key从list取出对应的value。。。
char *option_find(list *l, char *key)
{
    node *n = l->front;
    while(n){
        kvp *p = (kvp *)n->val;
        if(strcmp(p->key, key) == 0){
            p->used = 1;
            return p->val;
        }
        n = n->next;
    }
    return 0;
}

char *option_find_str(list *l, char *key, char *def)
{
    char *v = option_find(l, key);
    if(v) return v;
    if(def) fprintf(stderr, "%s: Using default '%s'\n", key, def);
    return def;
}
//从保存参数的list中根据关键字找到对应的键值，进行赋值
int option_find_int(list *l, char *key, int def)
{ 
    //从链表中根据关键字找到对应的值
    //具体实现参考src/option_list.c
    char *v = option_find(l, key);
    //将取到的字符串转换成int
    if(v) return atoi(v);
    fprintf(stderr, "%s: Using default '%d'\n", key, def);
    return def;
}

int option_find_int_quiet(list *l, char *key, int def)
{
    char *v = option_find(l, key);
    if(v) return atoi(v);
    return def;
}

float option_find_float_quiet(list *l, char *key, float def)
{
    char *v = option_find(l, key);
    if(v) return atof(v);
    return def;
}

float option_find_float(list *l, char *key, float def)
{
    char *v = option_find(l, key);
    if(v) return atof(v);
    fprintf(stderr, "%s: Using default '%lf'\n", key, def);
    return def;
}