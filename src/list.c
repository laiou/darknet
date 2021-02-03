#include <stdlib.h>
#include <string.h>
#include "list.h"
//建立链表节点并初始化。。
list *make_list()
{
	list *l = malloc(sizeof(list));
	l->size = 0;
	l->front = 0;
	l->back = 0;
	return l;
}

/*
void transfer_node(list *s, list *d, node *n)
{
    node *prev, *next;
    prev = n->prev;
    next = n->next;
    if(prev) prev->next = next;
    if(next) next->prev = prev;
    --s->size;
    if(s->front == n) s->front = next;
    if(s->back == n) s->back = prev;
}
*/

void *list_pop(list *l){
    if(!l->back) return 0;
    node *b = l->back;
    void *val = b->val;
    l->back = b->prev;
    if(l->back) l->back->next = 0;
    free(b);
    --l->size;
    
    return val;
}
//将一个链表节点插入链表中
void list_insert(list *l, void *val)
{ 
	//创建一个node类型的节点作为中转，从而将数据存入
	//将section类型的结构体指针作为数据传输给node结构体中的val指针变量，完成赋值
	node *new = malloc(sizeof(node));
	new->val = val;
	new->next = 0;
    //更新list结构里面的表头，表尾，和相关的node节点的前后指针
	if(!l->back){
		l->front = new;
		new->prev = 0;
	}else{
		l->back->next = new;
		new->prev = l->back;
	}
	l->back = new;
	//这里的l指代的是option，从paser.c中的read_cfg跳转过来的话。。此时更新的是option的size
	//后面读取具体的参数也会调用这个函数，那个时候更新的size表示的是某一层的参数个数。。。
	++l->size;
}

void free_node(node *n)
{
	node *next;
	while(n) {
		next = n->next;
		free(n);
		n = next;
	}
}

void free_list(list *l)
{
	free_node(l->front);
	free(l);
}

void free_list_contents(list *l)
{
	node *n = l->front;
	while(n){
		free(n->val);
		n = n->next;
	}
}
//将list中每个node的val放到一个指针数组里面
void **list_to_array(list *l)
{
    void **a = calloc(l->size, sizeof(void*));
    int count = 0;
    node *n = l->front;
    while(n){
        a[count++] = n->val;
        n = n->next;
    }
    return a;
}
