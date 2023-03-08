#ifndef TIMES2_H
#define TIMES2_H

#include <sys/time.h>
#include <time.h>

double get_time(void);

double get_time(void)
{
    struct timeval tv;
    double t;
    gettimeofday(&tv, (struct timezone *)0);
    //  tv_sec 秒 long tv_usec 微秒 
    t = tv.tv_sec*1000000 + tv.tv_usec;
    return t;
}
#endif