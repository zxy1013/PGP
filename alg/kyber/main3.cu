/*
nvcc -arch=sm_61 -rdc=true -cudart static --machine 64 -use_fast_math -O1 cbd3.cu fips2023.cu indcpa3.cu main3.cu ntt3.cu poly3.cu polyvec3.cu reduce3.cu rng3.cu symmetric-fips2023.cu verify3.cu -o kyber -lcudadevrt -std=c++11

nvprof ./kyber
*/

/*
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <iostream>
#include "main3.h"
#define up(x) ((int)(x)+(((double)((int)(x)))!=(double)(x)))
int main()
{

        // 初始化生成key
        std::cout << "Enter identity information: " << std::endl;
        int id;
        int re;
        std::cin>>id;
        re = init1(id);

        // 输入邮件长度
		int length;
		std::cout << "输入邮件长度" << std::endl;
		std::cin>>length;
		// 输入邮件内容
		int group = up(length/KYBER_INDCPA_MSGBYTES);
		char mess[group * KYBER_INDCPA_MSGBYTES];
		memset(mess, 0, group * KYBER_INDCPA_MSGBYTES);
		std::cout << "输入邮件内容" << std::endl;
		// 清除状态，吃掉\n
		char x = getchar();
		std::cin.getline(mess, group * KYBER_INDCPA_MSGBYTES, '\n');
		int idid;
        std::cout << "Enter identity information" << std::endl;
        re = scanf("%d",&idid);
        char ct[KYBER_INDCPA_BYTES * group];
        // 传入公钥文件
        re = enc1(group, (unsigned char *)mess, KYBER_INDCPA_MSGBYTES,  (unsigned char *)ct, idid);
		// std::cout << ct << std::endl;

        // 解密
        // 输入密文组数 id信息
        int ididid;
        char mess1[KYBER_INDCPA_MSGBYTES*group]; // 邮件
        memset(mess1, 0, KYBER_INDCPA_MSGBYTES*group);
        std::cout << "Enter identity information" << std::endl;
        re = scanf("%d",&ididid);
        char ct1[KYBER_INDCPA_BYTES * group];
        // 拷贝签名
        for(int i = 0;i < KYBER_INDCPA_BYTES * group;i++){
           ct1[i] = ct[i];
        }
        re = dec1(group, (unsigned char *)mess1 ,  KYBER_INDCPA_MSGBYTES, (unsigned char *)ct1, ididid);
        for(int i = 0 ;i < KYBER_INDCPA_MSGBYTES*group;i++ ){
            printf("%c",mess1[i] );
        }
        return 0;
}

*/