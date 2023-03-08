/*
nvcc -arch=sm_61 -rdc=true -cudart static --machine 64 -use_fast_math -O1 fips2022.cu main2.cu ntt2.cu packing2.cu poly2.cu polyvec2.cu reduce2.cu rng2.cu rounding2.cu sign2.cu symmetric-shake2.cu -o dilithium -lcudadevrt -std=c++11

nvprof ./dilithium
*/

/*

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <iostream>
#include "params2.h"
#include "main2.h"


int main()
{
        int re;
        // 输入邮件组数、长度、正文以及id信息
        int group;
        std::cout << "Enter the number of mail groups" << std::endl;
        std::cin>>group;
        int length;
        std::cout << "Enter message length, Make sure the number of words in the message is less than " << MESSAGELEN  << std::endl;
        std::cin>>length;
        char mess[length*group]; // 邮件
        std::cout << "Enter message body" << std::endl;
        // 清除状态，吃掉\n
        char x = getchar();
        for(int i=0;i<group;i++){
                std::cin.getline(mess+i*length, length, '\n');
        }
        int idid;
        std::cout << "Enter identity information" << std::endl;
        re = scanf("%d",&idid);
        char sig[(length + CRYPTO_BYTES)*MAXGROUP];
        size_t smlen;
        // 传入公钥文件
        re = sign(group, (unsigned char *)mess, length, (uint8_t *)sig, &smlen, idid);

        // 验证
        // 输入签名 签名长度 邮件长度 id信息
        int group1;
        std::cout << "Enter the number of mail groups" << std::endl;
        std::cin>>group1;
        int ididid;
        std::cout << "Enter identity information" << std::endl;
        re = scanf("%d",&ididid);
        char sig1[(length + CRYPTO_BYTES)*MAXGROUP];
        // 拷贝签名
        for(int i = 0;i < smlen*group1;i++){
           sig1[i] = sig[i];
        }
		// 0成功
        re = verify(group1,(uint8_t *)sig1, smlen, length, ididid);
		if(re == 0){
			std::cout << "success" << std::endl;
		}
        return re;
}

*/