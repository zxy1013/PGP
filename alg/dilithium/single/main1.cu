/*
nvcc -arch=sm_61 -rdc=true -cudart static --machine 64 -use_fast_math -O1 fips2021.cu main1.cu ntt1.cu packing1.cu poly1.cu polyvec1.cu reduce1.cu rng1.cu rounding1.cu sign1.cu symmetric-shake1.cu -o dilithium -lcudadevrt -std=c++11
nvprof ./dilithium
*/

/*
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <iostream>
#include "params1.h"
#include "main1.h"



int main()
{
        // 初始化生成key
        std::cout << "Enter identity information: " << std::endl;
        int id;
        int re;
        std::cin>>id;
        re = init11(id);
        

        // 输入邮件长度、正文以及id信息
        int length;
        std::cout << "Enter message length, Make sure the number of words in the message is less than " << MESSAGELEN << std::endl;
        std::cin>>length;
        char mess[length]; // 邮件
        std::cout << "Enter message body" << std::endl;
        // 清除状态，吃掉\n
        char x = getchar();
        std::cin.getline(mess, length, '\n');
        int idid;
        std::cout << "Enter identity information" << std::endl;
        std::cin>>idid;
        char sig[length + CRYPTO_BYTES];
        size_t smlen;
        // 传入公钥文件
        re = sign1((unsigned char *)mess, length, (uint8_t *)sig, &smlen, idid);

        

        // 输入签名 签名长度 邮件长度 id信息
        int ididid;
        std::cout << "Enter identity information" << std::endl;
        std::cin>>ididid;
        char sig1[length + CRYPTO_BYTES];
        // 拷贝签名
        for(int i = 0;i < smlen;i++){
           sig1[i] = sig[i];
        }
        re = verify13((uint8_t *)sig1, smlen, length, ididid);
		if(re == 0){
			std::cout << "Success" << std::endl;
		}
        return re;
        
}
*/