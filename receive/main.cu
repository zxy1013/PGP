/*
nvcc -arch=sm_61 -rdc=true -cudart static --machine 64 -use_fast_math -O1 main.cu sock.cu ../alg/dilithium/batch/fips2022.cu ../alg/dilithium/batch/ntt2.cu ../alg/dilithium/batch/packing2.cu ../alg/dilithium/batch/poly2.cu ../alg/dilithium/batch/polyvec2.cu ../alg/dilithium/batch/reduce2.cu ../alg/dilithium/batch/rng2.cu ../alg/dilithium/batch/rounding2.cu ../alg/dilithium/batch/sign2.cu ../alg/dilithium/batch/symmetric-shake2.cu ../alg/kyber/cbd3.cu ../alg/kyber/fips2023.cu ../alg/kyber/indcpa3.cu ../alg/kyber/ntt3.cu ../alg/kyber/poly3.cu ../alg/kyber/polyvec3.cu ../alg/kyber/reduce3.cu ../alg/kyber/rng3.cu ../alg/kyber/symmetric-fips2023.cu ../alg/kyber/verify3.cu ../alg/dilithium/single/fips2021.cu ../alg/dilithium/single/ntt1.cu ../alg/dilithium/single/packing1.cu ../alg/dilithium/single/poly1.cu ../alg/dilithium/single/polyvec1.cu ../alg/dilithium/single/reduce1.cu ../alg/dilithium/single/rng1.cu ../alg/dilithium/single/rounding1.cu ../alg/dilithium/single/sign1.cu ../alg/dilithium/single/symmetric-shake1.cu -o main -lcudadevrt -std=c++11 

nvprof ./main
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <openssl/bio.h>
#include <openssl/evp.h>
#include "sock.h"
#include "../alg/dilithium/batch/main2.h"
#include "../alg/dilithium/batch/params2.h"
#include "../alg/dilithium/single/main1.h"
#include "../alg/dilithium/single/params1.h"
#include "../alg/kyber/main3.h"
#include "../alg/kyber/params3.h"


int main()
{
	// 接收身份信息
	int id;
	std::cout << "•••••• 请输入您的身份id，用于查询密钥信息 ••••••" << std::endl;
	std::cin >> id;
	char fn_rsp[32];
	FILE *fp_rsp;
	
	// 查询密钥文件是否存在
	sprintf(fn_rsp, "PQCENCKAT_%d.rsp", id);
	if (access(fn_rsp, F_OK) != 0)
    {
        std::cout << "•••••• KYBER密钥文件不存在，正在生成 •••••• " << std::endl;
		// 初始化生成key
        init1(id);
    }
	sprintf(fn_rsp, "PQCsignKAT_%d.rsp", id);
	if (access(fn_rsp, F_OK) != 0)
    {
        std::cout << "•••••• DILITHIUM密钥文件不存在，正在生成 •••••• " << std::endl;
		// 初始化生成key
        init11(id);
    }
	
	std::cout << "•••••• 登录邮件服务器 •••••• " << std::endl;
	Sock sock;
	const char *host_id = "pop3.126.com";
	int port = 110;
	if(sock.Connect(host_id, port) == false)
	   return 1;
	sock.recv_socket();
	
	sock.send_socket("user ***************\r\n"); // 输入邮箱用户名
	sock.recv_socket();
	
	sock.send_socket("pass ***************\r\n"); // 邮箱密码
	sock.recv_socket();
	
	std::cout << "•••••• 读取接收到的邮件列表 •••••• " << std::endl;
	sock.send_socket("list\r\n");
	int num = sock.recv_socket();		       
	std::cout << sock.get_recvbuf() << std::endl;
	std::cout << "•••••• 进入系统 ••••••" << std::endl;

	while(1){
		sleep(1);
		std::cout << "\n\n•••••• 输入想要处理的邮件号，0为终止系统 ••••••" << std::endl;
		int index;
		num = scanf("%d",&index);
		if(index == 0){
			break;
		}
		std::string retrieve;
		retrieve = "retr " + std::to_string(index) + "\r\n";
		std::cout << "====== 正在读取第" << index << "封邮件 ======" << std::endl;
		sock.send_socket(retrieve.c_str());
		sock.recv_socket();
		std::string content = sock.get_recvbuf();
		
		// 拆分邮件内容
		int idx = content.find("START:");
		int endidx = content.find(":END");
		content = content.substr(idx+6,endidx-idx-6);
		//printf("%d: ", content.size());
		//for(int i=0; i< content.size(); i++){
			//printf("%d ", content[i]);
        //}
		std::cout << "------ 正在查询如何处理该邮件 ------" << std::endl;
		idx = content.find("DONOTHING:");
		if(idx != -1){
			// 该邮件不需要任何其他操作，直接进行读取
			std::cout << "------ 该邮件是明文形式，直接进行读取 ------" << std::endl;
			std::cout << "content: " << content.substr(10,content.length()-10) << std::endl;
			continue;
		}
		idx = content.find("SIGN:");
		if(idx != -1){
			// START:SIGN:自己的id:邮件长度:签名长度:更新idx:签名内容:END
			std::cout << "------ 该邮件需要进行验证 ------" << std::endl;
			content = content.substr(5,content.length()-5);
			idx = content.find(":");
			int idid = atoi(content.substr(0,idx).c_str());
			content = content.substr(idx+1,content.length()-idx-1);
			idx = content.find(":");
			size_t length = atoi(content.substr(0,idx).c_str());
			content = content.substr(idx+1,content.length()-idx-1);
			idx = content.find(":");
			size_t smlen = atoi(content.substr(0,idx).c_str());
			content = content.substr(idx+1,content.length()-idx-1);
			// 拆分index
			idx = content.find(":");
			std::string loc = content.substr(0,idx);
			// 拆分邮件内容
			content = content.substr(idx+1,content.length()-idx-1);
			char sig1[smlen];
			memset(sig1, 0, smlen);
			// 拷贝签名
			for(int i=0;i<smlen;i++){
				sig1[i] = content[i];
			}
			// 拆分loc
			std::vector<std::vector<int>> sub_sequences;
			std::vector<int> current_sub_sequence;
			std::stringstream ss(loc);
			std::string number_str;

			while (getline(ss, number_str, ',')) {
				if (number_str == "#") {
					sub_sequences.push_back(current_sub_sequence);
					current_sub_sequence.clear();
				} else {
					//std::cout << number_str << std::endl;
					current_sub_sequence.push_back(std::stoi(number_str));
				}
			}
			sub_sequences.push_back(current_sub_sequence);
			int val = 0;
			
			for (auto sub_sequence : sub_sequences) {
				if(val == 0){
					for (auto number : sub_sequence) {
						//printf("%d ",number);
						sig1[number] = 0;
					}
				}else if(val == 1){
					for (auto number : sub_sequence) {
						//printf("%d ",number);
						sig1[number] = 10;
					}
				}else if(val == 2){
					for (auto number : sub_sequence) {
						//printf("%d ",number);
						sig1[number] = 13;
					}
				}
				val ++;
			}
			
			// 查看sig的值
			//for(int i=0;i< smlen;i++){
				//printf("%d ",sig1[i]);
			//}
			
			int re = verify13((uint8_t *)sig1, smlen, length, idid);
			// 验证成功
			if (re == 0){
				std::cout << "------ 邮件验证成功 ------" << std::endl;
				printf("------ 邮件内容是：");
				for(int i = CRYPTO_BYTES;i<smlen;i++){
					printf("%c",sig1[i]);
				}
				printf(" ------");
			}
			std::cout << std::endl;
			continue;
		}
		idx = content.find("ENC:");
		if(idx != -1){
			std::cout << "------ 该邮件需要进行解密处理 ------" << std::endl;
			content = content.substr(4,content.length()-4);
			idx = content.find(":");
			int group1 = atoi(content.substr(0,idx).c_str());
			content = content.substr(idx+1,content.length()-idx-1);
			// 拆分index
			idx = content.find(":");
			std::string loc = content.substr(0,idx);
			// 拆分邮件内容
			content = content.substr(idx+1,content.length()-idx-1);
			char ct1[KYBER_INDCPA_BYTES * group1];
			// 拷贝密文
			for(int i=0;i<KYBER_INDCPA_BYTES * group1;i++){
				ct1[i] = content[i];
			}
			// 拆分loc
			std::vector<std::vector<int>> sub_sequences;
			std::vector<int> current_sub_sequence;
			std::stringstream ss(loc);
			std::string number_str;

			while (getline(ss, number_str, ',')) {
				if (number_str == "#") {
					sub_sequences.push_back(current_sub_sequence);
					current_sub_sequence.clear();
				} else {
					// std::cout << number_str << std::endl;
					current_sub_sequence.push_back(std::stoi(number_str));
				}
			}
			sub_sequences.push_back(current_sub_sequence);
			int val = 0;
			
			for (auto sub_sequence : sub_sequences) {
				if(val == 0){
					for (auto number : sub_sequence) {
						//printf("%d ",number);
						ct1[number] = 0;
					}
				}else if(val == 1){
					for (auto number : sub_sequence) {
						//printf("%d ",number);
						ct1[number] = 10;
					}
				}else if(val == 2){
					for (auto number : sub_sequence) {
						//printf("%d ",number);
						ct1[number] = 13;
					}
				}
				val ++;
			}
			/*
			// 查看ct的值
			for(int i=0;i< KYBER_INDCPA_BYTES * group1;i++){
				printf("%d ",ct1[i]);
			}
			*/
			// 解密
			char mess[KYBER_INDCPA_MSGBYTES*group1]; 
			memset(mess, 0, KYBER_INDCPA_MSGBYTES*group1);
			dec1(group1, (unsigned char *)mess, KYBER_INDCPA_MSGBYTES, (unsigned char *)ct1, id);
			printf("------ 解密后的邮件内容是：");
			for(int i = 0;i<KYBER_INDCPA_MSGBYTES*group1;i++){
				printf("%c",mess[i]);
			}
			printf(" ------\n");
			continue;
		}
	}
}



