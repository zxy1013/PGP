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

#define up(x) ((int)(x)+(((double)((int)(x)))!=(double)(x))) 


int sigbsend(char *sendto,char *dest,int id,int length, int smlen, std::string loc){
	std::cout << "------ 正在登录邮箱服务器 ------" << std::endl;
	Sock sock;
	const char *host_id = "smtp.qq.com";
	int port = 587; // smtp协议专用
	if(sock.Connect(host_id,port) == false)
		return 1;
	sock.recv_socket();
	
	/* EHLO指令是必须首先发的，相当于和服务器说hello */
	sock.send_socket("EHLO *********\r\n"); // 邮箱用户名
	sock.recv_socket();
	
	/* 发送 auth login 指令，告诉服务器要登录邮箱 */
	sock.send_socket("auth login\r\n");
	sock.recv_socket();
	
	/* 发送经过了base64编码的用户名 */
	sock.send_socket("**************");
	sock.send_socket("\r\n");
	sock.recv_socket();
	
	/* 经过base64编码后的用户密码 */
	sock.send_socket("**************");
	sock.send_socket("\r\n");
	sock.recv_socket();
	
	std::cout << "------ 正在发送邮件 ------" << std::endl;
	/* 发送 mail from 指令，相当于填写发件人 */
	sock.send_socket("mail from:<");
	sock.send_socket("1253141170@qq.com");
	sock.send_socket(">");
	sock.send_socket("\r\n");
	sock.recv_socket();
		
	/* 发送 rcpt to 指令，填写收件人 */
	sock.send_socket("rcpt to:<");
	sock.send_socket(sendto);
	sock.send_socket(">");
	sock.send_socket("\r\n");
	sock.recv_socket();
		
	/* 发送data, 开始发送邮件的主题部分 */
	sock.send_socket("data\r\n");
	sock.recv_socket();
	
	/* 发送邮件主体部分，先是邮件主题（subject), 后面是邮件内容 */
	sock.send_socket("subject:");
	sock.send_socket("PGP mail\r\n");
	sock.send_socket("\r\n\r\n");
	
	// START:SIGN:自己的id:邮件长度:签名长度:更新idx:签名内容:END
	std::string mail = "START:SIGN:" + std::to_string(id) + ":" +  to_string(length) + ":" + to_string(smlen) + ":" + loc + ":"+ std::string(dest, smlen) + ":END\r\n";
    //printf("%d", mail.size());

	//for(int i=0; i< mail.size(); i++){
      //printf("%d ", mail[i]);
    //}
	//std::cout<<" "<<std::endl;
	//std::cout<<mail<<std::endl;
	sock.send_socket1(mail.c_str(),mail.size());
	
	sock.send_socket(".\r\n");
	sock.recv_socket();
		
	sock.send_socket("quit\r\n");
	sock.recv_socket();
	std::cout << "------ 邮件发送完毕 ------" << std::endl;
}


int send(char *sendto,char *dest){
	std::cout << "------ 正在登录邮箱服务器 ------" << std::endl;
	Sock sock;
	const char *host_id = "smtp.qq.com";
	int port = 587; // smtp协议专用
	if(sock.Connect(host_id,port) == false)
		return 1;
	sock.recv_socket();

	/* EHLO指令是必须首先发的，相当于和服务器说hello */
	sock.send_socket("EHLO 1253141170@qq.com\r\n"); // 邮箱用户名
	sock.recv_socket();

	/* 发送 auth login 指令，告诉服务器要登录邮箱 */
	sock.send_socket("auth login\r\n");
	sock.recv_socket();

	/* 发送经过了base64编码的用户名 */
	sock.send_socket("MTI1MzE0MTE3MEBxcS5jb20=");
	sock.send_socket("\r\n");
	sock.recv_socket();

	/* 经过base64编码后的用户密码 */
	sock.send_socket("b2dlemZ4Zmhjb2lkYmFjYQ==");
	sock.send_socket("\r\n");
	sock.recv_socket();
    
	std::cout << "------ 正在发送邮件 ------" << std::endl;
	/* 发送 mail from 指令，相当于填写发件人 */
	sock.send_socket("mail from:<");
	sock.send_socket("1253141170@qq.com");
	sock.send_socket(">");
	sock.send_socket("\r\n");
	sock.recv_socket();
		
	/* 发送 rcpt to 指令，填写收件人 */
	sock.send_socket("rcpt to:<");
	sock.send_socket(sendto);
	sock.send_socket(">");
	sock.send_socket("\r\n");
	sock.recv_socket();
		
	/* 发送data, 开始发送邮件的主题部分 */
	sock.send_socket("data\r\n");
	sock.recv_socket();
	
	/* 发送邮件主体部分，先是邮件主题（subject), 后面是邮件内容 */
	std::string scontent(dest);
	sock.send_socket("subject:");
	sock.send_socket("PGP mail\r\n");
	sock.send_socket("\r\n\r\n");
	std::string mail;
	mail = "START:DONOTHING:";
	mail += scontent;
	mail += ":END\r\n";
	sock.send_socket(mail.c_str());
	sock.send_socket(".\r\n");
	sock.recv_socket();
	
	sock.send_socket("quit\r\n");
	sock.recv_socket();
	
	std::cout << "------ 邮件发送完毕 ------" << std::endl;
}


int encsend(char *sendto,char *buff,int group,int len,std::string loc){
	std::cout << "------ 正在登录邮箱服务器 ------" << std::endl;
	Sock sock;
	const char *host_id = "smtp.qq.com";
	int port = 587; // smtp协议专用
	if(sock.Connect(host_id,port) == false)
		return 1;
	sock.recv_socket();

	/* EHLO指令是必须首先发的，相当于和服务器说hello */
	sock.send_socket("EHLO 1253141170@qq.com\r\n"); // 邮箱用户名
	sock.recv_socket();


	/* 发送 auth login 指令，告诉服务器要登录邮箱 */
	sock.send_socket("auth login\r\n");
	sock.recv_socket();
	
	/* 发送经过了base64编码的用户名 */
	sock.send_socket("MTI1MzE0MTE3MEBxcS5jb20=");
	sock.send_socket("\r\n");
	sock.recv_socket();
	
	/* 经过base64编码后的用户密码 */
	sock.send_socket("b2dlemZ4Zmhjb2lkYmFjYQ==");
	sock.send_socket("\r\n");
	sock.recv_socket();
	
	std::cout << "------ 正在发送邮件 ------" << std::endl;
	/* 发送 mail from 指令，相当于填写发件人 */
	sock.send_socket("mail from:<");
	sock.send_socket("1253141170@qq.com");
	sock.send_socket(">");
	sock.send_socket("\r\n");
	sock.recv_socket();
		
	/* 发送 rcpt to 指令，填写收件人 */
	sock.send_socket("rcpt to:<");
	sock.send_socket(sendto);
	sock.send_socket(">");
	sock.send_socket("\r\n");
	sock.recv_socket();
		
	/* 发送data, 开始发送邮件的主题部分 */
	sock.send_socket("data\r\n");
	sock.recv_socket();
	
	
	/* 发送邮件主体部分，先是邮件主题（subject), 后面是邮件内容 */
	sock.send_socket("subject:");
	sock.send_socket("PGP mail\r\n");
	sock.send_socket("\r\n\r\n");
	
	std::string mail = "START:ENC:" + std::to_string(group) + ":" + loc + ":"+ std::string(buff, len) + ":END\r\n";
    //printf("%d", mail.size());

	//for(int i=0; i< mail.size(); i++){
      //printf("%d ", mail[i]);
    //}
	//std::cout<<" "<<std::endl;
	//std::cout<<mail<<std::endl;
	sock.send_socket1(mail.c_str(),mail.size());
	
	sock.send_socket(".\r\n");
	sock.recv_socket();
		
	sock.send_socket("quit\r\n");
	sock.recv_socket();
	std::cout << "------ 邮件发送完毕 ------" << std::endl;
	
}


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
        std::cout << "•••••• KYBER密钥文件不存在，正在生成 ••••••" << std::endl;
		// 初始化生成key
        init1(id);
    }
	sprintf(fn_rsp, "PQCsignKAT_%d.rsp", id);
	if (access(fn_rsp, F_OK) != 0)
    {
        std::cout << "•••••• DILITHIUM密钥文件不存在，正在生成 ••••••" << std::endl;
		// 初始化生成key
        init11(id);
    }
	while(1){
		// 选择如何处理邮件内容
		std::cout << "\n\n•••••• 选择如何处理邮件: 0.退出系统  1.直接发送  2.加密  3.签名 •••••• " << std::endl;
		int op;
		std::cin >> op;
		if(op == 0){
			std::cout << "•••••• 系统正在退出 •••••• " << std::endl;
			break;
		}
		if(op == 1){
			// 选择发送邮件的数量
			int group;
			std::cout << "====== 输入需要发送邮件的数量 ======" << std::endl;
			std::cin >> group;
			// 输入邮件长度
			int length;
			std::cout << "====== 输入邮件字数 ======" << std::endl;
			std::cin >> length;
			// 输入邮件内容
			char mess[length*group];
			memset(mess, 0, length*group);
			std::cout << "====== 输入邮件内容，以回车隔开 ======" << std::endl;
			// 清除状态，吃掉\n
			char x = getchar();
			for(int i=0;i<group;i++){
				std::cin.getline(mess+i*length, length, '\n');
			}
			for(int i = 0;i <group;i++){
				// 接收收件人邮箱
				char sendto[30];
				std::cout << "====== 输入邮件接收者邮箱 ======" << std::endl;
				std::cin.getline(sendto,30);
				char dest[length] = {""};
				strncpy(dest, mess + i*length, length);
				// 传入接收者邮箱 数据
				send(sendto,dest);
			}
		}else if(op == 2){
			// 输入邮件长度
			int length;
			std::cout << "====== 输入邮件字数 ======" << std::endl;
			std::cin >> length;
			// 输入邮件内容
			int group = up(length/KYBER_INDCPA_MSGBYTES);
			char mess[group * KYBER_INDCPA_MSGBYTES];
			memset(mess, 0, group * KYBER_INDCPA_MSGBYTES);
			std::cout << "====== 输入邮件内容 ======" << std::endl;
			// 清除状态，吃掉\n
			char x = getchar();
			std::cin.getline(mess, length, '\n');
			int idid;
			std::cout << "====== 输入接收方的身份id，用于加密 ======" << std::endl;
			int re = scanf("%d",&idid);
			char ct[KYBER_INDCPA_BYTES * group];
			memset(ct, 0, KYBER_INDCPA_BYTES * group);
			// 传入公钥文件
			re = enc1(group, (unsigned char *)mess, KYBER_INDCPA_MSGBYTES,  (unsigned char *)ct, idid);
			/*
			// 查看ct的值
			for(int i=0;i< KYBER_INDCPA_BYTES * group;i++){
				printf("%d ",ct[i]);
			}
			*/
			std::string loc;
			// 找 0
			for(int i=0;i< sizeof(ct) / sizeof(char);i++){
				if (ct[i] == 0){
					ct[i] = 1;
					loc += std::to_string(i);
					loc += ",";
				}
			}
			loc += "#,";
			// 找 10
			for(int i=0;i< sizeof(ct) / sizeof(char);i++){
				if (ct[i] == 10){
					ct[i] = 1;
					loc += std::to_string(i);
					loc += ",";
				}
			}
			loc += "#,";
			// 找 13
			for(int i=0;i< sizeof(ct) / sizeof(char);i++){
				if (ct[i] == 13){
					ct[i] = 1;
					loc += std::to_string(i);
					loc += ",";
				}
			}
			// std::cout << loc << std::endl;
			// 接收收件人邮箱
			char sendto[30];
			// 清除状态，吃掉\n
			x = getchar();
			std::cout << "====== 输入邮件接收者邮箱 ======" << std::endl;
			std::cin.getline(sendto,30);
			// 传入接收者邮箱 数据
			encsend(sendto,ct,group,sizeof(ct) / sizeof(char),loc);
			
		}else if(op == 3){
			// 选择发送邮件的数量
			int group;
			std::cout << "====== 输入需要发送邮件的数量 ======" << std::endl;
			std::cin>>group;
			// 输入邮件长度
			int length;
			std::cout << "====== 输入邮件字数 ======" << std::endl;
			std::cin>>length;
			// 输入邮件内容
			char mess[length*group];
			memset(mess, 0, length*group);
			std::cout << "====== 输入邮件内容，以回车隔开 ======" << std::endl;
			// 清除状态，吃掉\n
			char x = getchar();
			for(int i=0;i<group;i++){
				std::cin.getline(mess+i*length, length, '\n');
			}
			// 批处理模式
			if (group >= 2){
				std::cout << "====== 使用批处理模式处理邮件 ======" << std::endl;
				size_t smlen;
				char sig[(length + CRYPTO_BYTES)*MAXGROUP];
				memset(sig, 0, (length + CRYPTO_BYTES)*MAXGROUP);
				// 传入私钥文件 
				sign(group, (unsigned char *)mess, length, (uint8_t *)sig, &smlen, id);
				for(int k = 0;k <group;k++){
					// 接收收件人邮箱
					char sendto[30];
					std::cout << "====== 输入邮件接收者邮箱 ======" << std::endl;
					std::cin.getline(sendto,30);
					char dest[length + CRYPTO_BYTES] = {""};
					// 拷贝签名
					for(int i=0;i<smlen;i++){
						dest[i] = sig[i + k*(smlen)];
					}
					// 查看sig的值
					//for(int i=0;i< smlen;i++){
						//printf("%d ",dest[i]);
					//}
					//printf("\n\n ");
					// 修正数据
					std::string loc;
					// 找 0
					for(int i=0;i< length + CRYPTO_BYTES;i++){
						if (dest[i] == 0){
							dest[i] = 1;
							loc += std::to_string(i);
							loc += ",";
						}
					}
					loc += "#,";
					// 找 10
					for(int i=0;i< length + CRYPTO_BYTES;i++){
						if (dest[i] == 10){
							dest[i] = 1;
							loc += std::to_string(i);
							loc += ",";
						}
					}
					loc += "#,";
					// 找 13
					for(int i=0;i< length + CRYPTO_BYTES;i++){
						if (dest[i] == 13){
							dest[i] = 1;
							loc += std::to_string(i);
							loc += ",";
						}
					}
					
					// 传入接收者邮箱 数据 自己的id 数据长度
					sigbsend(sendto,dest,id,length,smlen,loc);
				}
			}else if(group == 1){
				// 单处理模式
				std::cout << "====== 使用单处理模式处理邮件 ======" << std::endl;
				// 接收收件人邮箱
				char sendto[30];
				std::cout << "====== 输入邮件接收者邮箱 ======" << std::endl;
				std::cin.getline(sendto,30);
				char sig[length + CRYPTO_BYTES];
				size_t smlen;
				// 传入公钥文件
				sign1((unsigned char *)mess, length, (uint8_t *)sig, &smlen, id);
				// 输出数据
				// 查看sig的值
				//for(int i=0;i< length + CRYPTO_BYTES;i++){
					//printf("%d ",sig[i]);
				//}
				// 修正数据
				std::string loc;
				// 找 0
				for(int i=0;i< sizeof(sig) / sizeof(char);i++){
					if (sig[i] == 0){
						sig[i] = 1;
						loc += std::to_string(i);
						loc += ",";
					}
				}
				loc += "#,";
				// 找 10
				for(int i=0;i< sizeof(sig) / sizeof(char);i++){
					if (sig[i] == 10){
						sig[i] = 1;
						loc += std::to_string(i);
						loc += ",";
					}
				}
				loc += "#,";
				// 找 13
				for(int i=0;i< sizeof(sig) / sizeof(char);i++){
					if (sig[i] == 13){
						sig[i] = 1;
						loc += std::to_string(i);
						loc += ",";
					}
				}
				// 传入接收者邮箱 数据 自己的id 数据长度，签名长度
				sigbsend(sendto,sig,id,length,smlen,loc);
			}
		}
	}
}

