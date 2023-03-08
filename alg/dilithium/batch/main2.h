/*
nvcc -arch=sm_61 -rdc=true -cudart static --machine 64 -use_fast_math -O1 fips2022.cu main2.cu ntt2.cu packing2.cu poly2.cu polyvec2.cu reduce2.cu rng2.cu rounding2.cu sign2.cu symmetric-shake2.cu -o dilithium -lcudadevrt -std=c++11

nvprof ./dilithium
*/
#ifndef __MAIN2_H__
#define __MAIN2_H__

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <iostream>
#include "params2.h"
#include "rng2.h"
#include "sign2.h"
#include "times2.h"
#include "fips2022.h"

#define MAX_MARKER_LEN          50
#define KAT_FILE_OPEN_ERROR -1
#define KAT_DATA_ERROR      -3
#define KAT_CRYPTO_FAILURE  -4
#define KAT_SUCCESS          0

// 处理错误
void HandleError(cudaError_t err, const char* file, int line)
{
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

// 初始化签名需要用到的变量
void allocateSign_Sign(sign_sig* sign){
    HANDLE_ERROR(cudaMalloc(&(sign->z), sizeof(polyvecl)*MAXGROUP )); 
    HANDLE_ERROR(cudaMalloc(&(sign->mat), K * sizeof(polyvecl)*MAXGROUP));
    HANDLE_ERROR(cudaMalloc(&(sign->s1), sizeof(polyvecl)*MAXGROUP ));
    HANDLE_ERROR(cudaMalloc(&(sign->y), sizeof(polyvecl)*MAXGROUP ));
    HANDLE_ERROR(cudaMalloc(&(sign->h), sizeof(polyveck)*MAXGROUP ));
    HANDLE_ERROR(cudaMalloc(&(sign->w0), sizeof(polyveck)*MAXGROUP ));
    HANDLE_ERROR(cudaMalloc(&(sign->w1), sizeof(polyveck)*MAXGROUP ));
    HANDLE_ERROR(cudaMalloc(&(sign->t0), sizeof(polyveck)*MAXGROUP ));
    HANDLE_ERROR(cudaMalloc(&(sign->s2), sizeof(polyveck)*MAXGROUP ));
    HANDLE_ERROR(cudaMalloc(&(sign->cp), sizeof(poly)*MAXGROUP ));
    HANDLE_ERROR(cudaMalloc(&(sign->state), sizeof(keccak_state)*MAXGROUP ));
    // CPU和GPU共用内存
    cudaMallocManaged(&(sign->flag), sizeof(int)*MAXGROUP );
    // 初始化为2
    for(int i=0;i<MAXGROUP;i++)sign->flag[i] = 2;
    cudaMallocManaged(&(sign->n), sizeof(unsigned int)*MAXGROUP );
    // 初始化为OMEGA+1
    for(int i=0;i<MAXGROUP;i++)sign->n[i]= OMEGA+1;
}

// 初始化验证需要用到的变量
void allocateSign_Verify(sign_verify* sign){
    HANDLE_ERROR(cudaMalloc(&(sign->z), sizeof(polyvecl)*MAXGROUP )); // keypaire s1hat sign z
    HANDLE_ERROR(cudaMalloc(&(sign->mat), K * sizeof(polyvecl)*MAXGROUP));
    HANDLE_ERROR(cudaMalloc(&(sign->h), sizeof(polyveck)*MAXGROUP ));
    HANDLE_ERROR(cudaMalloc(&(sign->w0), sizeof(polyveck)*MAXGROUP )); // keypaire/verify t1 sign w0
    HANDLE_ERROR(cudaMalloc(&(sign->w1), sizeof(polyveck)*MAXGROUP ));
    HANDLE_ERROR(cudaMalloc(&(sign->rho), sizeof(uint8_t) * SEEDBYTES*MAXGROUP ));
    HANDLE_ERROR(cudaMalloc(&(sign->c), sizeof(uint8_t) * SEEDBYTES*MAXGROUP ));
    HANDLE_ERROR(cudaMalloc(&(sign->buf), sizeof(uint8_t) * K * POLYW1_PACKEDBYTES*MAXGROUP ));
    HANDLE_ERROR(cudaMalloc(&(sign->c2), sizeof(uint8_t) * SEEDBYTES*MAXGROUP ));
    HANDLE_ERROR(cudaMalloc(&(sign->mu), sizeof(uint8_t) * CRHBYTES*MAXGROUP )); // keypaire tr verify mu 
    HANDLE_ERROR(cudaMalloc(&(sign->cp), sizeof(poly)*MAXGROUP ));
    HANDLE_ERROR(cudaMalloc(&(sign->state), sizeof(keccak_state)*MAXGROUP ));
    // CPU和GPU共用内存
    cudaMallocManaged(&(sign->flag), sizeof(int)*MAXGROUP );
    // 初始化为5
    for(int i=0;i<MAXGROUP;i++)sign->flag[i] = 5;
    cudaMallocManaged(&(sign->n), sizeof(unsigned int)*MAXGROUP );
    // 初始化为OMEGA+1
    for(int i=0;i<MAXGROUP;i++)sign->n[i]= OMEGA+1;
}

int
FindMarker(FILE *infile, const char *marker)
{
        char    line[MAX_MARKER_LEN];
        int             i, len;
        int curr_line;

        len = (int)strlen(marker);
        if ( len > MAX_MARKER_LEN-1 )
                len = MAX_MARKER_LEN-1;

        for ( i=0; i<len; i++ )
          {
            curr_line = fgetc(infile);
            line[i] = curr_line;
            if (curr_line == EOF )
              return 0;
          }
        line[len] = '\0';

        while ( 1 ) {
                if ( !strncmp(line, marker, len) )
                        return 1;

                for ( i=0; i<len-1; i++ )
                        line[i] = line[i+1];
                curr_line = fgetc(infile);
                line[len-1] = curr_line;
                if (curr_line == EOF )
                    return 0;
                line[len] = '\0';
        }

        // shouldn't get here
        return 0;
}

int
ReadHex(FILE *infile, unsigned char *A, int Length, char *str)
{
        int                     i, ch, started;
        unsigned char   ich;

        if ( Length == 0 ) {
                A[0] = 0x00;
                return 1;
        }
        memset(A, 0x00, Length);
        started = 0;
        if ( FindMarker(infile, str) )
                while ( (ch = fgetc(infile)) != EOF ) {
                        if ( !isxdigit(ch) ) {
                                if ( !started ) {
                                        if ( ch == '\n' )
                                                break;
                                        else
                                                continue;
                                }
                                else
                                        break;
                        }
                        started = 1;
                        if ( (ch >= '0') && (ch <= '9') )
                                ich = ch - '0';
                        else if ( (ch >= 'A') && (ch <= 'F') )
                                ich = ch - 'A' + 10;
                        else if ( (ch >= 'a') && (ch <= 'f') )
                                ich = ch - 'a' + 10;
            else // shouldn't ever get here
                ich = 0;

                        for ( i=0; i<Length-1; i++ )
                                A[i] = (A[i] << 4) | (A[i+1] >> 4);
                        A[Length-1] = (A[Length-1] << 4) | ich;
                }
        else
                return 0;

        return 1;
}

int sign(int group, unsigned char *mess, size_t mlen, uint8_t *sig, size_t *smlen, int id)
{
    int ret_val;
    // Device端创建内存 sign
    sign_sig sign[1];
    allocateSign_Sign(&sign[0]);

    // 在主机创建内存
    unsigned char *sm_h;
    unsigned char *msg_h;
    unsigned char *sk_h;
    uint8_t *signseedbuf_h;

    cudaHostAlloc((void**)&sm_h, (mlen + CRYPTO_BYTES)*MAXGROUP , cudaHostAllocDefault);
    cudaHostAlloc((void**)&msg_h, mlen*MAXGROUP , cudaHostAllocDefault);
    cudaHostAlloc((void**)&sk_h, CRYPTO_SECRETKEYBYTES*MAXGROUP , cudaHostAllocDefault);
    cudaHostAlloc((void**)&signseedbuf_h, (2*SEEDBYTES + 3*CRHBYTES)*MAXGROUP, cudaHostAllocDefault);

    // 将某一块内存中的内容全部设置为指定的值
    memset(sm_h, 0, (mlen + CRYPTO_BYTES)*MAXGROUP );
    memset(msg_h, 0, mlen*MAXGROUP );
    memset(sk_h, 0, CRYPTO_SECRETKEYBYTES*MAXGROUP );
    memset(signseedbuf_h, 0, (2*SEEDBYTES + 3*CRHBYTES)*MAXGROUP);

    // 拷贝邮件内容
    for(int i = 0;i < mlen*group;i++){
         msg_h[i] = mess[i];
    }
    // 拷贝m到sm
    for (int k = 0; k < group; ++k){
        for(int i = 0; i < mlen; ++i){
           sm_h[CRYPTO_BYTES + mlen - 1 - i + (mlen + CRYPTO_BYTES)*k] = msg_h[mlen - 1 - i + mlen * k];
        }
    }

    // 打开文件
    char fn_req[32];
    FILE *fp_req;
    sprintf(fn_req, "PQCsignKAT_%d.rsp", id);
    if ( (fp_req = fopen(fn_req, "r")) == NULL ) {
        printf("Couldn't open <%s> for read\n", fn_req);
        return KAT_FILE_OPEN_ERROR;
    }
    // 读取 sk
    if ( !ReadHex(fp_req, sk_h, CRYPTO_SECRETKEYBYTES, "sk = ") ) {
        printf("ERROR: unable to read 'sk' from <%s>\n", fn_req);
        return KAT_DATA_ERROR;
    }
    // 存储group组
    for (int k = 1; k < group; ++k){
        for(int i = 0; i < CRYPTO_SECRETKEYBYTES; ++i){
           sk_h[i+k*CRYPTO_SECRETKEYBYTES] = sk_h[i];
        }
    }

    // Host端创建一个指针变量，将这个指针变量传入到cudaMalloc()函数，Device端根据设置创建内存后，会将内存首地址赋值给Host端的指针变量
    unsigned char *sm_d;
    unsigned char *msg_d;
    unsigned char *sk_d;
    uint8_t *signseedbuf_d;

    cudaMalloc((void**)&sm_d, (mlen + CRYPTO_BYTES)*MAXGROUP );
    cudaMalloc((void**)&msg_d, mlen*MAXGROUP );
    cudaMalloc((void**)&sk_d, CRYPTO_SECRETKEYBYTES*MAXGROUP );
    cudaMalloc((void**)&signseedbuf_d, (2*SEEDBYTES + 3*CRHBYTES)*MAXGROUP);

    cudaMemcpy(sm_d, sm_h, (mlen + CRYPTO_BYTES) * group , cudaMemcpyHostToDevice);
    cudaMemcpy(msg_d, msg_h, mlen * group , cudaMemcpyHostToDevice);
    cudaMemcpy(sk_d, sk_h, CRYPTO_SECRETKEYBYTES * group , cudaMemcpyHostToDevice);
    cudaMemcpy(signseedbuf_d, signseedbuf_h, (2*SEEDBYTES + 3*CRHBYTES) * group, cudaMemcpyHostToDevice);


    cudaDeviceSynchronize();

    // smlen
    if ( (ret_val = crypto_sign(group,signseedbuf_d,&sign[0],sm_d, smlen, msg_d, mlen, sk_d)) != 0) {
        printf("crypto_sign returned <%d>\n", ret_val);
        cudaDeviceSynchronize();
        return KAT_CRYPTO_FAILURE;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(sm_h, sm_d, (mlen + CRYPTO_BYTES)*group , cudaMemcpyDeviceToHost);
    for(int i = 0;i < *smlen * group;i++){
        sig[i] = sm_h[i];
    }
    return KAT_SUCCESS;
}



int verify(int group, uint8_t *sig, size_t smlen, size_t mlen, int id){
    int ret_val;
    size_t mlen1;
    // Device端创建内存 verify
    sign_verify sign[1];
    allocateSign_Verify(&sign[0]);

    unsigned char *sm_h;
    unsigned char *msg1_h;
    unsigned char *pk_h;
    cudaHostAlloc((void**)&sm_h, (mlen + CRYPTO_BYTES)*MAXGROUP , cudaHostAllocDefault);
    cudaHostAlloc((void**)&msg1_h, mlen*MAXGROUP , cudaHostAllocDefault);
    cudaHostAlloc((void**)&pk_h, CRYPTO_PUBLICKEYBYTES*MAXGROUP , cudaHostAllocDefault);

    memset(sm_h, 0, (mlen + CRYPTO_BYTES)*MAXGROUP );
    memset(msg1_h, 0, mlen*MAXGROUP);
    memset(pk_h, 0, CRYPTO_PUBLICKEYBYTES*MAXGROUP );

    // 拷贝签名
    for(int i = 0;i < smlen*group;i++){
           sm_h[i] = sig[i];
     }
     // 打开文件
     char fn_req[32];
     FILE *fp_req;
     sprintf(fn_req, "PQCsignKAT_%d.rsp", id);
     if ( (fp_req = fopen(fn_req, "r")) == NULL ) {
            printf("Couldn't open <%s> for read\n", fn_req);
            return KAT_FILE_OPEN_ERROR;
     }
     // 读取 pk
     if ( !ReadHex(fp_req, pk_h, CRYPTO_PUBLICKEYBYTES, "pk = ") ) {
            printf("ERROR: unable to read 'sk' from <%s>\n", fn_req);
            return KAT_DATA_ERROR;
     }
    // 存储group组
    for (int k = 1; k < group; ++k){
        for(int i = 0; i < CRYPTO_PUBLICKEYBYTES; ++i){
           pk_h[i+k*CRYPTO_PUBLICKEYBYTES] = pk_h[i];
        }
    }

    unsigned char *sm_d;
    unsigned char *msg1_d;
    unsigned char *pk_d;
    cudaMalloc((void**)&sm_d, (mlen + CRYPTO_BYTES)*MAXGROUP );
    cudaMalloc((void**)&msg1_d, mlen*MAXGROUP );
    cudaMalloc((void**)&pk_d, CRYPTO_PUBLICKEYBYTES*MAXGROUP );

    cudaMemcpy(sm_d, sm_h, (mlen + CRYPTO_BYTES) * group , cudaMemcpyHostToDevice);
    cudaMemcpy(msg1_d, msg1_h, mlen * group , cudaMemcpyHostToDevice);
    cudaMemcpy(pk_d, pk_h, CRYPTO_PUBLICKEYBYTES * group , cudaMemcpyHostToDevice);

 
    // mlen1
    if ( (ret_val = crypto_sign_open(group,&sign[0],msg1_d, &mlen1, sm_d, smlen, pk_d)) != 0) {
            printf("crypto_sign_open returned <%d>\n", ret_val);
            cudaDeviceSynchronize();
            return KAT_CRYPTO_FAILURE;
    }
    cudaDeviceSynchronize();
    return KAT_SUCCESS;
}

#endif