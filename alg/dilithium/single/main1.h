/*
nvcc -arch=sm_61 -rdc=true -cudart static --machine 64 -use_fast_math -O1 fips2021.cu main1.cu ntt1.cu packing1.cu poly1.cu polyvec1.cu reduce1.cu rng1.cu rounding1.cu sign1.cu symmetric-shake1.cu -o dilithium -lcudadevrt -std=c++11
nvprof ./dilithium
*/
#ifndef MAIN1_H
#define MAIN1_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <iostream>
#include "params1.h"
#include "rng1.h"
#include "sign1.h"
#include "fips2021.h"

#define	MAX_MARKER_LEN		50
#define KAT_SUCCESS          0
#define KAT_CRYPTO_FAILURE  -4
#define KAT_FILE_OPEN_ERROR -1
#define KAT_DATA_ERROR      -3

void  fprintBstr11(FILE *fp, char *S, unsigned char *A, unsigned long long l)
{
	unsigned long long  i;

	fprintf(fp, "%s", S);

	for ( i=0; i<l; i++ )
		fprintf(fp, "%02X", A[i]);

	if (l == 0 )
		fprintf(fp, "00");

	fprintf(fp, "\n");
}

int
FindMarker11(FILE *infile, const char *marker)
{
	char	line[MAX_MARKER_LEN];
	int		i, len;
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
ReadHex11(FILE *infile, unsigned char *A, int Length, char *str)
{
	int			i, ch, started;
	unsigned char	ich;

	if ( Length == 0 ) {
		A[0] = 0x00;
		return 1;
	}
	memset(A, 0x00, Length);
	started = 0;
	if ( FindMarker11(infile, str) )
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

// 处理错误
void  HandleError11(cudaError_t err, const char* file, int line)
{
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR11( err ) (HandleError11( err, __FILE__, __LINE__ ));


// 初始化需要用到的变量
void allocateSign_init(sign_init1* sign){
    HANDLE_ERROR11(cudaMalloc(&(sign->z), sizeof(polyvecl1) ));
    HANDLE_ERROR11(cudaHostAlloc(&(sign->zh), sizeof(polyvecl1), cudaHostAllocDefault));
    HANDLE_ERROR11(cudaMalloc(&(sign->mat), K * sizeof(polyvecl1)));
    HANDLE_ERROR11(cudaHostAlloc(&(sign->math), K * sizeof(polyvecl1), cudaHostAllocDefault));
    HANDLE_ERROR11(cudaHostAlloc(&(sign->s1h), sizeof(polyvecl1), cudaHostAllocDefault));
    HANDLE_ERROR11(cudaMalloc(&(sign->w0), sizeof(polyveck1) ));
    HANDLE_ERROR11(cudaHostAlloc(&(sign->w0h), sizeof(polyveck1), cudaHostAllocDefault));
    HANDLE_ERROR11(cudaMalloc(&(sign->t0), sizeof(polyveck1) ));
    HANDLE_ERROR11(cudaHostAlloc(&(sign->t0h), sizeof(polyveck1), cudaHostAllocDefault));
    HANDLE_ERROR11(cudaMalloc(&(sign->s2), sizeof(polyveck1) ));
    HANDLE_ERROR11(cudaHostAlloc(&(sign->s2h), sizeof(polyveck1), cudaHostAllocDefault));
    HANDLE_ERROR11(cudaHostAlloc(&(sign->muh), sizeof(uint8_t) * CRHBYTES, cudaHostAllocDefault));
}

// 签名需要用到的变量
void allocateSign_Sign(sign_sig1* sign){
    HANDLE_ERROR11(cudaMalloc(&(sign->z), sizeof(polyvecl1) ));
    HANDLE_ERROR11(cudaHostAlloc(&(sign->zh), sizeof(polyvecl1), cudaHostAllocDefault));
    HANDLE_ERROR11(cudaMalloc(&(sign->mat), K * sizeof(polyvecl1)));
    HANDLE_ERROR11(cudaHostAlloc(&(sign->math), K * sizeof(polyvecl1), cudaHostAllocDefault));
    HANDLE_ERROR11(cudaMalloc(&(sign->s1), sizeof(polyvecl1) ));
    HANDLE_ERROR11(cudaHostAlloc(&(sign->s1h), sizeof(polyvecl1), cudaHostAllocDefault));
    HANDLE_ERROR11(cudaMalloc(&(sign->y), sizeof(polyvecl1) ));
    HANDLE_ERROR11(cudaHostAlloc(&(sign->yh), sizeof(polyvecl1), cudaHostAllocDefault));
    HANDLE_ERROR11(cudaMalloc(&(sign->h), sizeof(polyveck1) ));
    HANDLE_ERROR11(cudaHostAlloc(&(sign->hh), sizeof(polyveck1), cudaHostAllocDefault));
    HANDLE_ERROR11(cudaMalloc(&(sign->w0), sizeof(polyveck1) ));
    HANDLE_ERROR11(cudaHostAlloc(&(sign->w0h), sizeof(polyveck1), cudaHostAllocDefault));
    HANDLE_ERROR11(cudaMalloc(&(sign->w1), sizeof(polyveck1) ));
    HANDLE_ERROR11(cudaHostAlloc(&(sign->w1h), sizeof(polyveck1), cudaHostAllocDefault));
    HANDLE_ERROR11(cudaMalloc(&(sign->t0), sizeof(polyveck1) ));
    HANDLE_ERROR11(cudaHostAlloc(&(sign->t0h), sizeof(polyveck1), cudaHostAllocDefault));
    HANDLE_ERROR11(cudaMalloc(&(sign->s2), sizeof(polyveck1) ));
    HANDLE_ERROR11(cudaHostAlloc(&(sign->s2h), sizeof(polyveck1), cudaHostAllocDefault));
    HANDLE_ERROR11(cudaMalloc(&(sign->cp), sizeof(poly1) ));
    HANDLE_ERROR11(cudaHostAlloc(&(sign->cph), sizeof(poly1), cudaHostAllocDefault));
    HANDLE_ERROR11(cudaHostAlloc(&(sign->stateh), sizeof(keccak_state1), cudaHostAllocDefault));
}


// 验证需要用到的变量
void allocateVerify_Sign(sign_verify1* sign){
    HANDLE_ERROR11(cudaMalloc(&(sign->z), sizeof(polyvecl1) )); // keypaire s1hat sign z
    HANDLE_ERROR11(cudaHostAlloc(&(sign->zh), sizeof(polyvecl1), cudaHostAllocDefault));
    HANDLE_ERROR11(cudaMalloc(&(sign->mat), K * sizeof(polyvecl1)));
    HANDLE_ERROR11(cudaHostAlloc(&(sign->math), K * sizeof(polyvecl1), cudaHostAllocDefault));
    HANDLE_ERROR11(cudaMalloc(&(sign->h), sizeof(polyveck1) ));
    HANDLE_ERROR11(cudaHostAlloc(&(sign->hh), sizeof(polyveck1), cudaHostAllocDefault));
    HANDLE_ERROR11(cudaMalloc(&(sign->w0), sizeof(polyveck1) )); // keypaire/verify t1 sign w0
    HANDLE_ERROR11(cudaHostAlloc(&(sign->w0h), sizeof(polyveck1), cudaHostAllocDefault));
    HANDLE_ERROR11(cudaMalloc(&(sign->w1), sizeof(polyveck1) ));
    HANDLE_ERROR11(cudaHostAlloc(&(sign->muh), sizeof(uint8_t) * CRHBYTES, cudaHostAllocDefault));
    HANDLE_ERROR11(cudaMalloc(&(sign->cp), sizeof(poly1) ));
    HANDLE_ERROR11(cudaHostAlloc(&(sign->cph), sizeof(poly1), cudaHostAllocDefault));
    HANDLE_ERROR11(cudaHostAlloc(&(sign->rhoh), sizeof(uint8_t) * SEEDBYTES, cudaHostAllocDefault));
    HANDLE_ERROR11(cudaMalloc(&(sign->buf), sizeof(uint8_t) * K * POLYW1_PACKEDBYTES ));
    HANDLE_ERROR11(cudaHostAlloc(&(sign->bufh), sizeof(uint8_t) * K * POLYW1_PACKEDBYTES , cudaHostAllocDefault));
    HANDLE_ERROR11(cudaHostAlloc(&(sign->c2h), sizeof(uint8_t) * SEEDBYTES , cudaHostAllocDefault));
    HANDLE_ERROR11(cudaHostAlloc(&(sign->ch), sizeof(uint8_t) * SEEDBYTES , cudaHostAllocDefault));
    HANDLE_ERROR11(cudaHostAlloc(&(sign->stateh), sizeof(keccak_state1), cudaHostAllocDefault));
}


int init11(int id)
{
        int  ret_val;
        unsigned char *pk_h;
        unsigned char *sk_h;
        uint8_t *keypairseedbuf_h;

        // Device端创建内存 init
        sign_init1 init[1];
        allocateSign_init(&init[0]);

        cudaHostAlloc((void**)&pk_h, CRYPTO_PUBLICKEYBYTES,cudaHostAllocDefault);
        cudaHostAlloc((void**)&sk_h, CRYPTO_SECRETKEYBYTES,cudaHostAllocDefault);
        cudaHostAlloc((void**)&keypairseedbuf_h, (3*SEEDBYTES) , cudaHostAllocDefault);

        rrandombytes11(keypairseedbuf_h, 3 * SEEDBYTES , SEEDBYTES,id);
        cudaDeviceSynchronize();
        if ( (ret_val = crypto_sign_keypair(keypairseedbuf_h,&init[0], pk_h, sk_h)) != 0) {
            printf("crypto_sign_keypair returned <%d>\n", ret_val);
            cudaDeviceSynchronize();
            return KAT_CRYPTO_FAILURE;
        }
        cudaDeviceSynchronize();

        // 打开文件
        char fn_rsp[32];
        FILE *fp_rsp;
        sprintf(fn_rsp, "PQCsignKAT_%d.rsp", id);
        if ( (fp_rsp = fopen(fn_rsp, "w")) == NULL ) {
            printf("Couldn't open <%s> for write\n", fn_rsp);
            return KAT_FILE_OPEN_ERROR;
        }
        // 将key存入文件中
        fprintBstr11(fp_rsp, "pk = ", pk_h, CRYPTO_PUBLICKEYBYTES);
        fprintBstr11(fp_rsp, "sk = ", sk_h, CRYPTO_SECRETKEYBYTES);

        fclose(fp_rsp);
        return KAT_SUCCESS;
}



int sign1(unsigned char *mess, size_t mlen, uint8_t *sig, size_t *smlen, int id)
{
        int  ret_val;
        unsigned char *msg_h;
        cudaHostAlloc((void**)&msg_h, mlen , cudaHostAllocDefault);
        unsigned char *sk_h;
        cudaHostAlloc((void**)&sk_h, CRYPTO_SECRETKEYBYTES, cudaHostAllocDefault);
        unsigned char *sm_h;
        cudaHostAlloc((void**)&sm_h, (mlen + CRYPTO_BYTES) , cudaHostAllocDefault);
        uint8_t *signseedbuf_h;
        cudaHostAlloc((void**)&signseedbuf_h, (2*SEEDBYTES + 3*CRHBYTES), cudaHostAllocDefault);

        // Device端创建内存 sign
        sign_sig1 sign[1];
        allocateSign_Sign(&sign[0]);

        // 拷贝邮件内容
        for(int i = 0;i < mlen;i++){
           msg_h[i] = mess[i];
        }
        // 拷贝m到sm
        for(int i = 0; i < mlen; ++i){
           sm_h[CRYPTO_BYTES + mlen - 1 - i ] = msg_h[mlen - 1 - i ];
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
        if ( !ReadHex11(fp_req, sk_h, CRYPTO_SECRETKEYBYTES, "sk = ") ) {
            printf("ERROR: unable to read 'sk' from <%s>\n", fn_req);
            return KAT_DATA_ERROR;
        }

        if ( (ret_val = crypto_sign(signseedbuf_h,&sign[0],sm_h, smlen, msg_h, mlen, sk_h)) != 0) {
            printf("crypto_sign returned <%d>\n", ret_val);
            cudaDeviceSynchronize();
            return KAT_CRYPTO_FAILURE;
        }
        cudaDeviceSynchronize();

        for(int i = 0;i < *smlen;i++){
           sig[i] = sm_h[i];
        }
        return KAT_SUCCESS;
}


int verify13(uint8_t *sig, size_t smlen, size_t mlen, int id)
{
        int  ret_val;
        size_t mlen1;
        // Device端创建内存 verify
        sign_verify1 sign[1];
        allocateVerify_Sign(&sign[0]);

        unsigned char *msg1_h;
        cudaHostAlloc((void**)&msg1_h, mlen , cudaHostAllocDefault);
        unsigned char *sm_h;
        cudaHostAlloc((void**)&sm_h, (mlen + CRYPTO_BYTES) , cudaHostAllocDefault);
        unsigned char *pk_h;
        cudaHostAlloc((void**)&pk_h, CRYPTO_PUBLICKEYBYTES,cudaHostAllocDefault);
        // 拷贝签名
        for(int i = 0;i < smlen;i++){
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
        if ( !ReadHex11(fp_req, pk_h, CRYPTO_PUBLICKEYBYTES, "pk = ") ) {
            printf("ERROR: unable to read 'sk' from <%s>\n", fn_req);
            return KAT_DATA_ERROR;
        }

        if ( (ret_val = crypto_sign_open(&sign[0],msg1_h, &mlen1, sm_h, smlen, pk_h)) != 0) {
            printf("crypto_sign_open returned <%d>\n", ret_val);
            cudaDeviceSynchronize();
            return KAT_CRYPTO_FAILURE;
        }
        cudaDeviceSynchronize();
        return KAT_SUCCESS;
}

#endif