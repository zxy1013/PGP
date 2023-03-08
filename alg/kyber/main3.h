
// @Author: Arpan Jati
// Adapted from NewHope Reference Codebase and Parallelized using CUDA
// Updated : August 2019

#ifndef MAIN3_H
#define MAIN3_H

#include "rng3.h"
#include "api3.h"
#include "params3.h"
#include "indcpa3.h"
#include "poly3.h"
#include "polyvec3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <iostream>
#include <ctype.h>
#include <chrono>

#define MAX_MARKER_LEN		50
#define KAT_SUCCESS          0
#define KAT_FILE_OPEN_ERROR -1
#define KAT_DATA_ERROR      -3
#define KAT_CRYPTO_FAILURE  -4

using namespace std;




// 处理错误
void HandleError3(cudaError_t err, const char* file, int line)
{
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR3( err ) (HandleError3( err, __FILE__, __LINE__ ))


void allocatePolySet(poly_set4* polySet)
{
	HANDLE_ERROR3(cudaMalloc(&(polySet->a), sizeof(poly3)));
	HANDLE_ERROR3(cudaMalloc(&(polySet->b), sizeof(poly3)));
	HANDLE_ERROR3(cudaMalloc(&(polySet->c), sizeof(poly3)));
	HANDLE_ERROR3(cudaMalloc(&(polySet->d), sizeof(poly3)));
	HANDLE_ERROR3(cudaMalloc(&(polySet->av), sizeof(polyvec)));
	HANDLE_ERROR3(cudaMalloc(&(polySet->bv), sizeof(polyvec)));
	HANDLE_ERROR3(cudaMalloc(&(polySet->cv), sizeof(polyvec)));
	HANDLE_ERROR3(cudaMalloc(&(polySet->dv), sizeof(polyvec)));
	HANDLE_ERROR3(cudaMalloc(&(polySet->ev), sizeof(polyvec)));
	HANDLE_ERROR3(cudaMalloc(&(polySet->fv), sizeof(polyvec)));
	HANDLE_ERROR3(cudaMalloc(&(polySet->AV), sizeof(polyvec) * 4));
	HANDLE_ERROR3(cudaMalloc(&(polySet->seed), (KYBER_SYMBYTES * 2) * N_TESTS));
	HANDLE_ERROR3(cudaMalloc(&(polySet->large_buffer_a), LARGE_BUFFER_SZ * N_TESTS));
	HANDLE_ERROR3(cudaMalloc(&(polySet->large_buffer_b), LARGE_BUFFER_SZ * N_TESTS));
}

void freePolySet(poly_set4* polySet)
{
	HANDLE_ERROR3(cudaFree(polySet->a));
	HANDLE_ERROR3(cudaFree(polySet->b));
	HANDLE_ERROR3(cudaFree(polySet->c));
	HANDLE_ERROR3(cudaFree(polySet->d));
	HANDLE_ERROR3(cudaFree(polySet->av));
	HANDLE_ERROR3(cudaFree(polySet->bv));
	HANDLE_ERROR3(cudaFree(polySet->cv));
	HANDLE_ERROR3(cudaFree(polySet->dv));
	HANDLE_ERROR3(cudaFree(polySet->ev));
	HANDLE_ERROR3(cudaFree(polySet->fv));
	HANDLE_ERROR3(cudaFree(polySet->seed));
	HANDLE_ERROR3(cudaFree(polySet->large_buffer_a));
	HANDLE_ERROR3(cudaFree(polySet->large_buffer_b));
}

void fprintBstr1(FILE *fp, char *S, unsigned char *A, unsigned long long l)
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
FindMarker1(FILE *infile, const char *marker)
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
ReadHex1(FILE *infile, unsigned char *A, int Length, char *str)
{
        int                     i, ch, started;
        unsigned char   ich;

        if ( Length == 0 ) {
                A[0] = 0x00;
                return 1;
        }
        memset(A, 0x00, Length);
        started = 0;
        if ( FindMarker1(infile, str) )
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


int init1(int id)
{
	int COUNT = 1;
	poly_set4 tempPoly_0[4];
	allocatePolySet(&tempPoly_0[0]);

	unsigned char* pk_h_0;
	unsigned char* sk_h_0;
	unsigned char* rng_buf_h_0;
	HANDLE_ERROR3(cudaHostAlloc((void**)&pk_h_0, KYBER_INDCPA_PUBLICKEYBYTES * N_TESTS, cudaHostAllocDefault));
	HANDLE_ERROR3(cudaHostAlloc((void**)&sk_h_0, KYBER_INDCPA_SECRETKEYBYTES * N_TESTS, cudaHostAllocDefault));
	HANDLE_ERROR3(cudaHostAlloc((void**)&rng_buf_h_0, KYBER_SYMBYTES * 2 * N_TESTS, cudaHostAllocDefault));

	rrandombytes(rng_buf_h_0, KYBER_SYMBYTES * N_TESTS * 2,id);

	unsigned char* pk_d_0;
	unsigned char* sk_d_0;
	unsigned char* rng_buf_d_0;
	HANDLE_ERROR3(cudaMalloc((void**)&pk_d_0, KYBER_INDCPA_PUBLICKEYBYTES * N_TESTS));
	HANDLE_ERROR3(cudaMalloc((void**)&sk_d_0, KYBER_INDCPA_SECRETKEYBYTES * N_TESTS));
	HANDLE_ERROR3(cudaMalloc((void**)&rng_buf_d_0, KYBER_SYMBYTES * 2 * N_TESTS));
	
	cudaStream_t stream_0;
	HANDLE_ERROR3(cudaStreamCreate(&stream_0));
	HANDLE_ERROR3(cudaMemcpyAsync(pk_d_0, pk_h_0, KYBER_INDCPA_PUBLICKEYBYTES * COUNT, cudaMemcpyHostToDevice, stream_0));
	HANDLE_ERROR3(cudaMemcpyAsync(sk_d_0, sk_h_0, KYBER_INDCPA_SECRETKEYBYTES * COUNT, cudaMemcpyHostToDevice, stream_0));
	HANDLE_ERROR3(cudaMemcpyAsync(rng_buf_d_0, rng_buf_h_0, KYBER_SYMBYTES * 2 * COUNT, cudaMemcpyDeviceToHost, stream_0));

	cudaDeviceSynchronize();
	indcpa_keypair(1, &tempPoly_0[0], pk_d_0, sk_d_0, rng_buf_d_0, stream_0);
	cudaDeviceSynchronize();

	HANDLE_ERROR3(cudaMemcpyAsync(pk_h_0, pk_d_0, KYBER_INDCPA_PUBLICKEYBYTES * COUNT, cudaMemcpyDeviceToHost, stream_0));
	HANDLE_ERROR3(cudaMemcpyAsync(sk_h_0, sk_d_0, KYBER_INDCPA_SECRETKEYBYTES * COUNT, cudaMemcpyDeviceToHost, stream_0));
	cudaDeviceSynchronize();

	HANDLE_ERROR3(cudaStreamDestroy(stream_0));
	// 写入文件
	char fn_rsp[32];
	FILE *fp_rsp;
	sprintf(fn_rsp, "PQCENCKAT_%d.rsp", id);
	if ( (fp_rsp = fopen(fn_rsp, "w")) == NULL ) {
		printf("Couldn't open <%s> for write\n", fn_rsp);
		return KAT_FILE_OPEN_ERROR;
	}

	// 将key存入文件中
	fprintBstr1(fp_rsp, "pk = ", pk_h_0, KYBER_INDCPA_PUBLICKEYBYTES);
	fprintBstr1(fp_rsp, "sk = ", sk_h_0, KYBER_INDCPA_SECRETKEYBYTES);
	fclose(fp_rsp);
	return 0;
}



int enc1(int group, unsigned char *mess, size_t mlen, unsigned char *ct, int id)
{
	poly_set4 tempPoly_0[4];
	allocatePolySet(&tempPoly_0[0]);

	unsigned char* pk_h_0;
	unsigned char* msg1_h_0;
	unsigned char* ct_h_0;
	unsigned char* coins_h_0;

	HANDLE_ERROR3(cudaHostAlloc((void**)&pk_h_0, KYBER_INDCPA_PUBLICKEYBYTES * N_TESTS, cudaHostAllocDefault));
	HANDLE_ERROR3(cudaHostAlloc((void**)&msg1_h_0, mlen * N_TESTS, cudaHostAllocDefault));
	HANDLE_ERROR3(cudaHostAlloc((void**)&ct_h_0, KYBER_INDCPA_BYTES * N_TESTS, cudaHostAllocDefault));
	HANDLE_ERROR3(cudaHostAlloc((void**)&coins_h_0, KYBER_SYMBYTES * N_TESTS, cudaHostAllocDefault));

	unsigned char* pk_d_0;
	unsigned char* ct_d_0;
	unsigned char* msg1_d_0;
	unsigned char* coins_d_0;

	HANDLE_ERROR3(cudaMalloc((void**)&pk_d_0, KYBER_INDCPA_PUBLICKEYBYTES * N_TESTS));
	HANDLE_ERROR3(cudaMalloc((void**)&ct_d_0, KYBER_INDCPA_BYTES * N_TESTS));
	HANDLE_ERROR3(cudaMalloc((void**)&msg1_d_0, mlen * N_TESTS));
	HANDLE_ERROR3(cudaMalloc((void**)&coins_d_0, KYBER_SYMBYTES * N_TESTS));
	
	// 拷贝邮件内容
	memset(msg1_h_0, 0, KYBER_SYMBYTES * N_TESTS);
	for(int i = 0;i < mlen*group;i++){
	     msg1_h_0[i] = mess[i];
	}
	randombytes1(coins_h_0, KYBER_SYMBYTES * N_TESTS);

	// 查询pk
	// 打开文件
	char fn_req[32];
	FILE *fp_req;
	sprintf(fn_req, "PQCENCKAT_%d.rsp", id);
	if ( (fp_req = fopen(fn_req, "r")) == NULL ) {
	    printf("Couldn't open <%s> for read\n", fn_req);
	    return KAT_FILE_OPEN_ERROR;
	}
	// 读取pk
	if ( !ReadHex1(fp_req, pk_h_0, KYBER_INDCPA_PUBLICKEYBYTES, "pk = ") ) {
	    printf("ERROR: unable to read 'pk' from <%s>\n", fn_req);
	    return KAT_DATA_ERROR;
	}
	// 存储group组
	for (int k = 1; k < group; ++k){
	    for(int i = 0; i < KYBER_INDCPA_PUBLICKEYBYTES; ++i){
		pk_h_0[i+k*KYBER_INDCPA_PUBLICKEYBYTES] = pk_h_0[i];
	    }
	}

	cudaStream_t stream_0;
	HANDLE_ERROR3(cudaStreamCreate(&stream_0));
	cudaDeviceSynchronize();
	HANDLE_ERROR3(cudaMemcpyAsync(pk_d_0, pk_h_0, KYBER_INDCPA_PUBLICKEYBYTES * group, cudaMemcpyHostToDevice, stream_0));
	HANDLE_ERROR3(cudaMemcpyAsync(ct_d_0, ct_h_0, KYBER_INDCPA_BYTES * group, cudaMemcpyHostToDevice, stream_0));
	HANDLE_ERROR3(cudaMemcpyAsync(msg1_d_0, msg1_h_0, mlen * group, cudaMemcpyHostToDevice, stream_0));
	HANDLE_ERROR3(cudaMemcpyAsync(coins_d_0, coins_h_0, KYBER_SYMBYTES * group, cudaMemcpyHostToDevice, stream_0));

	cudaDeviceSynchronize();
	indcpa_enc(group, &tempPoly_0[0], ct_d_0, msg1_d_0, pk_d_0, coins_d_0, stream_0);
	cudaDeviceSynchronize();
	
	HANDLE_ERROR3(cudaMemcpyAsync(ct_h_0, ct_d_0, KYBER_INDCPA_BYTES * group, cudaMemcpyDeviceToHost, stream_0));
	cudaDeviceSynchronize();
	// 存储密文
	for(int i = 0;i < KYBER_INDCPA_BYTES *group;i++){
	     ct[i] = ct_h_0[i];
	}
	HANDLE_ERROR3(cudaStreamDestroy(stream_0));
	return 0;
}



int dec1(int group, unsigned char *mess, size_t mlen, unsigned char *ct, int id){

	poly_set4 tempPoly_0[4];
	allocatePolySet(&tempPoly_0[0]);

	unsigned char* sk_h_0;
	unsigned char* ct_h_0;
	unsigned char* msg2_h_0;
	HANDLE_ERROR3(cudaHostAlloc((void**)&sk_h_0, KYBER_INDCPA_SECRETKEYBYTES * N_TESTS, cudaHostAllocDefault));
	HANDLE_ERROR3(cudaHostAlloc((void**)&ct_h_0, KYBER_INDCPA_BYTES * N_TESTS, cudaHostAllocDefault));
	HANDLE_ERROR3(cudaHostAlloc((void**)&msg2_h_0, mlen * N_TESTS, cudaHostAllocDefault));

	// 拷贝密文
	for(int i = 0;i < KYBER_INDCPA_BYTES * group;i++){
	    ct_h_0[i] = ct[i];
	}

	// 查询sk
	// 打开文件
	char fn_req[32];
	FILE *fp_req;
	sprintf(fn_req, "PQCENCKAT_%d.rsp", id);
	if ( (fp_req = fopen(fn_req, "r")) == NULL ) {
	    printf("Couldn't open <%s> for read\n", fn_req);
	    return KAT_FILE_OPEN_ERROR;
	}
	// 读取 sk
	if ( !ReadHex1(fp_req, sk_h_0, KYBER_INDCPA_SECRETKEYBYTES, "sk = ") ) {
	    printf("ERROR: unable to read 'sk' from <%s>\n", fn_req);
	    return KAT_DATA_ERROR;
	}
	// 存储group组
	for (int k = 1; k < group; ++k){
	    for(int i = 0; i < KYBER_INDCPA_SECRETKEYBYTES; ++i){
		sk_h_0[i+k*KYBER_INDCPA_SECRETKEYBYTES] = sk_h_0[i];
	    }
	}

	unsigned char* sk_d_0;
	unsigned char* ct_d_0;
	unsigned char* msg2_d_0;
	HANDLE_ERROR3(cudaMalloc((void**)&sk_d_0, KYBER_INDCPA_SECRETKEYBYTES * N_TESTS));
	HANDLE_ERROR3(cudaMalloc((void**)&ct_d_0, KYBER_INDCPA_BYTES * N_TESTS));
	HANDLE_ERROR3(cudaMalloc((void**)&msg2_d_0, mlen * N_TESTS));

	cudaStream_t stream_0;
	HANDLE_ERROR3(cudaStreamCreate(&stream_0));
	cudaDeviceSynchronize();
	HANDLE_ERROR3(cudaMemcpyAsync(sk_d_0, sk_h_0, KYBER_INDCPA_SECRETKEYBYTES * group, cudaMemcpyHostToDevice, stream_0));
	HANDLE_ERROR3(cudaMemcpyAsync(ct_d_0, ct_h_0, KYBER_INDCPA_BYTES * group, cudaMemcpyHostToDevice, stream_0));
	HANDLE_ERROR3(cudaMemcpyAsync(msg2_d_0, msg2_h_0, mlen * group, cudaMemcpyHostToDevice, stream_0));

	cudaDeviceSynchronize();
	indcpa_dec(group, &tempPoly_0[0], msg2_d_0, ct_d_0, sk_d_0, stream_0);
	cudaDeviceSynchronize();

	HANDLE_ERROR3(cudaMemcpyAsync(msg2_h_0, msg2_d_0, mlen * group, cudaMemcpyDeviceToHost, stream_0));
	cudaDeviceSynchronize();

	// 拷贝邮件内容
	for(int i = 0;i < mlen*group;i++){
	     mess[i] = msg2_h_0[i];
	}

	HANDLE_ERROR3(cudaStreamDestroy(stream_0));
	freePolySet(&tempPoly_0[0]);
	// Check for any errors launching the kernel
	HANDLE_ERROR3(cudaGetLastError());
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	HANDLE_ERROR3(cudaDeviceSynchronize());
	HANDLE_ERROR3(cudaDeviceReset());
	return 0;
}
#endif