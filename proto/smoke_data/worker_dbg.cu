// Minimal debug worker: same as Step 1 + adds cudaMemcpyAsync. Per-iter sync.
#include <cuda.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define SHM_PATH    "/dev/shm/loro_smoke_data.bin"
#define SHM_SIZE    (4ULL * 1024 * 1024)
#define OFF_W2E 0
#define OFF_E2W 4
#define OFF_READY 12
#define REQ_OFF 4096
#define RESP_OFF (4096 + 1024*1024)

#define CK(e) do { cudaError_t _e = (e); if (_e) { fprintf(stderr, "RT err %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); exit(1); } } while(0)
#define CKD(e) do { CUresult _r = (e); if (_r) { const char *m; cuGetErrorString(_r, &m); fprintf(stderr, "DRV err %s:%d %s\n", __FILE__, __LINE__, m); exit(1); } } while(0)

static double now_us(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec*1e6 + ts.tv_nsec/1e3; }

int main(int argc, char **argv) {
    int iters = (argc > 1) ? atoi(argv[1]) : 5;
    size_t payload = (argc > 2) ? atoll(argv[2]) : 16384;

    fprintf(stderr, "[dbg] open SHM\n");
    int fd = open(SHM_PATH, O_RDWR);
    if (fd < 0) { perror("open"); return 1; }
    void *hptr = mmap(NULL, SHM_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (hptr == MAP_FAILED) { perror("mmap"); return 1; }

    fprintf(stderr, "[dbg] cudaHostRegister %lluMB\n", (unsigned long long)(SHM_SIZE >> 20));
    CK(cudaHostRegister(hptr, SHM_SIZE, cudaHostRegisterPortable | cudaHostRegisterMapped));
    CK(cudaFree(0));

    void *dptr; CK(cudaHostGetDevicePointer(&dptr, hptr, 0));
    fprintf(stderr, "[dbg] hptr=%p dptr=%p\n", hptr, dptr);

    fprintf(stderr, "[dbg] cudaMalloc 2x %zu bytes\n", payload);
    void *d_req, *d_resp;
    CK(cudaMalloc(&d_req,  payload));
    CK(cudaMalloc(&d_resp, payload));
    CK(cudaMemset(d_req,  0x5a, payload));

    void *h_req  = (char*)hptr + REQ_OFF;
    void *h_resp = (char*)hptr + RESP_OFF;

    // wait enclave
    volatile uint32_t *h_ready = (volatile uint32_t*)((char*)hptr + OFF_READY);
    fprintf(stderr, "[dbg] waiting enclave (ready=%u)\n", *h_ready);
    for (int i = 0; *h_ready == 0 && i < 60; i++) usleep(500000);
    if (*h_ready != 1) { fprintf(stderr, "[dbg] no enclave\n"); return 2; }
    fprintf(stderr, "[dbg] enclave ready\n");

    cudaStream_t s; CK(cudaStreamCreate(&s));
    CUdeviceptr w2e = (CUdeviceptr)((uintptr_t)dptr + OFF_W2E);
    CUdeviceptr e2w = (CUdeviceptr)((uintptr_t)dptr + OFF_E2W);

    volatile uint32_t *h_e2w = (volatile uint32_t *)((char*)hptr + OFF_E2W);
    volatile uint32_t *h_w2e = (volatile uint32_t *)((char*)hptr + OFF_W2E);

    for (int i = 1; i <= iters; i++) {
        fprintf(stderr, "[dbg] iter %d: issuing D2H %zuB ... ", i, payload); fflush(stderr);
        double t0 = now_us();
        CK(cudaMemcpyAsync(h_req, d_req, payload, cudaMemcpyDeviceToHost, s));
        fprintf(stderr, "writeValue ... "); fflush(stderr);
        CKD(cuStreamWriteValue32((CUstream)s, w2e, i, 0));
        fprintf(stderr, "waitValue ... "); fflush(stderr);
        CKD(cuStreamWaitValue32 ((CUstream)s, e2w, i, CU_STREAM_WAIT_VALUE_EQ));
        fprintf(stderr, "H2D ... "); fflush(stderr);
        CK(cudaMemcpyAsync(d_resp, h_resp, payload, cudaMemcpyHostToDevice, s));
        fprintf(stderr, "sync ... "); fflush(stderr);
        CK(cudaStreamSynchronize(s));
        double t1 = now_us();
        fprintf(stderr, "done %.1fus  (host w2e=%u e2w=%u)\n", t1 - t0, *h_w2e, *h_e2w);
    }
    fprintf(stderr, "[dbg] all done\n");

    CK(cudaFree(d_req)); CK(cudaFree(d_resp));
    CK(cudaStreamDestroy(s));
    CK(cudaHostUnregister(hptr));
    munmap(hptr, SHM_SIZE);
    close(fd);
    return 0;
}
