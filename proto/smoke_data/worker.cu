// Step 2/7: data-bearing handoff sweep with configurable batch size.
// Usage: ./worker ITERS WARMUP BATCH SZ1 [SZ2 ...]
//   BATCH = number of handoffs to issue into the stream before one
//           cudaStreamSynchronize. BATCH=1 reproduces Step 2's per-iter-sync
//           baseline; larger BATCH amortizes the sync overhead.
// Each handoff: D2H -> WriteValue(seq) -> WaitValue(seq, EQ) -> H2D
// Enclave detects session-reset by W2E going down.
#include <cuda.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define SHM_PATH    "/dev/shm/loro_smoke_data.bin"
#define SHM_SIZE    (256ULL * 1024 * 1024)
#define OFF_W2E 0
#define OFF_E2W 4
#define OFF_SHUTDOWN 8
#define OFF_READY 12
#define HDR_BYTES 4096
#define MAX_PAYLOAD ((SHM_SIZE - HDR_BYTES) / 2)
#define REQ_OFF HDR_BYTES
#define RESP_OFF (HDR_BYTES + MAX_PAYLOAD)

#define CK(e) do { cudaError_t _e=(e); if(_e){ fprintf(stderr,"RT %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); exit(1);} } while(0)
#define CKD(e) do { CUresult _r=(e); if(_r){ const char *m; cuGetErrorString(_r,&m); fprintf(stderr,"DRV %s:%d %s\n",__FILE__,__LINE__,m); exit(1);} } while(0)

static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6 + t.tv_nsec/1e3; }

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s ITERS WARMUP BATCH SZ1 [SZ2...]\n", argv[0]); return 1; }
    int iters  = atoi(argv[1]);
    int warmup = atoi(argv[2]);
    int batch  = atoi(argv[3]);
    int n_sizes = argc - 4;
    size_t sizes[16];
    for (int i = 0; i < n_sizes && i < 16; i++) sizes[i] = (size_t)atoll(argv[4+i]);
    if (batch < 1) batch = 1;
    if (iters % batch != 0) iters = (iters / batch) * batch;   // round down to batch multiple

    int fd = open(SHM_PATH, O_RDWR);
    if (fd < 0) { perror("open"); return 1; }
    void *hptr = mmap(NULL, SHM_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (hptr == MAP_FAILED) { perror("mmap"); return 1; }

    CK(cudaHostRegister(hptr, SHM_SIZE, cudaHostRegisterPortable | cudaHostRegisterMapped));
    CK(cudaFree(0));
    void *dptr; CK(cudaHostGetDevicePointer(&dptr, hptr, 0));

    int dev; CK(cudaGetDevice(&dev));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
    printf("[worker] %s (cc %d.%d) BATCH=%d ITERS=%d\n", prop.name, prop.major, prop.minor, batch, iters); fflush(stdout);

    size_t maxp = 0;
    for (int i = 0; i < n_sizes; i++) if (sizes[i] > maxp) maxp = sizes[i];
    void *d_req, *d_resp;
    CK(cudaMalloc(&d_req,  maxp));
    CK(cudaMalloc(&d_resp, maxp));
    CK(cudaMemset(d_req,  0x5a, maxp));

    void *h_req  = (char*)hptr + REQ_OFF;
    void *h_resp = (char*)hptr + RESP_OFF;

    volatile uint32_t *h_ready = (volatile uint32_t*)((char*)hptr + OFF_READY);
    for (int i = 0; *h_ready == 0 && i < 120; i++) usleep(100000);
    if (*h_ready != 1) { fprintf(stderr, "[worker] enclave never ready\n"); return 2; }

    cudaStream_t s; CK(cudaStreamCreate(&s));
    CUdeviceptr w2e = (CUdeviceptr)((uintptr_t)dptr + OFF_W2E);
    CUdeviceptr e2w = (CUdeviceptr)((uintptr_t)dptr + OFF_E2W);

    uint32_t seq = 1;

    printf("%9s %9s %12s %12s %12s   p95(us)\n", "payload", "iters", "mean(us)", "p50(us)", "bw(GB/s)");
    fflush(stdout);

    for (int s_idx = 0; s_idx < n_sizes; s_idx++) {
        size_t payload = sizes[s_idx];
        if (payload > MAX_PAYLOAD) { fprintf(stderr,"size %zu too large\n",payload); continue; }

        // Warmup (per-batch sync).
        for (int i = 0; i < warmup; i += batch) {
            int b = (warmup - i < batch) ? (warmup - i) : batch;
            for (int j = 0; j < b; j++, seq++) {
                CK(cudaMemcpyAsync(h_req,  d_req,  payload, cudaMemcpyDeviceToHost, s));
                CKD(cuStreamWriteValue32((CUstream)s, w2e, seq, 0));
                CKD(cuStreamWaitValue32 ((CUstream)s, e2w, seq, CU_STREAM_WAIT_VALUE_EQ));
                CK(cudaMemcpyAsync(d_resp, h_resp, payload, cudaMemcpyHostToDevice, s));
            }
            CK(cudaStreamSynchronize(s));
        }

        // Measured: time per batch, divide by batch for per-handoff.
        int n_batches = iters / batch;
        double *samples = (double*)malloc(sizeof(double) * n_batches);
        for (int i = 0; i < n_batches; i++) {
            double t0 = now_us();
            for (int j = 0; j < batch; j++, seq++) {
                CK(cudaMemcpyAsync(h_req,  d_req,  payload, cudaMemcpyDeviceToHost, s));
                CKD(cuStreamWriteValue32((CUstream)s, w2e, seq, 0));
                CKD(cuStreamWaitValue32 ((CUstream)s, e2w, seq, CU_STREAM_WAIT_VALUE_EQ));
                CK(cudaMemcpyAsync(d_resp, h_resp, payload, cudaMemcpyHostToDevice, s));
            }
            CK(cudaStreamSynchronize(s));
            samples[i] = (now_us() - t0) / (double)batch;     // per-handoff
        }

        for (int a = 0; a < n_batches - 1; a++)
            for (int b = a + 1; b < n_batches; b++)
                if (samples[a] > samples[b]) { double t = samples[a]; samples[a] = samples[b]; samples[b] = t; }
        double sum = 0; for (int i = 0; i < n_batches; i++) sum += samples[i];
        double mean = sum / n_batches;
        double p50 = samples[n_batches/2];
        double p95 = samples[(int)(n_batches * 0.95)];
        double bw = (double)payload * 2.0 / mean;
        printf("%7zuB %9d %12.2f %12.2f %12.2f      %8.2f\n",
               payload, iters, mean, p50, bw / 1000.0, p95);
        fflush(stdout);
        free(samples);
    }

    CK(cudaFree(d_req)); CK(cudaFree(d_resp));
    CK(cudaStreamDestroy(s));
    CK(cudaHostUnregister(hptr));
    munmap(hptr, SHM_SIZE);
    close(fd);
    return 0;
}
