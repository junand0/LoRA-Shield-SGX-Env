// Step 4: multi-lane worker. Fans handoffs across N CUDA streams; each lane
// has its own header (W2E/E2W flags) and payload region in shared memory.
// Measures wall-clock per "round" (one handoff per lane) so we can see how
// well lanes scale in parallel.
//
// Usage: ./worker_ml N_LANES ITERS WARMUP BATCH SZ1 [SZ2 ...]
//   ITERS = rounds (each round issues one handoff to every lane)
//   BATCH = rounds queued before one sync (per-stream)
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

#define SHM_PATH        "/dev/shm/loro_smoke_ml.bin"
#define SHM_SIZE        (256ULL * 1024 * 1024)
#define LANE_HDR_SIZE   64
#define MAX_LANES       16
#define HDR_BYTES       4096
#define LANE_OFF_W2E    0
#define LANE_OFF_E2W    4
#define LANE_OFF_READY  8
#define OFF_GLOBAL_SHUTDOWN  (LANE_HDR_SIZE * MAX_LANES)

#define CK(e)  do { cudaError_t _e=(e); if(_e){ fprintf(stderr,"RT %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); exit(1);} } while(0)
#define CKD(e) do { CUresult _r=(e); if(_r){ const char *m; cuGetErrorString(_r,&m); fprintf(stderr,"DRV %s:%d %s\n",__FILE__,__LINE__,m); exit(1);} } while(0)

static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6 + t.tv_nsec/1e3; }

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s N_LANES ITERS WARMUP BATCH SZ1 [SZ2...]\n", argv[0]);
        return 1;
    }
    int n_lanes = atoi(argv[1]);
    int iters   = atoi(argv[2]);
    int warmup  = atoi(argv[3]);
    int batch   = atoi(argv[4]);
    int n_sizes = argc - 5;
    size_t sizes[16];
    for (int i = 0; i < n_sizes && i < 16; i++) sizes[i] = (size_t)atoll(argv[5+i]);
    if (n_lanes < 1 || n_lanes > MAX_LANES) { fprintf(stderr,"bad n_lanes\n"); return 1; }
    if (batch < 1) batch = 1;

    int fd = open(SHM_PATH, O_RDWR);
    if (fd < 0) { perror("open"); return 1; }
    void *hptr = mmap(NULL, SHM_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (hptr == MAP_FAILED) { perror("mmap"); return 1; }

    CK(cudaHostRegister(hptr, SHM_SIZE, cudaHostRegisterPortable | cudaHostRegisterMapped));
    CK(cudaFree(0));
    void *dptr; CK(cudaHostGetDevicePointer(&dptr, hptr, 0));

    int dev; CK(cudaGetDevice(&dev));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
    printf("[worker] %s (cc %d.%d) N_LANES=%d BATCH=%d ITERS=%d\n",
           prop.name, prop.major, prop.minor, n_lanes, batch, iters); fflush(stdout);

    // Per-lane buffer plan: payload region size per-lane = (SHM_SIZE - HDR_BYTES) / n_lanes,
    // split into req half + resp half.
    size_t lane_region = (SHM_SIZE - HDR_BYTES) / n_lanes;
    size_t lane_half   = lane_region / 2;

    size_t maxp = 0;
    for (int i = 0; i < n_sizes; i++) if (sizes[i] > maxp) maxp = sizes[i];
    if (maxp > lane_half) { fprintf(stderr,"payload %zu exceeds lane half %zu\n", maxp, lane_half); return 1; }

    // GPU buffers (one set per lane).
    void *d_req[MAX_LANES], *d_resp[MAX_LANES];
    void *h_req[MAX_LANES], *h_resp[MAX_LANES];
    CUdeviceptr w2e_dev[MAX_LANES], e2w_dev[MAX_LANES];
    cudaStream_t streams[MAX_LANES];
    uint32_t seq[MAX_LANES];
    volatile uint32_t *h_ready[MAX_LANES];

    for (int i = 0; i < n_lanes; i++) {
        CK(cudaMalloc(&d_req[i],  maxp));
        CK(cudaMalloc(&d_resp[i], maxp));
        CK(cudaMemset(d_req[i],  0x5a, maxp));
        h_req[i]  = (char*)hptr + HDR_BYTES + i * lane_region;
        h_resp[i] = (char*)hptr + HDR_BYTES + i * lane_region + lane_half;
        CK(cudaStreamCreate(&streams[i]));
        w2e_dev[i] = (CUdeviceptr)((uintptr_t)dptr + i * LANE_HDR_SIZE + LANE_OFF_W2E);
        e2w_dev[i] = (CUdeviceptr)((uintptr_t)dptr + i * LANE_HDR_SIZE + LANE_OFF_E2W);
        h_ready[i] = (volatile uint32_t *)((char*)hptr + i * LANE_HDR_SIZE + LANE_OFF_READY);
        seq[i] = 1;
    }

    // Wait for all lanes ready.
    for (int i = 0; i < n_lanes; i++) {
        for (int t = 0; *h_ready[i] == 0 && t < 120; t++) usleep(100000);
        if (*h_ready[i] != 1) { fprintf(stderr,"[worker] lane %d not ready\n", i); return 2; }
    }

    printf("%9s %9s %9s %12s %12s %12s   p95(us)\n", "payload", "lanes", "rounds", "mean(us)", "p50(us)", "agg_bw");
    fflush(stdout);

    for (int s_idx = 0; s_idx < n_sizes; s_idx++) {
        size_t payload = sizes[s_idx];

        // Warmup.
        for (int i = 0; i < warmup; i++) {
            for (int lane = 0; lane < n_lanes; lane++) {
                CK(cudaMemcpyAsync(h_req[lane],  d_req[lane],  payload, cudaMemcpyDeviceToHost, streams[lane]));
                CKD(cuStreamWriteValue32((CUstream)streams[lane], w2e_dev[lane], seq[lane], 0));
                CKD(cuStreamWaitValue32 ((CUstream)streams[lane], e2w_dev[lane], seq[lane], CU_STREAM_WAIT_VALUE_EQ));
                CK(cudaMemcpyAsync(d_resp[lane], h_resp[lane], payload, cudaMemcpyHostToDevice, streams[lane]));
                seq[lane]++;
            }
            for (int lane = 0; lane < n_lanes; lane++) CK(cudaStreamSynchronize(streams[lane]));
        }

        int n_batches = iters / batch;
        if (n_batches < 1) n_batches = 1;
        double *samples = (double*)malloc(sizeof(double) * n_batches);

        for (int bi = 0; bi < n_batches; bi++) {
            double t0 = now_us();
            for (int rnd = 0; rnd < batch; rnd++) {
                for (int lane = 0; lane < n_lanes; lane++) {
                    CK(cudaMemcpyAsync(h_req[lane],  d_req[lane],  payload, cudaMemcpyDeviceToHost, streams[lane]));
                    CKD(cuStreamWriteValue32((CUstream)streams[lane], w2e_dev[lane], seq[lane], 0));
                    CKD(cuStreamWaitValue32 ((CUstream)streams[lane], e2w_dev[lane], seq[lane], CU_STREAM_WAIT_VALUE_EQ));
                    CK(cudaMemcpyAsync(d_resp[lane], h_resp[lane], payload, cudaMemcpyHostToDevice, streams[lane]));
                    seq[lane]++;
                }
            }
            // Sync all lanes for this batch.
            for (int lane = 0; lane < n_lanes; lane++) CK(cudaStreamSynchronize(streams[lane]));
            samples[bi] = (now_us() - t0) / (double)batch;     // time per "round" (one handoff per lane)
        }

        for (int a = 0; a < n_batches - 1; a++)
            for (int b = a + 1; b < n_batches; b++)
                if (samples[a] > samples[b]) { double t = samples[a]; samples[a] = samples[b]; samples[b] = t; }
        double sum = 0; for (int i = 0; i < n_batches; i++) sum += samples[i];
        double mean = sum / n_batches;
        double p50 = samples[n_batches/2];
        double p95 = samples[(int)(n_batches * 0.95)];
        // Aggregate bandwidth = (n_lanes lanes × 2 directions × payload bytes) / round_time(us) -> MB/s
        double agg_bw = (double)n_lanes * payload * 2.0 / mean;
        printf("%7zuB %9d %9d %12.2f %12.2f %12.2f      %8.2f\n",
               payload, n_lanes, iters, mean, p50, agg_bw / 1000.0, p95);
        fflush(stdout);
        free(samples);
    }

    // Trigger shutdown via global shutdown flag.
    volatile uint32_t *gshut = (volatile uint32_t *)((char*)hptr + OFF_GLOBAL_SHUTDOWN);
    *gshut = 0xFFFFFFFFu;

    for (int i = 0; i < n_lanes; i++) {
        CK(cudaFree(d_req[i]));
        CK(cudaFree(d_resp[i]));
        CK(cudaStreamDestroy(streams[i]));
    }
    CK(cudaHostUnregister(hptr));
    munmap(hptr, SHM_SIZE);
    close(fd);
    return 0;
}
