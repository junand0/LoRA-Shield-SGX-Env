// Smoke test for cuStreamWriteValue32 / cuStreamWaitValue32 with
// cudaHostRegister'd memory shared across a process boundary (via /dev/shm
// mmap also mounted into a Gramine SGX enclave as untrusted_shm).
//
// In CUDA 13 the runtime-API wrappers were removed; we use the driver API
// (cu*) for the stream value ops and the runtime API (cuda*) for the rest.

#include <cuda.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define SHM_PATH        "/dev/shm/loro_smoke.bin"
#define SHM_SIZE        4096
#define OFF_W2E         0
#define OFF_E2W         4
#define OFF_SHUTDOWN    8
#define OFF_READY      12

#define CK(expr) do {                                                          \
    cudaError_t _e = (expr);                                                   \
    if (_e != cudaSuccess) {                                                   \
        fprintf(stderr, "CUDA runtime err %s:%d: %s\n", __FILE__, __LINE__,    \
                cudaGetErrorString(_e));                                       \
        exit(1);                                                               \
    }                                                                          \
} while (0)

#define CKD(expr) do {                                                         \
    CUresult _r = (expr);                                                      \
    if (_r != CUDA_SUCCESS) {                                                  \
        const char *msg = NULL; cuGetErrorString(_r, &msg);                    \
        fprintf(stderr, "CUDA driver err %s:%d: %s\n", __FILE__, __LINE__,     \
                msg ? msg : "?");                                              \
        exit(1);                                                               \
    }                                                                          \
} while (0)

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

int main(int argc, char **argv) {
    int iters  = (argc > 1) ? atoi(argv[1]) : 10000;
    int warmup = (argc > 2) ? atoi(argv[2]) : 100;

    int fd = open(SHM_PATH, O_RDWR);
    if (fd < 0) { perror("open shm"); return 1; }

    void *hptr = mmap(NULL, SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (hptr == MAP_FAILED) { perror("mmap"); return 1; }

    CK(cudaHostRegister(hptr, SHM_SIZE,
                        cudaHostRegisterPortable | cudaHostRegisterMapped));

    // Force a runtime primary context to be created on the current device.
    CK(cudaFree(0));

    void *dptr = NULL;
    CK(cudaHostGetDevicePointer(&dptr, hptr, 0));

    int dev = 0;
    CK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, dev));
    printf("[worker] device %d: %s (cc %d.%d)\n", dev, prop.name, prop.major, prop.minor);

    // Wait for enclave ready.
    volatile uint32_t *h_ready = (volatile uint32_t *)((char *)hptr + OFF_READY);
    printf("[worker] waiting for enclave ready...\n");
    fflush(stdout);
    for (int i = 0; *h_ready == 0 && i < 600; i++) usleep(100000);
    if (*h_ready != 1) {
        fprintf(stderr, "[worker] enclave never signaled ready\n");
        return 3;
    }
    printf("[worker] enclave ready; iters=%d warmup=%d\n", iters, warmup);

    cudaStream_t s;
    CK(cudaStreamCreate(&s));

    CUdeviceptr w2e_dev = (CUdeviceptr)((uintptr_t)dptr + OFF_W2E);
    CUdeviceptr e2w_dev = (CUdeviceptr)((uintptr_t)dptr + OFF_E2W);

    uint32_t seq = 1;
    // Warmup.
    for (int i = 0; i < warmup; i++, seq++) {
        CKD(cuStreamWriteValue32((CUstream)s, w2e_dev, seq, 0));
        CKD(cuStreamWaitValue32 ((CUstream)s, e2w_dev, seq, CU_STREAM_WAIT_VALUE_EQ));
    }
    CK(cudaStreamSynchronize(s));

    // Measured loop.
    double t0 = now_us();
    for (int i = 0; i < iters; i++, seq++) {
        CKD(cuStreamWriteValue32((CUstream)s, w2e_dev, seq, 0));
        CKD(cuStreamWaitValue32 ((CUstream)s, e2w_dev, seq, CU_STREAM_WAIT_VALUE_EQ));
    }
    CK(cudaStreamSynchronize(s));
    double t1 = now_us();

    double total = t1 - t0;
    double per   = total / (double)iters;
    printf("[worker] iters=%d total=%.1fus per_handoff=%.3fus (= %.0fns)\n",
           iters, total, per, per * 1000.0);

    // Signal shutdown.
    volatile uint32_t *h_shut = (volatile uint32_t *)((char *)hptr + OFF_SHUTDOWN);
    *h_shut = 0xFFFFFFFFu;

    CK(cudaStreamDestroy(s));
    CK(cudaHostUnregister(hptr));
    munmap(hptr, SHM_SIZE);
    close(fd);
    return 0;
}
