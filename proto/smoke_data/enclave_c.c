/*
 * Step 3: C-based enclave responder. Same protocol as enclave.py but native.
 *
 * Tight spin on W2E flag, write E2W on match. Detects session reset (W2E goes
 * down) and resets expected=1. Detects shutdown (W2E==0xFFFFFFFF).
 *
 * Built to drive the Step 2 handoff sweep with lower polling overhead than
 * Python (Python per-iter cost ~us; C ~ns).
 */
#define _GNU_SOURCE
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define SHM_PATH        "/dev/shm/loro_smoke_data.bin"
#define SHM_SIZE        (256ULL * 1024 * 1024)
#define OFF_W2E         0
#define OFF_E2W         4
#define OFF_SHUTDOWN    8
#define OFF_READY      12

static inline void cpu_pause(void) {
#if defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__("pause");
#endif
}

int main(void) {
    int fd = open(SHM_PATH, O_RDWR);
    if (fd < 0) { perror("[enclave] open"); return 1; }
    void *p = mmap(NULL, SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) { perror("[enclave] mmap"); return 1; }

    volatile uint32_t *w2e   = (volatile uint32_t *)((char *)p + OFF_W2E);
    volatile uint32_t *e2w   = (volatile uint32_t *)((char *)p + OFF_E2W);
    volatile uint32_t *ready = (volatile uint32_t *)((char *)p + OFF_READY);

    memset(p, 0, 4096);
    *ready = 1;

    fprintf(stderr, "[enclave] ready; C tight spin (pause)\n");
    fflush(stderr);

    uint32_t expected = 1;
    uint32_t prev_w2e = 0;
    uint32_t sessions = 0;
    uint64_t served = 0;

    while (1) {
        uint32_t v = *w2e;
        if (v == 0xFFFFFFFFu) break;
        if (v != 0 && v < prev_w2e) {
            expected = 1;
            *e2w = 0;
            sessions++;
            fprintf(stderr, "[enclave] session reset #%u: prev_w2e=%u new_w2e=%u\n",
                    sessions, prev_w2e, v);
            fflush(stderr);
        }
        prev_w2e = v;
        if (v == expected) {
            *e2w = expected;
            expected++;
            served++;
        } else {
            cpu_pause();
        }
    }

    fprintf(stderr, "[enclave] shutdown; sessions=%u served=%" PRIu64 "\n", sessions, served);
    fflush(stderr);
    munmap(p, SHM_SIZE);
    close(fd);
    return 0;
}
