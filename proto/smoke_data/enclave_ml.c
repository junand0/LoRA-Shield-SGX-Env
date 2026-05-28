/*
 * Step 4: multi-lane enclave responder. One pthread per lane, each running
 * the same session-aware protocol on its own header/flags. Lane headers are
 * laid out 64 bytes apart (cache-line) at the start of the shared file.
 */
#define _GNU_SOURCE
#include <fcntl.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define SHM_PATH        "/dev/shm/loro_smoke_ml.bin"
#define SHM_SIZE        (256ULL * 1024 * 1024)
#define LANE_HDR_SIZE   64
#define MAX_LANES       16
#define HDR_BYTES       4096                            // start of payload area
#define OFF_GLOBAL_SHUTDOWN  (LANE_HDR_SIZE * MAX_LANES)   // a single global shutdown after lane headers

// Per-lane header offsets within a lane's 64-byte slot:
#define LANE_OFF_W2E    0
#define LANE_OFF_E2W    4
#define LANE_OFF_READY  8
#define LANE_OFF_RESET  12   // optional — currently unused, kept for future

static inline void cpu_pause(void) {
#if defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__("pause");
#endif
}

static void *base_ptr;
static int n_lanes;
static volatile int g_shutdown;

typedef struct {
    int lane_id;
    uint64_t served;
    uint32_t sessions;
} lane_state_t;

static void *lane_thread(void *arg) {
    lane_state_t *st = (lane_state_t *)arg;
    volatile uint32_t *hdr = (volatile uint32_t *)((char *)base_ptr + st->lane_id * LANE_HDR_SIZE);

    hdr[LANE_OFF_READY / 4] = 1;

    uint32_t expected = 1;
    uint32_t prev_w2e = 0;

    while (!g_shutdown) {
        uint32_t v = hdr[LANE_OFF_W2E / 4];
        if (v == 0xFFFFFFFFu) break;
        if (v != 0 && v < prev_w2e) {
            expected = 1;
            hdr[LANE_OFF_E2W / 4] = 0;
            st->sessions++;
        }
        prev_w2e = v;
        if (v == expected) {
            hdr[LANE_OFF_E2W / 4] = expected;
            expected++;
            st->served++;
        } else {
            cpu_pause();
        }
    }
    return NULL;
}

int main(int argc, char **argv) {
    n_lanes = (argc > 1) ? atoi(argv[1]) : 1;
    if (n_lanes < 1 || n_lanes > MAX_LANES) {
        fprintf(stderr, "[enclave] bad n_lanes=%d (1..%d)\n", n_lanes, MAX_LANES);
        return 1;
    }

    int fd = open(SHM_PATH, O_RDWR);
    if (fd < 0) { perror("[enclave] open"); return 1; }
    base_ptr = mmap(NULL, SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base_ptr == MAP_FAILED) { perror("[enclave] mmap"); return 1; }

    // Zero lane headers.
    memset(base_ptr, 0, LANE_HDR_SIZE * MAX_LANES + 16);

    fprintf(stderr, "[enclave] starting n_lanes=%d\n", n_lanes); fflush(stderr);

    pthread_t threads[MAX_LANES];
    lane_state_t states[MAX_LANES];
    for (int i = 0; i < n_lanes; i++) {
        states[i].lane_id = i;
        states[i].served = 0;
        states[i].sessions = 0;
        if (pthread_create(&threads[i], NULL, lane_thread, &states[i]) != 0) {
            perror("[enclave] pthread_create");
            return 2;
        }
    }

    // Main thread: poll global shutdown flag.
    volatile uint32_t *gshut = (volatile uint32_t *)((char *)base_ptr + OFF_GLOBAL_SHUTDOWN);
    while (1) {
        if (*gshut == 0xFFFFFFFFu) { g_shutdown = 1; break; }
        usleep(50000);   // 50 ms — coarse, fine for shutdown detection
    }

    // Signal each lane thread to exit by writing the magic shutdown to its W2E.
    for (int i = 0; i < n_lanes; i++) {
        volatile uint32_t *hdr = (volatile uint32_t *)((char *)base_ptr + i * LANE_HDR_SIZE);
        hdr[LANE_OFF_W2E / 4] = 0xFFFFFFFFu;
    }
    for (int i = 0; i < n_lanes; i++) pthread_join(threads[i], NULL);

    uint64_t total = 0; uint32_t total_sess = 0;
    for (int i = 0; i < n_lanes; i++) {
        fprintf(stderr, "[enclave] lane %d: served=%" PRIu64 " sessions=%u\n",
                i, states[i].served, states[i].sessions);
        total += states[i].served;
        total_sess += states[i].sessions;
    }
    fprintf(stderr, "[enclave] shutdown total_served=%" PRIu64 " total_sessions=%u\n",
            total, total_sess);
    fflush(stderr);

    munmap(base_ptr, SHM_SIZE);
    close(fd);
    return 0;
}
