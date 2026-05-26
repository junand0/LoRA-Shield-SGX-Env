# SGX 환경 세팅 작업 정리

> 대상: `snuti-h100-4-001` (Ubuntu 25.10, kernel 6.8.0-intel, SGX2 + TDX, H100 NVL ×4)
> 작업일: 2026-05-22
> 목표: Python AI(LoRA) 워크로드를 SGX(Gramine) 기반 기밀 컴퓨팅 환경에서 실행

---

## 1. 목표와 최종 아키텍처

- **민감 자산**: 학습/추론 데이터 **와** LoRA 가중치 (둘 다 비밀)
- **보호 방식**: enclave(CPU)에서 **OTP 가법 마스킹** → 마스킹된 데이터만 GPU로 전달
- **최종 구조**: **분리형(split)** — Gramine SGX enclave(trusted) + 별도 GPU 워커(untrusted) + IPC
  - GPU를 enclave 안에서 직접 쓰는 방식(ioctl 포워딩)은 **불가능에 가까움**을 실측으로 확인 (→ §8)

```
┌─ enclave (EPC, trusted) ───┐        ┌─ GPU worker (untrusted) ──┐
│ Python + 민감 로직         │        │ 일반 PyTorch + CUDA       │
│ 데이터/LoRA 가중치, OTP    │  TCP   │ (nvcr pytorch, --gpus)    │
│ X' = X + R   ──────────────┼──────▶ │ Y' = X' @ W (H100)        │
│ Y  = Y' − (R@W) ◀──────────┼────────┤ (마스킹된 데이터만 봄)    │
└────────────────────────────┘        └───────────────────────────┘
```

---

## 2. 호스트 환경 (확인된 사실)

| 항목 | 상태 |
|---|---|
| CPU SGX | `sgx`, `sgx_lc`(FLC) — SGX2 + TDX 플랫폼 |
| 커널 드라이버 | in-tree, `/dev/sgx_enclave`, `/dev/sgx_provision`, `/dev/sgx_vepc` |
| EPC 크기 | ~234 MB (EDMM 지원 → 동적 메모리 가능) |
| PSW/DCAP | 풀스택 설치 (`libsgx-urts`, `dcap-ql`, `quote-ex`, ...), aesmd 실행 중 |
| PCCS | 실행 중(localhost:8081), self-signed 인증서, `CachingFillMode: LAZY`, ApiKey 설정됨 |
| GPU | NVIDIA H100 NVL ×4, 드라이버 580.159.03 |
| Docker | 29.4.1, nvidia-container-toolkit 1.19.0 (`--gpus all` 동작) |
| Intel SGX apt 저장소 | **미설정** (기존 패키지는 로컬 .deb로 설치됨) |

---

## 3. Phase 0 — 호스트 베이스라인 보정

1. **그룹 추가**: `usermod -aG sgx,sgx_prv user` → `/dev/sgx_*` 비루트 접근 가능 (재로그인 필요)
2. **QCNL 수정**: `/etc/sgx_default_qcnl.conf`의 `use_secure_cert: true → false`
   - 이유: PCCS가 self-signed 인증서 → QPL이 collateral 수신하려면 필요. 이후 `qgsd`/`aesmd` 재시작.
3. **PCK 등록 확인**: PCCS가 LAZY + ApiKey 설정됨 → "첫 quote 때 자동 등록"으로 판단해 보류 (→ 실제론 §6에서 막힘 발견)

---

## 4. Stage 1 — Gramine enclave 환경 구축 ✅

이미지 `tee-gramine:stage1` = `nvcr.io/nvidia/pytorch:25.11-py3`(noble) + **Gramine 1.9** + SGX/DCAP 런타임.

검증 결과:
- 컨테이너 안에서 `is-sgx-available` → SGX1/SGX2/FLC/AEX-Notify 모두 인식
- **SGX enclave 안에서 Python + 전체 PyTorch import·실행 성공** (CPU)
- python 경로: `/usr/bin/python3.12`, stdlib `/usr/lib/python3.12`, site-packages `/usr/local/lib/python3.12/dist-packages`

### 빌드 시 겪은 이슈 (해결됨)
- **Gramine apt 키 오류** (`NO_PUBKEY 4B8D8EC2F8BE4647`): 일반 `gramine-keyring.gpg`는 2021년 키 → **배포판별 `gramine-keyring-noble.gpg`** 사용으로 해결.
- **manifest 문법(1.9)**: `loader.entrypoint = ...` → **`loader.entrypoint.uri = ...`** (TOML 테이블).

---

## 5. DCAP 원격 증명 — 차단 요인 (미해결, 보류) ⛔

근본 원인 체인 (전부 진단 완료):

```
Gramine DCAP quote 요청
 └→ 호스트 aesm (Gramine은 out-of-proc aesm 필수) → "error 44"
     └→ aesm 로그: [QPL] No certificate data for this platform (0xe011 = SGX_QL_NO_PLATFORM_CERT_DATA)
         └→ PCCS 로그: Intel PCS에 PCK 조회 → 404 "No cache data for this platform"
             └→ Intel에 이 플랫폼 PCK 없음
                 └→ /var/log/mpa_registration.log: 등록 실패, error code 163 (여러 부팅 일관)
                     UEFI: SgxRegistrationConfiguration/Status 변수 존재
```

- **결론**: 이 서버(멀티패키지 플랫폼)의 **SGX 등록이 Intel에 안 됨**. Gramine/Docker/코드 문제 아님 → 호스트/BIOS 레벨.
- **영향 범위**: 원격 증명만 막힘. **enclave 실행/개발에는 무관.**
- **복구 방향(추후)**: 자체 장비 + BIOS 접근 가능 → BIOS SGX manifest 재생성 + `mpa_registration_tool` 재실행 검토.

---

## 6. GPU 베이스라인 ✅

- `docker run --gpus all` → 컨테이너에서 H100 NVL ×4 인식
- `torch.cuda.is_available() == True`, `device_count == 4`
- `/dev/nvidia0-3`, `nvidiactl`, `nvidia-uvm` + 호스트 드라이버 libcuda(580.159.03) 주입 확인

---

## 7. Gramine GPU passthrough 조사 — 한계 확인 ⛔

"전체 앱을 enclave에 넣고 GPU ioctl만 포워딩" 방식을 실측:
- `/dev/nvidia*` 디바이스 enclave 마운트 성공 (`dev:` 문법)
- 라이브러리 경로(`/opt/hpcx`의 libmpi 등), `tmpfs /tmp` 보정 후 **PyTorch 전체가 enclave에서 import됨**
- **그러나 CUDA 초기화 실패**: `CUDA_ERROR_OPERATING_SYSTEM (Error 304)`, `torch.cuda.is_available()=False`

### 왜 막히나
- Gramine은 ioctl 인자 구조체를 enclave 경계 너머로 **deep-copy**해야 함 → 모든 NVIDIA ioctl마다 `sys.ioctl_structs` 정의 필요
- NVIDIA ioctl은 수십 개·비공개·복잡한 중첩 + GPU mmap(UVM); UVM 폴링 문제는 Gramine **소스 수정** 필요 (ref: arxiv 2203.01813)
- 공개된 NVIDIA ioctl manifest 정의 없음 → **연구 과제 수준, config로 불가**

→ 이 결과로 **분리형(§1)** 으로 전환 결정.

---

## 8. Prototype — 분리형 환경 E2E 검증 ✅

`proto/run-proto.sh` 실행 결과:

```
[worker] device=cuda gpus=4
[worker] received W shape=(1024,1024)
[client] connected to GPU worker
[client] gpu round-trip=233.3ms  max_abs_err=6.076e-02
[client] PASS: OTP mask -> GPU -> unmask verified
```

검증된 것: enclave 안 Python 실행 → OTP 생성·마스킹 → TCP IPC → GPU 워커 matmul(H100) → enclave 언마스킹·정확도 확인.

- **주의(실측 발견)**: `max_abs_err ≈ 6e-2` — float32 가법 마스킹의 정밀도 손실(catastrophic cancellation). 운영에서 진짜 OTP 보안 + 정확도를 원하면 **고정소수점/정수환 인코딩** 권장. (마스킹 로직은 사용자 코드 영역)

---

## 8b. IPC 벤치마크 — TCP vs untrusted_shm ✅

같은 워크로드(256KB 페이로드, 64×1024 ⊗ 1024×1024 = H100 matmul, OTP 마스킹·언마스킹 포함)를 200회 라운드트립으로 측정.

| metric | TCP (μs) | SHM (μs) | speedup |
|---|---:|---:|---:|
| mean | 1302 | **475** | **2.7×** |
| p50 | 1278 | 451 | 2.8× |
| p95 | 1377 | 555 | 2.5× |
| **p99** | **7088** | **918** | **7.7×** |
| max | 15950 | 1218 | **13×** |

- 평균 2.7×, 꼬리(p99/max)는 7.7~13× 개선. TCP의 직렬화·소켓 syscall·커널 버퍼링 비용이 통째로 사라짐.
- **분해 측정(`--noop` 추가)으로 IPC 자체 비용 확정:**
  - SHM 256KB noop: mean **198μs** (≒ 데이터 이동 ~193μs + 순수 sync ~5μs)
  - SHM 16B noop: mean **5.1μs** ← 순수 동기화 바닥(폴링+atomic+python)
  - TCP 256KB noop: mean **520μs** → 같은 size에서 SHM은 IPC만 봐도 ~2.6× 빠름
  - GPU 추가 비용(=전체−noop) ≈ 700~1000μs (matmul + cudaMemcpy + sync), run-to-run 변동 큼
  - 즉 SHM 256KB 전체의 IPC 자체 비중은 약 ~200μs이고, 그중 데이터 memcpy/Python wrap이 대부분. 진짜 zero-copy(`np.frombuffer` 뷰 직접 대입)로 더 줄일 여지 있음.
- 메커니즘: Gramine `untrusted_shm` 마운트(`{ type="untrusted_shm", path="/dev/shm", uri="dev:/dev/shm/" }`) + 두 컨테이너 `--ipc=host` → 같은 물리 페이지. 동기화는 x86 TSO 기반 sequence-number 폴링(추가 lock 불필요).
- caveat: 워커 tight spin이라 1코어 100% 점유 — 운영에선 hybrid spin/sleep 권장. 페이로드는 untrusted 메모리지만 분리형 설계상 마스킹된 데이터만 흐르므로 OK.

산출 파일: `proto/bench_worker.py`, `app/bench_client.py`, `proto/run-bench.sh`. manifest에 `/dev/shm` `untrusted_shm` 마운트 + `dev:/dev/shm/` allowed_files 추가됨.

## 9. 산출물 (`/home/user/tee/`)

| 파일 | 역할 |
|---|---|
| `Dockerfile` | `tee-gramine:stage1` 이미지 (nvcr + Gramine 1.9 + SGX/DCAP) |
| `python.manifest.template` | trusted enclave 앱용 Gramine manifest (CPU, `ra_type` 토글) |
| `python-gpu.manifest.template` | GPU passthrough 실험용 (참고용; §7에서 한계 확인) |
| `app/client.py` | enclave 안 trusted 클라이언트 (OTP 마스킹 데모) |
| `app/smoke_dcap.py` | DCAP quote 생성 스모크(§5에서 차단 확인) |
| `app/gpu_test.py` | enclave 내 CUDA 테스트(§7) |
| `proto/worker.py` | untrusted GPU 워커 (TCP, 일반 CUDA) |
| `proto/run-proto.sh` | 분리형 E2E 실행 스크립트 |
| `proto/bench_worker.py` / `app/bench_client.py` | TCP/SHM 양쪽 모드 벤치마크 |
| `proto/run-bench.sh` | TCP vs SHM 벤치 일괄 실행 |
| `run-shell.sh` | enclave 컨테이너 대화형 진입 (전 SGX 배선 포함) |

---

## 10. 실제 코드 연결 방법

1. **Trusted 측**: enclave 코드를 `app/`에 넣고 `gramine-sgx python /workspace/app/<your>.py` 실행. manifest 재사용.
2. **GPU 워커 측**: `proto/worker.py`의 통신 골격(길이-prefix + numpy)은 두고 matmul 자리에 실제 GPU 연산 삽입.
3. **IPC**: TCP(host net)로 검증됨. gRPC/공유메모리로 교체 가능.

### SGX 컨테이너 실행 배선 (요약)
```bash
docker run --rm \
  --device=/dev/sgx_enclave --device=/dev/sgx_provision \
  -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \   # 증명 쓸 때만
  -v /etc/sgx_default_qcnl.conf:/etc/sgx_default_qcnl.conf:ro \  # 증명 쓸 때만
  --network host \                                               # PCCS(127.0.0.1) 도달용
  -v /home/user/tee:/workspace -w /workspace \
  tee-gramine:stage1 ...
```

---

## 8c. Step 1 — cross-process GPU↔enclave handoff (cuStreamWaitValue) ✅

LoRO-on-SGX 디자인의 마스터 가정 검증. enclave가 `/dev/shm/loro_smoke.bin`에 u32 write → worker가 `cudaHostRegister`로 같은 페이지를 pin → GPU 스트림이 `cuStreamWaitValue32`로 polling.

```
[worker] device 0: NVIDIA H100 NVL (cc 9.0)
[worker] iters= 10000 per_handoff= 7.955us
[worker] iters=100000 per_handoff= 6.850us  ← warmup amortized
```

| 메커니즘 | 1회 handoff RTT |
|---|---:|
| in-process `cudaLaunchHostFunc` (원본) | ~1~3 μs |
| **enclave SHM + cuStreamWaitValue (실측)** | **~7 μs** |
| Python tight-spin SHM (이전) | ~200 μs |
| TCP SHM (이전) | ~1300 μs |

→ 이전 Python SHM 대비 ~28× 빠르고, 원본 대비 2~3× 손실 정도. enclave 측을 C로 바꾸면 ~3μs까지. 220 handoff/forward × 7μs = ~1.5ms → **decode 50ms → ~52ms (+4%)** 추정이 실측으로 뒷받침됨.

CUDA-13 gotcha: runtime API `cudaStreamWaitValue32`/`WriteValue32` 심볼은 **삭제됨**. driver API `cuStreamWaitValue32`/`cuStreamWriteValue32`를 `<cuda.h>` + `-lcuda`로 사용. flag enum은 `CU_STREAM_WAIT_VALUE_EQ`.

Manifest 갱신: `/workspace/proto/`를 `sgx.allowed_files`에 추가 (이전엔 `/workspace/app/`만 허용).

산출: `proto/smoke_csv/{worker.cu,enclave.py,run-smoke.sh}`.

## 8d. Step 2 — 데이터 운반 handoff RTT sweep ✅ (2026-05-26)

같은 cross-process 메커니즘(`cuStreamWriteValue`/`WaitValue` on `cudaHostRegister`'d untrusted_shm)으로 페이로드까지 운반. Per-iter 라운드트립 = `cudaMemcpyAsync(D2H)` + signal + signal + `cudaMemcpyAsync(H2D)` + per-iter `cudaStreamSynchronize`. 200 iter씩, single-process sweep.

| payload | mean (μs) | p50 | p95 | bw (GB/s) |
|---:|---:|---:|---:|---:|
| 16 KB | 19.93 | 19.53 | 22.25 | 1.6 |
| 64 KB | 21.46 | 21.25 | 22.22 | 6.1 |
| 256 KB | 36.85 | 36.78 | 38.07 | 14.2 |
| 1 MB | 90.27 | 90.10 | 91.17 | 23.2 |
| 4 MB | 319.53 | 319.77 | 321.74 | 26.3 |
| 16 MB | 1206.72 | 1206.28 | 1209.20 | 27.8 |
| 64 MB | 4825.88 | 4825.52 | 4858.35 | 27.8 |

- ~20μs floor (sync + tiny memcpy). 256KB 이하 = latency-dominated. 1MB 이상 = bandwidth-bound ~28 GB/s.
- p95가 mean 거의 같음 → 매우 안정적.
- session reset 프로토콜: enclave가 W2E가 내려가는 걸 보고 expected를 1로 reset (`prev_w2e > v` 검출). 여러 worker invocation 사이에서 stale enclave state 문제 해결.
- 트러블슈팅 메모: docker stdio 버퍼링이 hang처럼 보이게 함 → 워커 stdout/stderr를 컨테이너 내부 파일로 redirect한 뒤 host에서 읽으면 됨. CUDA 13에서 `cudaStreamWaitValue32`/`WriteValue32` runtime API는 삭제됨 — driver API (`cu*`) + `-lcuda` 필요.

lora_shield 영향 추정 (실측 기반): decode +10~15% (multi-lane으로 추가 단축), prefill +10~15% (대부분 GPU compute에 hide). 원본 in-process 대비 충분히 근접.

산출: `proto/smoke_data/{worker.cu,enclave.py,run-smoke.sh}`.

## 11. 남은 작업

- [ ] **운영 하드닝**: `sgx.debug=false`, `allowed_files` → `trusted_files`(측정)로 전환해 코드/모델을 MRENCLAVE에 고정, 서명키 분리(HSM)
- [ ] **DCAP 등록 복구**(§5): BIOS SGX manifest 재생성 + `mpa_registration_tool` 재실행
- [ ] **실제 코드 연결**: 사용자 trusted 클라이언트 + GPU 워커 연동
- [ ] (선택) 정밀도/보안: 고정소수점·정수환 마스킹 인코딩 검토

---

## 부록 — 핵심 트러블슈팅 빠른 참조

| 증상 | 원인 | 해결 |
|---|---|---|
| `NO_PUBKEY 4B8D8EC2F8BE4647` | 잘못된 Gramine 키링 | `gramine-keyring-noble.gpg` 사용 |
| `Unknown loader.entrypoint format` | Gramine 1.9 문법 변경 | `loader.entrypoint.uri = ...` |
| `AESM service returned error 44` | 플랫폼 PCK 미등록 | Intel SGX 등록 필요(§5, 보류) |
| `No usable temporary directory` | /tmp 없음 | manifest에 `{ type="tmpfs", path="/tmp" }` |
| `libmpi.so.40 not found` | LD_LIBRARY_PATH 누락 | 이미지 기본 경로 + `/opt/hpcx/ompi/lib` 추가, `/opt` 마운트 |
| `CUDA Error 304` | NVIDIA ioctl 미지원 | Gramine GPU passthrough 한계 → 분리형 전환(§7) |
