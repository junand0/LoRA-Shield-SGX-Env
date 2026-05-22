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

## 9. 산출물 (`/home/user/tee/`)

| 파일 | 역할 |
|---|---|
| `Dockerfile` | `tee-gramine:stage1` 이미지 (nvcr + Gramine 1.9 + SGX/DCAP) |
| `python.manifest.template` | trusted enclave 앱용 Gramine manifest (CPU, `ra_type` 토글) |
| `python-gpu.manifest.template` | GPU passthrough 실험용 (참고용; §7에서 한계 확인) |
| `app/client.py` | enclave 안 trusted 클라이언트 (OTP 마스킹 데모) |
| `app/smoke_dcap.py` | DCAP quote 생성 스모크(§5에서 차단 확인) |
| `app/gpu_test.py` | enclave 내 CUDA 테스트(§7) |
| `proto/worker.py` | untrusted GPU 워커 (일반 CUDA) |
| `proto/run-proto.sh` | 분리형 E2E 실행 스크립트 |
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
