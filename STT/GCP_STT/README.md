### GCP STT PoC (간단/대용량 대응)

이 스크립트는 로컬 오디오 파일을 GCS로 업로드한 뒤 Google Cloud Speech-to-Text로 전사합니다. 대용량 파일(10MB 초과)에도 대응합니다.

---

### 1) 준비
- GCP 프로젝트 + 결제 활성화
- Speech-to-Text API, Cloud Storage API 활성화
- 서비스 계정 키(JSON) 다운로드

PowerShell에서(세션 한정):
```
$env:GOOGLE_APPLICATION_CREDENTIALS = "C:\path\to\service-account.json"
```
혹은 실행 시 `--credentials`로 경로 전달 가능

---

### 2) 설치
```
pip install -r STT/GCP_STT/requirements.txt
```

---

### 3) 버킷 생성(옵션)
gcloud가 없어도 스크립트로 생성 가능:
```
python STT/GCP_STT/main.py --bucket <YOUR_BUCKET> --create-bucket --region ASIA-NORTHEAST3 \
  --credentials C:\path\to\service-account.json --file dummy --prefix stt
```
주의: 버킷 이름은 전 세계 유일해야 합니다.

---

### 4) 실행 예시
- 로컬 파일 업로드 후 전사:
```
python STT/GCP_STT/main.py --file STT/GCP_STT/travel_test.wav --bucket <YOUR_BUCKET> \
  --prefix stt --lang ko-KR --credentials C:\path\to\service-account.json
```

- 이미 업로드된 GCS 파일 전사:
```
python STT/GCP_STT/main.py --gcs-uri gs://<YOUR_BUCKET>/stt/travel_test.wav \
  --lang ko-KR --credentials C:\path\to\service-account.json
```

---

### 5) 결과
- 전사 텍스트는 `STT/GCP_STT/result/transcript.txt`에 저장됩니다.

