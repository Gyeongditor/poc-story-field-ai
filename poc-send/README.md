## POC: 이미지 + 스토리 멀티파트 업로드

### 1) 설치

```bash
python -m pip install -r poc-send/requirements.txt
```

### 2) 서버 실행

```bash
python poc-send/server.py
# 또는
python -m uvicorn poc-send.server:app --reload --port 8000
```

### 3) 클라이언트로 전송

```bash
# 간단 업로드 (기존 방식)
python poc-send/client.py simple --endpoint http://127.0.0.1:8000/upload \
  --images-dir IMAGE/generated_images \
  --story-file integration_test/result/gen_story/story.txt \
  --title "여행 이야기" --author "chan" --meta '{"project":"poc"}'

# 구조화된 동화 업로드 (uuid, 제목, 썸네일, 페이지 텍스트/이미지)
python poc-send/client.py story --endpoint http://127.0.0.1:8000/upload-story \
  --uuid 123e4567-e89b-12d3-a456-426614174000 \
  --title "여행 이야기" \
  --thumbnail IMAGE/generated_images/page_1.png \
  --page-texts STORY/StoryLine/generated_story.txt \
  --page-images-dir IMAGE/generated_images
```

스토리 파일을 생략하면, 리포지토리 내에서 몇 가지 기본 경로를 자동 탐색합니다.

### 4) 스프링 백엔드로 포워딩 (/forward-story → Spring /upload)

환경 변수로 스프링 주소를 설정합니다(기본: http://127.0.0.1:8080).

```powershell
$env:SPRING_BASE_URL="http://127.0.0.1:8080"
```

클라이언트에서 `/forward-story`로 전송하면 FastAPI가 Spring의 `/upload`로 멀티파트를 그대로 포워딩합니다.

```bash
python poc-send/client.py story --endpoint http://127.0.0.1:8000/forward-story \
  --uuid 123e4567-e89b-12d3-a456-426614174000 \
  --title "여행 이야기" \
  --thumbnail IMAGE/generated_images/page_1.png \
  --page-texts STORY/StoryLine/generated_story.txt \
  --page-images-dir IMAGE/generated_images
```

PowerShell 원라이너(임의 UUID/제목 자동 생성):

```powershell
powershell -NoProfile -Command "$uuid=[guid]::NewGuid().ToString(); $title='자동 생성된 동화 ' + (Get-Date -Format 'yyyyMMdd-HHmmss'); python poc-send/client.py story --endpoint http://127.0.0.1:8000/forward-story --uuid $uuid --title $title --thumbnail integration_test/result/gen_img/page_1.png --page-texts poc-send/result/gen_story/story.txt --page-images-dir integration_test/result/gen_img; Write-Host ('UUID: ' + $uuid); Write-Host ('TITLE: ' + $title)"
```

curl 예시(필드명 참고):

```bash
curl -X POST http://127.0.0.1:8000/forward-story \
  -F "uuid=123e4567-e89b-12d3-a456-426614174000" \
  -F "title=여행 이야기" \
  -F "thumbnail=@integration_test/result/gen_img/page_1.png;type=image/png" \
  -F "page_texts=1페이지 내용" \
  -F "page_texts=2페이지 내용" \
  -F "page_images=@integration_test/result/gen_img/page_1.png;type=image/png" \
  -F "page_images=@integration_test/result/gen_img/page_2.png;type=image/png"
```

주의: Spring이 미기동이면 502로 에러 JSON이 반환됩니다.

### 5) 확인 방법(로컬 저장 경로 점검)

로컬 저장 엔드포인트(`/upload-story`) 사용 시 저장 구조:

```
poc-send/uploads/{uuid}/
  ├─ thumbnail.png
  └─ pages/
     ├─ 001.txt, 001.png
     ├─ 002.txt, 002.png
     └─ ...
```

PowerShell 예시:

```powershell
$uuid="11111111-2222-3333-4444-555555555555"
Get-ChildItem poc-send/uploads/$uuid
Get-ChildItem poc-send/uploads/$uuid/pages
Get-Content poc-send/uploads/$uuid/pages/001.txt | Out-Host
```

### 응답

서버는 저장된 파일 경로와 개요를 JSON으로 반환합니다.


