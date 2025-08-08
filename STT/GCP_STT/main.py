import argparse
import os
from typing import Optional

from google.cloud import speech
from google.cloud import storage
from google.oauth2 import service_account


def create_speech_client(credentials_path: Optional[str]) -> speech.SpeechClient:
    if credentials_path:
        creds = service_account.Credentials.from_service_account_file(credentials_path)
        return speech.SpeechClient(credentials=creds)
    return speech.SpeechClient()


def create_storage_client(credentials_path: Optional[str]) -> storage.Client:
    if credentials_path:
        creds = service_account.Credentials.from_service_account_file(credentials_path)
        return storage.Client(credentials=creds, project=creds.project_id)
    return storage.Client()


def ensure_bucket(bucket_name: str, location: str, credentials_path: Optional[str]) -> None:
    client = create_storage_client(credentials_path)
    bucket = client.lookup_bucket(bucket_name)
    if bucket is not None:
        print(f"Bucket already exists: gs://{bucket_name}")
        return
    bucket = storage.Bucket(client, name=bucket_name)
    bucket.storage_class = "STANDARD"
    client.create_bucket(bucket, location=location)
    print(f"Created bucket: gs://{bucket_name} (location={location})")


def upload_to_gcs(local_path: str, bucket_name: str, prefix: str, credentials_path: Optional[str]) -> str:
    client = create_storage_client(credentials_path)
    bucket = client.bucket(bucket_name)
    file_name = os.path.basename(local_path)
    blob_path = f"{prefix.strip('/')}/{file_name}" if prefix else file_name
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)
    return f"gs://{bucket_name}/{blob_path}"


def transcribe_gcs(gcs_uri: str, language_code: str, credentials_path: Optional[str]) -> str:
    client = create_speech_client(credentials_path)
    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        language_code=language_code,
        enable_automatic_punctuation=True,
        encoding=speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED,
    )
    operation = client.long_running_recognize(config=config, audio=audio)
    response = operation.result(timeout=1800)
    texts = []
    for result in response.results:
        if result.alternatives:
            texts.append(result.alternatives[0].transcript)
    return " ".join(texts).strip()


def save_text(text: str, out_dir: str = os.path.join(os.path.dirname(__file__), "result")) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "transcript.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GCP STT PoC (GCS 업로드 + 전사)")
    p.add_argument("--file", type=str, help="로컬 오디오 파일 경로")
    p.add_argument("--gcs-uri", type=str, help="이미 업로드된 GCS URI (gs://bucket/path)")
    p.add_argument("--bucket", type=str, help="업로드에 사용할 GCS 버킷명")
    p.add_argument("--prefix", type=str, default="stt", help="업로드 경로 prefix (기본: stt)")
    p.add_argument("--lang", type=str, default="ko-KR", help="언어 코드 (기본: ko-KR)")
    p.add_argument("--credentials", "--credential", dest="credentials", type=str, default=None, help="서비스 계정 키 JSON 경로")
    p.add_argument("--create-bucket", action="store_true", help="버킷이 없으면 생성")
    p.add_argument("--region", type=str, default="ASIA-NORTHEAST3", help="버킷 리전 (기본: ASIA-NORTHEAST3)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.gcs_uri and not args.file:
        raise SystemExit("--file 또는 --gcs-uri 중 하나는 반드시 지정하세요.")

    if args.gcs_uri:
        gcs_uri = args.gcs_uri
    else:
        if not args.bucket:
            raise SystemExit("로컬 파일을 업로드하려면 --bucket 을 지정해야 합니다.")
        if args.create_bucket:
            ensure_bucket(args.bucket, args.region, args.credentials)
        gcs_uri = upload_to_gcs(args.file, args.bucket, args.prefix, args.credentials)
        print(f"Uploaded: {gcs_uri}")

    text = transcribe_gcs(gcs_uri, args.lang, args.credentials)
    print("===== Transcription =====")
    print(text)
    out_path = save_text(text)
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()

