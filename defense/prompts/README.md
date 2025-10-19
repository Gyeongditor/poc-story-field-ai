# 적대적 프롬프트 방어 시스템

이 폴더는 적대적 프롬프트(Adversarial Prompts)에 대한 방어 기능을 제공하는 프롬프트 파일들을 포함합니다.

## 파일 구조

### 1. system_prompts.json
시스템 프롬프트들을 저장하는 파일입니다.

- `defense_system_prompt`: 기본 방어 시스템 프롬프트
- `story_creation_system`: 동화 생성용 시스템 프롬프트  
- `formal_conversion_system`: 문체 변환용 시스템 프롬프트

### 2. user_prompts.json
사용자 프롬프트 템플릿들을 저장하는 파일입니다.

- `safe_story_prompt`: 안전한 동화 생성용 프롬프트
- `safe_image_prompt`: 안전한 이미지 생성용 프롬프트
- `safe_formal_conversion`: 안전한 문체 변환용 프롬프트
- `defense_check_prompt`: 적대적 프롬프트 탐지용 프롬프트

### 3. adversarial_defense_rules.json
방어 규칙들을 정의하는 파일입니다.

- `blocked_patterns`: 차단할 패턴들의 목록
- `safety_keywords`: 안전한 키워드들의 목록
- `response_templates`: 부적절한 내용 감지 시 사용할 응답 템플릿

## 사용법

### def_main.py 실행
```bash
cd integration_test
python def_main.py
```

### 주요 기능

1. **적대적 프롬프트 탐지**: 사용자 입력에서 부적절한 패턴을 자동으로 탐지
2. **안전한 프롬프트 생성**: 시스템 프롬프트와 사용자 프롬프트를 조합하여 안전한 요청 생성
3. **입력 정화**: 부적절한 내용을 건전한 내용으로 자동 변환
4. **안전한 콘텐츠 생성**: 동화, 이미지, 텍스트 변환 시 안전성 보장

### 방어 메커니즘

- **패턴 기반 탐지**: 미리 정의된 부적절한 패턴들을 탐지
- **시스템 프롬프트 보호**: 각 기능별로 안전한 시스템 프롬프트 적용
- **템플릿 기반 생성**: 안전한 프롬프트 템플릿을 사용하여 일관된 안전성 보장
- **대체 응답**: 부적절한 요청 시 안전한 대체 응답 제공

## 커스터마이징

### 새로운 차단 패턴 추가
`adversarial_defense_rules.json`의 `blocked_patterns` 배열에 새로운 패턴을 추가할 수 있습니다.

### 새로운 안전 키워드 추가
`adversarial_defense_rules.json`의 `safety_keywords` 배열에 새로운 키워드를 추가할 수 있습니다.

### 새로운 응답 템플릿 추가
`adversarial_defense_rules.json`의 `response_templates` 객체에 새로운 템플릿을 추가할 수 있습니다.

## 주의사항

- 프롬프트 파일들은 UTF-8 인코딩으로 저장되어야 합니다.
- JSON 파일의 형식을 변경할 때는 문법 오류가 없도록 주의하세요.
- 새로운 방어 규칙을 추가한 후에는 시스템을 재시작해야 합니다.
