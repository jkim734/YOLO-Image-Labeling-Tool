# YOLO Image Labeling Tool

YOLO Image Labeling Tool is a graphical application for labeling images with bounding boxes and class information, supporting YOLO format. It is designed for fast annotation, easy navigation, and efficient workflow for computer vision datasets.

## 주요 기능

- **이미지 폴더/단일 파일 불러오기**: 여러 이미지를 한 번에 불러와서 빠르게 라벨링 가능
- **바운딩 박스 생성/삭제/수정**: 마우스로 드래그하여 박스 생성, 리스트/캔버스에서 삭제
- **클래스 관리**: 클래스 추가/수정, 라벨별 클래스 선택
- **자동 저장 모드**: 다음/이전 이미지 이동 시 자동으로 라벨 및 회전 정보 저장
- **이미지 회전**: 90도 단위 및 마우스 드래그로 섬세하게 회전
- **썸네일 사이드바**: 이미지 썸네일을 스크롤/키보드/트랙패드로 탐색 가능
- **라벨 텍스트 파일 자동 관리**: 라벨 삭제 시 txt 파일도 자동 삭제

## 설치 방법

1. **Python 3.8 이상 설치**
2. **필수 패키지 설치**

```bash
pip install -r requirements.txt
```

3. **프로젝트 다운로드**

```bash
git clone https://github.com/yourusername/YOLO-Image-Labeling-Tool.git
cd YOLO-Image-Labeling-Tool
```

## 실행 방법

```bash
python labelling.py
```

## 사용법

### 기본 라벨링
- 이미지를 불러오면 캔버스에 표시됩니다.
- 마우스로 드래그하여 바운딩 박스 생성
- 클래스 선택 후 박스 생성 시 해당 클래스가 할당됨
- 박스 위 텍스트로 클래스명 표시
- 박스 삭제: 캔버스에서 마우스 오른쪽 버튼 클릭 또는 라벨 리스트에서 더블 클릭

### 이미지 탐색
- **이전/다음**: 하단 버튼 또는 좌/우 방향키
- **썸네일 사이드바**: 마우스 휠, 트랙패드 두 손가락 스와이프, 위/아래 방향키, 스크롤 바로 탐색

### 자동 저장 모드
- 우측 '자동 저장 모드' 체크박스 활성화 시, 이미지 이동 시 라벨 및 회전 정보 자동 저장
- 라벨이 모두 삭제되면 txt 파일도 자동 삭제

### 이미지 회전
- 우측 회전 버튼(왼쪽/오른쪽 90도)
- 캔버스에서 마우스 드래그로 섬세하게 회전

### 클래스 관리
- '클래스 설정' 버튼 클릭 후 텍스트로 클래스명 입력 및 저장

## 단축키

| 기능                | 단축키           |
|---------------------|------------------|
| 이전 이미지         | ← (Left Arrow)   |
| 다음 이미지         | → (Right Arrow)  |
| 라벨링 모드 전환    | W                |
| 회전 모드 전환      | R                |
| 썸네일 스크롤       | ↑/↓, 마우스 휠, 트랙패드, 스크롤 바 |

## 라벨 파일 포맷
- 각 이미지와 동일한 이름의 .txt 파일에 저장
- 각 줄: `class_id x_center y_center width height` (YOLO format, float)

## 개발 및 기여
- 이 저장소는 누구나 자유롭게 포크/기여할 수 있습니다.
- 버그/기능 요청은 이슈로 남겨주세요.

## 라이선스
MIT License

---
문의: your.email@example.com

