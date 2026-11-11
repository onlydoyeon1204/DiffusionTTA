import gradio as gr
import os
import random
import pandas as pd
import re

# --- 1. 파일 탐색 헬퍼 함수 ---
def find_image(base_dir, case_name, modality=None):
    """
    os.walk를 사용하여 주어진 base_dir 내에서 case_name과 일치하는 첫 번째 이미지 경로를 반환합니다.
    modality가 주어질 경우, 경로에 해당 모달리티가 포함되어 있는지 확인합니다.
    """
    if not os.path.exists(base_dir):
        return None
        
    for root, dirs, files in os.walk(base_dir):
        # modality 조건이 있다면 폴더 경로(root)에 모달리티 이름이 포함되어 있는지 확인
        if modality and modality not in root.lower():
            continue
            
        for file in files:
            if case_name.lower() in file.lower() and file.lower().endswith(('.png', '.jpg', '.jpeg')):
                return os.path.join(root, file)
    return None
def get_auto_case_list(base_dir="original_image"):
    case_set = set()
    if not os.path.exists(base_dir):
        return ["160051860_2_14"] # 폴더가 없을 경우를 대비한 기본값
        
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # '숫자_숫자_숫자' 패턴(예: 160051860_2_14)만 텍스트에서 쏙 뽑아냅니다.
                match = re.search(r'(\d+_\d+_\d+)', file)
                if match:
                    case_set.add(match.group(1))
    
    # 중복을 제거하고 알파벳/숫자 순으로 정렬해서 반환합니다.
    case_list = sorted(list(case_set))
    return case_list if case_list else ["160051860_2_14"]
    
def find_gallery_images(base_dir, case_name, modality=None):
    """
    Traditional TTA 갤러리를 위해 여러 폴더(aug01~aug26)를 탐색하여 이미지 리스트를 반환합니다.
    """
    image_paths = []
    if not os.path.exists(base_dir):
        return image_paths
        
    for root, dirs, files in os.walk(base_dir):
        if modality and modality not in root.lower():
            continue
            
        for file in files:
            if case_name.lower() in file.lower() and file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    
    # 정렬하여 aug01 -> aug26 순서가 유지되도록 함
    return sorted(image_paths)

# --- 2. 메인 처리 로직 ---
def process_case(case_name, modality_display):
    # UI의 모달리티 이름을 폴더명에 맞게 매핑
    modality_map = {"T2WI": "t2", "DWI": "dwi", "ADC": "adc"}
    modality = modality_map.get(modality_display, "t2")
    
    # 이미지 찾기
    original_img = find_image("original_image", case_name, modality)
    baseline_img = original_img # Baseline은 원본과 동일하게 처리
    
    # Diffusion TTA 이미지 찾기
    diff_02 = find_image(os.path.join("Diffusion_TTA", "strength_0.2"), case_name, modality)
    diff_04 = find_image(os.path.join("Diffusion_TTA", "strength_0.4"), case_name, modality)
    diff_06 = find_image(os.path.join("Diffusion_TTA", "strength_0.6"), case_name, modality)
    
    # Traditional TTA 갤러리 이미지 찾기
    trad_gallery = find_gallery_images("Traditional_TTA", case_name, modality)
    
    # 더미 데이터 생성 함수
    def gen_dummy(base, variance):
        return [round(base + random.uniform(-variance, variance), 3) for _ in range(3)]

    # 테이블 1 (예측 성능 지표) 구성
    df_metrics1 = pd.DataFrame({
        "Method": ["Baseline", "Traditional TTA", "Diffusion TTA"],
        "Accuracy": [0.820, 0.845, 0.895],
        "Balanced Accuracy": [0.805, 0.830, 0.880],
        "Sensitivity": [0.790, 0.825, 0.875],
        "Specificity": [0.850, 0.865, 0.915],
        "AUC": [0.860, 0.885, 0.935]
    })
    
    # 테이블 2 (영상 유사도 및 거리 지표) 구성
    df_metrics2 = pd.DataFrame({
        "Method": ["Baseline", "Traditional TTA", "Diffusion TTA"],
        "SSIM": [1.000, 0.824, 0.912],
        "Wasserstein distance": [0.000, 45.2, 12.8],
        "Radiomics distance": [0.000, 18.5, 5.4]
    })

    return original_img, baseline_img, diff_02, diff_04, diff_06, trad_gallery, df_metrics1, df_metrics2

# --- 3. Gradio UI 레이아웃 구성 ---
with gr.Blocks(title="Diffusion Model TTA Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Medical Image Diffusion-based Test-Time Augmentation (TTA) Demo")
    
    with gr.Row():
        # --- 좌측 패널 ---
        with gr.Column(scale=1):
            gr.Markdown("### Input Settings")
            # 자동으로 폴더를 뒤져서 환자 번호 리스트를 만듭니다.
            case_list = get_auto_case_list("original_image")

            # 만들어진 리스트를 드롭다운 메뉴에 쏙 집어넣습니다.
            case_input = gr.Dropdown(choices=case_list, value=case_list[0], label="Case Name")
            modality_input = gr.Dropdown(choices=["T2WI", "DWI", "ADC"], value="T2WI", label="Modality")
            run_btn = gr.Button("결과 확인 (Run)", variant="primary")
            
            gr.Markdown("### Original Image")
            original_output = gr.Image(label="Original", interactive=False)
            
        # --- 우측 패널 ---
        with gr.Column(scale=3):
            # 우측 패널 1: 영상 결과
            gr.Markdown("### Image Results")
        
            gr.Markdown("#### 1. Baseline")
            with gr.Row():
                baseline_output = gr.Image(label="Baseline (Original)", interactive=False)
        
            gr.Markdown("#### 2. Traditional TTA Generated Images (aug01 ~ aug26)")
            trad_gallery_output = gr.Gallery(label="Traditional TTA", show_label=True, elem_id="gallery", columns=[6], rows=[1], object_fit="contain", height="auto")
        
            gr.Markdown("#### 3. Diffusion TTA Images")
            with gr.Row():
                diff_02_output = gr.Image(label="Diffusion (0.2)", interactive=False)
                diff_04_output = gr.Image(label="Diffusion (0.4)", interactive=False)
                diff_06_output = gr.Image(label="Diffusion (0.6)", interactive=False)
            
            # 우측 패널 2: 평가지표
            gr.Markdown("### Evaluation Metrics")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Table 1. Prediction Metrics")
                    table1_output = gr.Dataframe(headers=["Method", "Accuracy", "Balanced Accuracy", "Sensitivity", "Specificity", "AUC"], interactive=False)
                with gr.Column():
                    gr.Markdown("#### Table 2. Similarity & Distance Metrics")
                    table2_output = gr.Dataframe(headers=["Method", "SSIM", "Wasserstein distance", "Radiomics distance"], interactive=False)

    # 버튼 클릭 시 이벤트 매핑
    run_btn.click(
        fn=process_case,
        inputs=[case_input, modality_input],
        outputs=[
            original_output, baseline_output, diff_02_output, diff_04_output, diff_06_output, 
            trad_gallery_output, table1_output, table2_output
        ]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=10000)
