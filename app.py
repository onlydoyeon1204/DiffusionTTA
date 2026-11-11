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

'''
def get_auto_case_list(base_dir="original_image"):
    case_set = set()
    if not os.path.exists(base_dir):
        return ["160051860_2_14"] # 폴더가 없을 경우를 대비한 기본값
        
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # '숫자_숫자_숫자' 패턴(예: 160051860_2_14)만 텍스트에서 쏙 뽑아냅니다.
                # 맨 앞 숫자가 최소 5자리 이상(예: 160051860)인 경우만 환자 번호로 인식하도록 강화
                match = re.search(r'(\d{5,}_\d+_\d+)', file)
                if match:
                    case_set.add(match.group(1))
    
    # 중복을 제거하고 알파벳/숫자 순으로 정렬해서 반환합니다.
    case_list = sorted(list(case_set))
    return case_list if case_list else ["160051860_2_14"] 
    
    '''
    
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

    # 1. CSV에서 Diffusion TTA 실제 데이터 읽어오기
    try:
        metrics_df = pd.read_csv("tta_summary_metrics.csv")
        # UI에서 선택된 모달리티(t2, dwi, adc)에 해당하는 행만 필터링
        mod_df = metrics_df[metrics_df['modality'] == modality]
    
        if not mod_df.empty:
            # 해당 모달리티의 5가지 지표 평균값을 소수점 3자리까지 추출
            diff_acc = round(mod_df['accuracy'].mean(), 3)
            diff_bacc = round(mod_df['balanced_accuracy'].mean(), 3)
            diff_sens = round(mod_df['sensitivity'].mean(), 3)
            diff_spec = round(mod_df['specificity'].mean(), 3)
            diff_auc = round(mod_df['auc'].mean(), 3)
        else:
            diff_acc, diff_bacc, diff_sens, diff_spec, diff_auc = [0.0, 0.0, 0.0, 0.0, 0.0]
    except Exception as e:
        # 파일이 없거나 에러가 날 경우 임시 데이터
        diff_acc, diff_bacc, diff_sens, diff_spec, diff_auc = ["---", "---", "---", "---", "---"]

    # 2. 테이블 1 (예측 성능 지표) 구성 (Diffusion TTA에 실제 데이터 반영)
    df_metrics1 = pd.DataFrame({
        "Method": ["Baseline", "Traditional TTA", "Diffusion TTA"],
        "Accuracy": ["0.771","0.803", diff_acc],
        "Balanced Accuracy": ["0.761","0.813", diff_bacc],
        "Sensitivity": ["0.707","0.746", diff_sens],
        "Specificity": ["0.826","0.873", diff_spec],
        "AUC": ["0.751","0.765", diff_auc]
    })
    
    # 테이블 2 (영상 유사도 및 거리 지표) 실제 데이터 연동

    # 1. Baseline (자기 자신과 비교하므로 고정값)
    base_ssim, base_wd, base_rd = 1.000, 0.000, 0.000

    # 2. Traditional TTA 실제 데이터 읽어오기
    try:
        df_trad = pd.read_csv("raw_augmented_metrics_by_image.csv")
        # 선택된 환자 번호(case_name)가 포함되고, 선택된 모달리티와 일치하는 데이터만 필터링
        sub_trad = df_trad[(df_trad['case_id'].str.contains(case_name, case=False)) & (df_trad['modality'] == modality)]
        if not sub_trad.empty:
            trad_ssim = round(sub_trad['ssim'].mean(), 3)
            trad_wd = round(sub_trad['wasserstein_distance'].mean(), 3)
            trad_rd = round(sub_trad['radiomics_distance'].mean(), 3)
        else:
            trad_ssim, trad_wd, trad_rd = [0.0, 0.0, 0.0]
    except Exception:
        trad_ssim, trad_wd, trad_rd = ["---", "---", "---"] # 에러 시 더미 데이터

    # 3. Diffusion TTA 실제 데이터 읽어오기
    try:
        df_diff = pd.read_csv("metrics_by_set.csv")
        # 선택된 환자 번호(case_name)가 포함된 데이터만 필터링
        sub_diff = df_diff[df_diff['case_id'].str.contains(case_name, case=False)]
        if not sub_diff.empty:
            # 컬럼명이 t2_ssim, dwi_wasserstein_distance 형태로 되어 있으므로 동적으로 변수 매핑
            diff_ssim = round(sub_diff[f'{modality}_ssim'].mean(), 3)
            diff_wd = round(sub_diff[f'{modality}_wasserstein_distance'].mean(), 3)
            diff_rd = round(sub_diff[f'{modality}_radiomics_distance'].mean(), 3)
        else:
            diff_ssim, diff_wd, diff_rd = [0.0, 0.0, 0.0]
    except Exception:
        diff_ssim, diff_wd, diff_rd = ["---", "---", "---"] # 에러 시 더미 데이터

    # 추출한 평균값들로 최종 테이블(Dataframe) 생성
    df_metrics2 = pd.DataFrame({
        "Method": ["Baseline", "Traditional TTA", "Diffusion TTA"],
        "SSIM": [base_ssim, trad_ssim, diff_ssim],
        "Wasserstein distance": [base_wd, trad_wd, diff_wd],
        "Radiomics distance": [base_rd, trad_rd, diff_rd]
    })

    return original_img, baseline_img, diff_02, diff_04, diff_06, trad_gallery, df_metrics1, df_metrics2

# --- 3. Gradio UI 레이아웃 구성 ---
with gr.Blocks(title="Diffusion Model TTA Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Medical Image Diffusion Test-Time Augmentation (TTA)")
    
    with gr.Row():
        # --- 좌측 패널 ---
        with gr.Column(scale=1):
            gr.Markdown("## [Input Settings]")
            
            # 사진 2장(g1, g2 폴더)에서 추출한 52명 환자 번호 전체 명단
            case_list = [
                "160051860_2_14", "160051860_3_13", "160053159_2_9", "160056308_1_10", "160056308_2_7",
                "170001022_1_13", "170001022_2_9", "170004726_1_14", "170008345_15", "170009104_14",
                "170014714_13", "170016169_15", "170018993_12", "170020388_12", "170027851_10",
                "170031928_8", "170035899_13", "170038426_15", "170038657_11", "170042537_2_10",
                "170042797_14", "170043280_1_15", "170043280_2_12", "170046491_11", "170048486_14",
                "170054561_16", "170056654_15", "170057519_11", "170059655_12", "180003768_17",
                "180005707_11", "180006738_12", "180007060_2_14", "180007060_3_13", "180007995_13",
                "180008861_13", "180011417_12", "180012048_14", "180012596_14", "180013002_11",
                "180013383_11", "180015672_13", "180015759_13", "180016858_12", "180017003_13",
                "180018777_12", "180019108_13", "180020049_15", "180020484_13", "180021705_1_12",
                "180021705_2_12", "180027879_11"
            ]

            # 52개 환자 번호를 드롭다운 메뉴에 표시
            case_input = gr.Dropdown(
                choices=case_list,
                value=case_list[0],
                label="Case Name"
            )

            modality_input = gr.Dropdown(choices=["T2WI", "DWI", "ADC"], value="T2WI", label="Modality")
            run_btn = gr.Button("결과 확인 (Run)", variant="primary")
            
            gr.Markdown("## [Original Image]")
            original_output = gr.Image(label="Original", interactive=False)
            
        # --- 우측 패널 ---
        with gr.Column(scale=3):
            # 우측 패널 1: 영상 결과
            gr.Markdown("## [Image Results]")
        
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
            
            gr.Markdown("---")

            gr.Markdown("<div style='height: 20px;'></div>")

            gr.Markdown("## [Evaluation Metrics]")

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
