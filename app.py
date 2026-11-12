import gradio as gr
import os
import random
import pandas as pd
import re

# --- 1. 파일 탐색 헬퍼 함수 ---
def find_image(base_dir, case_name, modality=None):
    if not os.path.exists(base_dir):
        return None
        
    for root, dirs, files in os.walk(base_dir):
        if modality and modality not in root.lower():
            continue
            
        for file in files:
            if case_name.lower() in file.lower() and file.lower().endswith(('.png', '.jpg', '.jpeg')):
                return os.path.join(root, file)
    return None

def find_gallery_images(base_dir, case_name, modality=None):
    image_paths = []
    if not os.path.exists(base_dir):
        return image_paths
        
    for root, dirs, files in os.walk(base_dir):
        if modality and modality not in root.lower():
            continue
            
        for file in files:
            if case_name.lower() in file.lower() and file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    
    return sorted(image_paths)

# --- 2. 메인 처리 로직 ---
def process_case(case_name, modality_display):
    # UI의 모달리티 이름을 폴더명에 맞게 매핑
    modality_map = {"T2WI": "t2", "DWI": "dwi", "ADC": "adc"}
    modality = modality_map.get(modality_display, "t2")
    
    # 이미지 찾기
    original_img = find_image("original_image", case_name, modality)
    baseline_img = original_img
    
    # Diffusion TTA 이미지 찾기
    diff_02 = find_image(os.path.join("Diffusion_TTA", "strength_0.2"), case_name, modality)
    diff_04 = find_image(os.path.join("Diffusion_TTA", "strength_0.4"), case_name, modality)
    diff_06 = find_image(os.path.join("Diffusion_TTA", "strength_0.6"), case_name, modality)
    
    # Traditional TTA 갤러리 이미지 찾기
    trad_gallery = find_gallery_images("Traditional_TTA", case_name, modality)

    # 예측 점수(Prediction Scores) 추출
    base_score, trad_score, diff_score = "---", "---", "---"
    try:
        # Baseline 점수 읽기
        df_base = pd.read_excel("baseline_scores.xlsx")
        base_row = df_base[df_base['case_id'].str.contains(case_name, case=False, na=False)]
        if not base_row.empty:
            b_vals = base_row.iloc[0][['baseline_score_seed1', 'baseline_score_seed2', 'baseline_score_seed3', 'baseline_score_seed4', 'baseline_score_seed5']]
            base_score = f"{b_vals.mean():.3f}"
            row_idx = base_row.index[0] 
            
            # Traditional TTA 점수 읽기
            df_trad = pd.read_excel("traditionalTTA_scores.xlsx")
            if len(df_trad) > row_idx:
                t_vals = df_trad.iloc[row_idx][['tta_score_seed1', 'tta_score_seed2', 'tta_score_seed3', 'tta_score_seed4', 'tta_score_seed5']]
                trad_score = f"{t_vals.mean():.3f}"
        
        # Diffusion TTA 점수 읽기
        df_diff = pd.read_excel("diffusionTTA_scores.xlsx", sheet_name=0)
        diff_row = df_diff[df_diff['case_id'].str.contains(case_name, case=False, na=False)]
        if not diff_row.empty:
            d_vals = diff_row.iloc[0][['diffusion_score_seed1', 'diffusion_score_seed2', 'diffusion_score_seed3', 'diffusion_score_seed4', 'diffusion_score_seed5']]
            diff_score = f"{d_vals.mean():.3f}"
    except Exception:
        pass 
    
    # 테이블 1 (예측 성능 지표) 구성
    df_metrics1 = pd.DataFrame({
        "Method": ["Baseline", "Traditional TTA", "Diffusion TTA"],
        "Accuracy": ["0.77±0.01","0.80±0.03", "0.84±0.01"],
        "Balanced Accuracy": ["0.76±0.01","0.81±0.03", "0.84±0.01"],
        "Sensitivity": ["0.70±0.07","0.74±0.06", "0.86±0.02"],
        "Specificity": ["0.82±0.06","0.87±0.03", "0.82±0.03"],
        "AUC": ["0.75±0.01","0.76±0.05", "0.80±0.01"]
    })
    
    # 테이블 2 (영상 유사도 및 거리 지표) 실제 데이터 연동
    base_ssim, base_wd, base_rd = 1.000, 0.000, 0.000

    try:
        df_trad = pd.read_csv("raw_augmented_metrics_by_image.csv")
        sub_trad = df_trad[(df_trad['case_id'].str.contains(case_name, case=False)) & (df_trad['modality'] == modality)]
        if not sub_trad.empty:
            trad_ssim = round(sub_trad['ssim'].mean(), 3)
            trad_wd = round(sub_trad['wasserstein_distance'].mean(), 3)
            trad_rd = round(sub_trad['radiomics_distance'].mean(), 3)
        else:
            trad_ssim, trad_wd, trad_rd = [0.0, 0.0, 0.0]
    except Exception:
        trad_ssim, trad_wd, trad_rd = ["---", "---", "---"] 

    try:
        df_diff = pd.read_csv("metrics_by_set.csv")
        sub_diff = df_diff[df_diff['case_id'].str.contains(case_name, case=False)]
        if not sub_diff.empty:
            diff_ssim = round(sub_diff[f'{modality}_ssim'].mean(), 3)
            diff_wd = round(sub_diff[f'{modality}_wasserstein_distance'].mean(), 3)
            diff_rd = round(sub_diff[f'{modality}_radiomics_distance'].mean(), 3)
        else:
            diff_ssim, diff_wd, diff_rd = [0.0, 0.0, 0.0]
    except Exception:
        diff_ssim, diff_wd, diff_rd = ["---", "---", "---"] 

    df_metrics2 = pd.DataFrame({
        "Method": ["Baseline", "Traditional TTA", "Diffusion TTA"],
        "SSIM": [base_ssim, trad_ssim, diff_ssim],
        "Wasserstein distance": [base_wd, trad_wd, diff_wd],
        "Radiomics distance": [base_rd, trad_rd, diff_rd]
    })

    return original_img, baseline_img, base_score, diff_02, diff_04, diff_06, diff_score, trad_gallery, trad_score, df_metrics1, df_metrics2

# --- 3. Gradio UI 레이아웃 구성 ---
with gr.Blocks(title="Diffusion Model TTA Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Medical Image Diffusion Test-Time Augmentation (TTA)")
    
    with gr.Row():
        # --- 좌측 패널 ---
        with gr.Column(scale=1):
            gr.Markdown("## [Input Settings]")
            
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

            case_input = gr.Dropdown(choices=case_list, value=case_list[0], label="Case Name")
            modality_input = gr.Dropdown(choices=["T2WI", "DWI", "ADC"], value="T2WI", label="Modality")
            run_btn = gr.Button("결과 확인 (Run)", variant="primary")
            
            gr.Markdown("## [Original Image]")
            original_output = gr.Image(label="Original", interactive=False)
            
        # --- 우측 패널 ---
        with gr.Column(scale=3):
            gr.Markdown("## [Image Results]")
        
            gr.Markdown("#### 1. Baseline")
            with gr.Row():
                baseline_output = gr.Image(label="Baseline (Original)", interactive=False)
                base_score_output = gr.Textbox(label="Prediction Score", interactive=False)
        
            gr.Markdown("#### 2. Traditional TTA Generated Images (aug01 ~ aug26)")
            trad_gallery_output = gr.Gallery(label="Traditional TTA", show_label=True, elem_id="gallery", columns=[6], rows=[1], object_fit="contain", height="auto")
            trad_score_output = gr.Textbox(label="Prediction Score", interactive=False)
        
            gr.Markdown("#### 3. Diffusion TTA Images")
            with gr.Row():
                diff_02_output = gr.Image(label="Diffusion (0.2)", interactive=False)
                diff_04_output = gr.Image(label="Diffusion (0.4)", interactive=False)
                diff_06_output = gr.Image(label="Diffusion (0.6)", interactive=False)
            diff_score_output = gr.Textbox(label="Prediction Score", interactive=False)
            
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
            original_output, 
            baseline_output, base_score_output, 
            diff_02_output, diff_04_output, diff_06_output, diff_score_output, 
            trad_gallery_output, trad_score_output, 
            table1_output, table2_output
        ]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=10000)
