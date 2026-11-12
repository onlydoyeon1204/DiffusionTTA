import gradio as gr
import os
import pandas as pd
import openpyxl

# --- 0. 환자 ID 리스트 (case01 ~ case52 자동 생성) ---
case_list = [f"case{i:02d}" for i in range(1, 53)]

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

    # 예측 점수(Prediction Scores) 및 실제 정답(Ground Truth) 추출
    base_score, trad_score, diff_score = "---", "---", "---"
    ground_truth = "---"
    
    try:
        df_base = pd.read_excel("baseline_scores.xlsx")
        base_row = df_base[df_base['case_id'].str.contains(case_name, case=False, na=False)]
        if not base_row.empty:
            b_vals = base_row.iloc[0][['baseline_score_seed1', 'baseline_score_seed2', 'baseline_score_seed3', 'baseline_score_seed4', 'baseline_score_seed5']]
            base_score = float(b_vals.mean())
            row_idx = base_row.index[0] 
            
            if 'y_true' in base_row.columns:
                y_val = base_row.iloc[0]['y_true']
                ground_truth = "High Grade (악성)" if y_val == 1 else "Low Grade (양성)"
            
            df_trad = pd.read_excel("traditionalTTA_scores.xlsx")
            if len(df_trad) > row_idx:
                t_vals = df_trad.iloc[row_idx][['tta_score_seed1', 'tta_score_seed2', 'tta_score_seed3', 'tta_score_seed4', 'tta_score_seed5']]
                trad_score = float(t_vals.mean())
        
        df_diff = pd.read_excel("diffusionTTA_scores.xlsx", sheet_name=0)
        diff_row = df_diff[df_diff['case_id'].str.contains(case_name, case=False, na=False)]
        if not diff_row.empty:
            d_vals = diff_row.iloc[0][['diffusion_score_seed1', 'diffusion_score_seed2', 'diffusion_score_seed3', 'diffusion_score_seed4', 'diffusion_score_seed5']]
            diff_score = float(d_vals.mean())
    except Exception:
        pass 
    
    # 테이블 1
    df_metrics1 = pd.DataFrame({
        "Method": ["Baseline", "Traditional TTA", "Diffusion TTA"],
        "Accuracy": ["0.77±0.01","0.80±0.03", "0.84±0.01"],
        "Balanced Accuracy": ["0.76±0.01","0.81±0.03", "0.84±0.01"],
        "Sensitivity": ["0.70±0.07","0.74±0.06", "0.86±0.02"],
        "Specificity": ["0.82±0.06","0.87±0.03", "0.82±0.03"],
        "AUC": ["0.75±0.01","0.76±0.05", "0.80±0.01"]
    })
    
    # 테이블 2
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

    def format_score(score):
        if score == "---" or "에러" in str(score):
            return f"<div style='font-size:16px; margin-top:5px;'><b>Prediction Score:</b> {score}</div>"
        
        score_val = float(score)
        pred_class = "High Grade" if score_val >= 0.35 else "Low Grade"
        color = "#e74c3c" if score_val >= 0.35 else "#27ae60"
        
        return f"<div style='font-size:16px; margin-top:5px;'><b>Prediction Score:</b> <span style='color:{color}; font-weight:bold;'>{score_val:.3f} ({pred_class})</span></div>"
    
    base_md = format_score(base_score)
    trad_md = format_score(trad_score)
    diff_md = format_score(diff_score)

    if ground_truth != "---":
        gt_color = "#e74c3c" if "High" in ground_truth else "#27ae60"
        bg_color = "#fadbd8" if "High" in ground_truth else "#d5f5e3"
        gt_html = f"<div style='padding: 10px; background-color: {bg_color}; border-left: 5px solid {gt_color}; font-size: 16px; margin-bottom: 15px; border-radius: 4px;'><b>🎯 실제 정답 (Ground Truth):</b> <span style='color:{gt_color}; font-weight:bold;'>{ground_truth}</span></div>"
    else:
        gt_html = "<div style='padding: 10px; background-color: #f2f2f2; border-left: 5px solid #ccc; font-size: 16px; margin-bottom: 15px; border-radius: 4px;'><b>🎯 실제 정답 (Ground Truth):</b> 데이터 없음</div>"

    return original_img, gt_html, baseline_img, base_md, diff_02, diff_04, diff_06, diff_md, trad_gallery, trad_md, df_metrics1, df_metrics2


# --- 3. Gradio UI 레이아웃 구성 ---
with gr.Blocks(title="Clinical TTA Dashboard", theme=gr.themes.Base()) as demo:
    gr.Markdown("# 🏥 Medical Image Diffusion TTA Clinical Dashboard")
    gr.Markdown("Test-Time Augmentation (TTA) 기반의 악성 종양(Malignancy) 예측 및 영상 분석 시스템입니다.")
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 📋 Patient Information")
                case_input = gr.Dropdown(choices=case_list, value=case_list[0], label="Patient ID (Anonymized)")
                modality_input = gr.Dropdown(choices=["T2WI", "DWI", "ADC"], value="T2WI", label="Scan Modality")
                run_btn = gr.Button("Analyze Scan (Run)", variant="primary")
                
            gr.Markdown("<br>")
            with gr.Group():
                gr.Markdown("### 🔍 Original Image")
                # 이미지 크기를 크게(height 늘림) 유지
                original_output = gr.Image(label="Input Image", interactive=False, height=450)
                
        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.TabItem("🖼️ Image Analysis"):
                    # 점수 설명 추가 보강
                    gr.Markdown("""
                    <div style='background-color:#f0f4f8; padding:15px; border-radius:8px; line-height:1.6;'>
                        <b>💡 Malignancy Prediction Score (악성 예측 점수)</b><br>
                        인공지능 모델이 해당 의료 영상을 분석하여 종양이 악성(Malignancy)일 확률을 0과 1 사이의 수치로 나타낸 값입니다.<br>
                        본 모델의 임상 최적 기준점(Optimal Threshold)인 <b>0.35를 기준으로, 이상일 경우 악성(High Grade), 미만일 경우 양성/저악성(Low Grade)으로 판별</b>합니다. 
                        특히 Diffusion TTA를 적용하면, 기존에 놓치기 쉬웠던(False Negative) High Grade 종양을 더욱 민감하고 정확하게 잡아낼 수 있습니다.
                    </div>
                    """)
                    gr.Markdown("<br>")
                    
                    gt_output = gr.HTML()
                    
                    # 각 Group(테두리)에 여백(마진)을 주어 답답함을 해소
                    with gr.Group(elem_classes="margin-bottom-20"):
                        gr.Markdown("### 1. Baseline (Original)")
                        gr.Markdown("*아무런 변형(Augmentation)을 가하지 않은 원본 의료 영상 그대로를 인공지능 모델에 입력하여 얻은 예측 결과입니다.*")
                        baseline_output = gr.Image(label="Baseline", interactive=False)
                        base_score_output = gr.HTML()
                        gr.Markdown("<br>")
                
                    with gr.Group(elem_classes="margin-bottom-20"):
                        gr.Markdown("### 2. Traditional TTA Generated Images")
                        gr.Markdown("*회전, 반전, 이동 등 전통적인 기하학적 변형(Geometric Augmentation) 기법을 사용하여 생성된 26장의 테스트 이미지(aug01 ~ aug26)입니다. 다양성이 다소 부족할 수 있습니다.*")
                        trad_gallery_output = gr.Gallery(label="Traditional TTA", show_label=False, elem_id="gallery", columns=[6], rows=[1], object_fit="contain", height="auto")
                        trad_score_output = gr.HTML()
                        gr.Markdown("<br>")
                
                    with gr.Group(elem_classes="margin-bottom-20"):
                        gr.Markdown("### 3. Diffusion TTA Images")
                        gr.Markdown("*Diffusion 생성 모델을 활용하여 질감(Texture)과 미세 구조에 사실적인 변형을 가한 이미지입니다. Strength(강도) 수치가 높을수록 원본과 더 많은 차이를 보입니다.*")
                        with gr.Row():
                            # 강도 텍스트가 잘 보이도록 제목으로 배치
                            with gr.Column():
                                gr.Markdown("<h4 style='text-align:center;'>Strength 0.2</h4>")
                                diff_02_output = gr.Image(label="Strength: 0.2", interactive=False, show_label=False)
                            with gr.Column():
                                gr.Markdown("<h4 style='text-align:center;'>Strength 0.4</h4>")
                                diff_04_output = gr.Image(label="Strength: 0.4", interactive=False, show_label=False)
                            with gr.Column():
                                gr.Markdown("<h4 style='text-align:center;'>Strength 0.6</h4>")
                                diff_06_output = gr.Image(label="Strength: 0.6", interactive=False, show_label=False)
                        diff_score_output = gr.HTML()
                        gr.Markdown("<br>")
                
                with gr.TabItem("📊 Clinical Metrics"):
                    gr.Markdown("### Evaluation & Similarity Metrics")
                    
                    with gr.Group():
                        gr.Markdown("#### Table 1. Prediction Metrics (예측 성능 지표)")
                        gr.Markdown("""
                        <ul style='line-height:1.8;'>
                            <li><b>Accuracy / Balanced Accuracy <span style='color:#e74c3c;'>(↑ 높을수록 좋음)</span></b> : 모델이 양성과 악성을 얼마나 정확하게 맞추었는지를 나타내는 종합 지표입니다.</li>
                            <li><b>Sensitivity <span style='color:#e74c3c;'>(↑ 높을수록 좋음)</span></b> : 실제 악성(High Grade) 환자를 악성이라고 정확히 잡아낸 비율입니다. 임상에서 가장 중요한 지표입니다.</li>
                            <li><b>Specificity <span style='color:#e74c3c;'>(↑ 높을수록 좋음)</span></b> : 실제 양성(Low Grade) 환자를 양성이라고 올바르게 판별한 비율입니다.</li>
                            <li><b>AUC <span style='color:#e74c3c;'>(↑ 높을수록 좋음)</span></b> : 모델의 예측 점수 분포가 두 클래스(High/Low)를 얼마나 잘 구별해 내는지 나타내는 곡선 아래 면적입니다. 1에 가까울수록 완벽합니다.</li>
                        </ul>
                        """)
                        table1_output = gr.Dataframe(headers=["Method", "Accuracy", "Balanced Accuracy", "Sensitivity", "Specificity", "AUC"], interactive=False)
                    
                    gr.Markdown("<br>")
                    
                    with gr.Group():
                        gr.Markdown("#### Table 2. Similarity & Distance Metrics (영상 변형 지표)")
                        gr.Markdown("""
                        <ul style='line-height:1.8;'>
                            <li><b>SSIM <span style='color:#e74c3c;'>(↑ 1에 가까울수록 원본과 유사)</span></b> : 생성된 이미지가 원본 영상의 구조적 특성(Structural)을 얼마나 잘 유지하고 있는지 평가합니다. 너무 낮으면 해부학적 구조가 망가졌음을 의미합니다.</li>
                            <li><b>Wasserstein / Radiomics distance <span style='color:#3498db;'>(↓ 낮을수록 원본 특징 보존)</span></b> : 원본 영상과 TTA 이미지 사이의 픽셀 분포 및 특징적 거리(Distance) 차이를 측정합니다. 값이 너무 크면 임상적 특징이 훼손되었을 가능성이 있습니다.</li>
                        </ul>
                        *※ **Baseline**은 원본 영상 그 자체이므로 자신과의 비교 수치(SSIM=1, Distance=0)는 의미가 없어 고정값 처리되었습니다.*
                        """)
                        table2_output = gr.Dataframe(headers=["Method", "SSIM", "Wasserstein distance", "Radiomics distance"], interactive=False)

    # 마진(여백)을 주기 위한 간단한 CSS 주입
    demo.css = """
    .margin-bottom-20 {
        margin-bottom: 25px !important;
        padding-bottom: 10px !important;
    }
    """

    run_btn.click(
        fn=process_case,
        inputs=[case_input, modality_input],
        outputs=[
            original_output, 
            gt_output,             
            baseline_output, base_score_output, 
            diff_02_output, diff_04_output, diff_06_output, diff_score_output, 
            trad_gallery_output, trad_score_output, 
            table1_output, table2_output
        ]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    demo.launch(server_name="0.0.0.0", server_port=port)
