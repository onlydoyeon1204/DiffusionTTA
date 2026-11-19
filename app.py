import gradio as gr
import os
import pandas as pd
import openpyxl
import base64

# --- 0. Patient ID List ---
# 💡 여기에 Baseline/Traditional TTA는 틀렸지만 Diffusion TTA는 맞춘 대표 케이스 번호를 입력하세요.
representative_cases = ["case03", "case15", "case42"] 
all_cases = [f"case{i:02d}" for i in range(1, 53)]

# --- 1. File Search Helper Functions ---
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

# --- 2. Main Processing Logic ---
def process_case(case_name, modality_display):
    modality_map = {"T2WI": "t2", "DWI": "dwi", "ADC": "adc"}
    modality = modality_map.get(modality_display, "t2")
    
    # 원본 및 Baseline 이미지
    original_img = find_image("original_image", case_name, modality)
    baseline_img = original_img
    
    # Traditional TTA 갤러리 이미지
    trad_gallery = find_gallery_images("Traditional_TTA", case_name, modality)
    
    # 🚨 수정됨: Diffusion TTA 갤러리 이미지 (strength 0.2, 0.4, 0.6 폴더 모두 탐색하여 통합)
    diff_gallery = []
    for strength in ["strength_0.2", "strength_0.4", "strength_0.6"]:
        strength_paths = find_gallery_images(os.path.join("Diffusion_TTA", strength), case_name, modality)
        diff_gallery.extend(strength_paths)

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
                ground_truth = "High Grade (고위험군)" if y_val == 1 else "Low Grade (저위험군)"
            
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
    
    # --- Table 1: HTML Table ---
    table1_html = """
    <table style="width:100%; border-collapse: collapse; text-align: center; font-size: 15px; margin-top:10px;">
        <thead>
            <tr style="background-color: #f3f4f6; border-bottom: 2px solid #d1d5db;">
                <th style="padding: 12px; border: 1px solid #e5e7eb;">Method</th>
                <th style="padding: 12px; border: 1px solid #e5e7eb;">Accuracy <span style="color:#e74c3c;">(↑)</span></th>
                <th style="padding: 12px; border: 1px solid #e5e7eb;">Balanced Accuracy <span style="color:#e74c3c;">(↑)</span></th>
                <th style="padding: 12px; border: 1px solid #e5e7eb;">Sensitivity <span style="color:#e74c3c;">(↑)</span></th>
                <th style="padding: 12px; border: 1px solid #e5e7eb;">Specificity <span style="color:#e74c3c;">(↑)</span></th>
                <th style="padding: 12px; border: 1px solid #e5e7eb;">AUC <span style="color:#e74c3c;">(↑)</span></th>
            </tr>
        </thead>
        <tbody>
            <tr><td style="padding: 10px; border: 1px solid #e5e7eb;">Baseline</td><td style="padding: 10px; border: 1px solid #e5e7eb;">0.77±0.01</td><td style="padding: 10px; border: 1px solid #e5e7eb;">0.76±0.01</td><td style="padding: 10px; border: 1px solid #e5e7eb;">0.70±0.07</td><td style="padding: 10px; border: 1px solid #e5e7eb;">0.82±0.06</td><td style="padding: 10px; border: 1px solid #e5e7eb;">0.75±0.01</td></tr>
            <tr style="background-color: #f9fafb;"><td style="padding: 10px; border: 1px solid #e5e7eb;">Traditional TTA</td><td style="padding: 10px; border: 1px solid #e5e7eb;">0.80±0.03</td><td style="padding: 10px; border: 1px solid #e5e7eb;">0.81±0.03</td><td style="padding: 10px; border: 1px solid #e5e7eb;">0.74±0.06</td><td style="padding: 10px; border: 1px solid #e5e7eb;">0.87±0.03</td><td style="padding: 10px; border: 1px solid #e5e7eb;">0.76±0.05</td></tr>
            <tr style="background-color: #eaf2f8; font-weight: bold;"><td style="padding: 10px; border: 1px solid #e5e7eb; color: #2c3e50;">Diffusion TTA</td><td style="padding: 10px; border: 1px solid #e5e7eb; color: #e74c3c;">0.84±0.01</td><td style="padding: 10px; border: 1px solid #e5e7eb; color: #e74c3c;">0.84±0.01</td><td style="padding: 10px; border: 1px solid #e5e7eb; color: #e74c3c;">0.86±0.02</td><td style="padding: 10px; border: 1px solid #e5e7eb; color: #e74c3c;">0.82±0.03</td><td style="padding: 10px; border: 1px solid #e5e7eb; color: #e74c3c;">0.80±0.01</td></tr>
        </tbody>
    </table>
    """
    
    # Table 2 Data Preparation
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

    def fmt(val): return f"{val:.3f}" if isinstance(val, (int, float)) else str(val)

    # --- Table 2: HTML Table ---
    table2_html = f"""
    <table style="width:100%; border-collapse: collapse; text-align: center; font-size: 15px; margin-top:10px;">
        <thead>
            <tr style="background-color: #f3f4f6; border-bottom: 2px solid #d1d5db;">
                <th style="padding: 12px; border: 1px solid #e5e7eb;">Method</th>
                <th style="padding: 12px; border: 1px solid #e5e7eb;">SSIM <span style="color:#e74c3c;">(↑)</span></th>
                <th style="padding: 12px; border: 1px solid #e5e7eb;">Wasserstein distance <span style="color:#3498db;">(↓)</span></th>
                <th style="padding: 12px; border: 1px solid #e5e7eb;">Radiomics distance <span style="color:#3498db;">(↓)</span></th>
            </tr>
        </thead>
        <tbody>
            <tr><td style="padding: 10px; border: 1px solid #e5e7eb;">Baseline</td><td style="padding: 10px; border: 1px solid #e5e7eb;">{fmt(base_ssim)}</td><td style="padding: 10px; border: 1px solid #e5e7eb;">{fmt(base_wd)}</td><td style="padding: 10px; border: 1px solid #e5e7eb;">{fmt(base_rd)}</td></tr>
            <tr style="background-color: #f9fafb;"><td style="padding: 10px; border: 1px solid #e5e7eb;">Traditional TTA</td><td style="padding: 10px; border: 1px solid #e5e7eb;">{fmt(trad_ssim)}</td><td style="padding: 10px; border: 1px solid #e5e7eb;">{fmt(trad_wd)}</td><td style="padding: 10px; border: 1px solid #e5e7eb;">{fmt(trad_rd)}</td></tr>
            <tr style="background-color: #eaf2f8; font-weight: bold;"><td style="padding: 10px; border: 1px solid #e5e7eb; color: #2c3e50;">Diffusion TTA</td><td style="padding: 10px; border: 1px solid #e5e7eb; color: #e74c3c;">{fmt(diff_ssim)}</td><td style="padding: 10px; border: 1px solid #e5e7eb; color: #3498db;">{fmt(diff_wd)}</td><td style="padding: 10px; border: 1px solid #e5e7eb; color: #3498db;">{fmt(diff_rd)}</td></tr>
        </tbody>
    </table>
    """

    # 🚨 수정됨: Score(수치)를 감추고 Result(최종 분류 결과)만 출력하도록 변경
    def format_result(score):
        if score == "---" or "Error" in str(score).title():
            return f"<div style='font-size:16px; margin-top:5px;'><b>Prediction Result:</b> Data Not Available</div>"
        
        score_val = float(score)
        pred_class = "High Grade (고위험군)" if score_val >= 0.5 else "Low Grade (저위험군)"
        color = "#e74c3c" if score_val >= 0.5 else "#27ae60"
        
        return f"<div style='font-size:16px; margin-top:5px;'><b>Prediction Result:</b> <span style='color:{color}; font-weight:bold;'>{pred_class}</span></div>"
    
    base_res = format_result(base_score)
    trad_res = format_result(trad_score)
    diff_res = format_result(diff_score)

    if ground_truth != "---":
        gt_color = "#e74c3c" if "High" in ground_truth else "#27ae60"
        bg_color = "#fadbd8" if "High" in ground_truth else "#d5f5e3"
        gt_html = f"<div style='padding: 10px; background-color: {bg_color}; border-left: 5px solid {gt_color}; font-size: 16px; margin-bottom: 15px; border-radius: 4px;'><b>🎯 Ground Truth:</b> <span style='color:{gt_color}; font-weight:bold;'>{ground_truth}</span></div>"
    else:
        gt_html = "<div style='padding: 10px; background-color: #f2f2f2; border-left: 5px solid #ccc; font-size: 16px; margin-bottom: 15px; border-radius: 4px;'><b>🎯 Ground Truth:</b> No Data</div>"

    return original_img, gt_html, baseline_img, base_res, trad_gallery, trad_res, diff_gallery, diff_res, table1_html, table2_html


# Custom CSS
custom_css = """
.margin-bottom-20 {
    margin-bottom: 25px !important;
    padding-bottom: 10px !important;
}
button[aria-label="Download"], button[title="Download"], a[download] {
    display: none !important;
    pointer-events: none !important;
}
img {
    -webkit-user-drag: none;
    -khtml-user-drag: none;
    -moz-user-drag: none;
    -o-user-drag: none;
    user-drag: none;
    pointer-events: none;
}
"""

logo_path = "unilogo.jpg"
logo_html_str = "<span style='font-size: 32px; margin-right: 10px;'>🏥</span>" 
if os.path.exists(logo_path):
    try:
        with open(logo_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            logo_html_str = f"<img src='data:image/jpeg;base64,{encoded_string}' style='height: 48px; margin-right: 15px; border-radius: 4px;'>"
    except Exception as e:
        pass

# --- 3. Gradio UI Layout Configuration ---
with gr.Blocks(title="Clinical TTA Dashboard", css=custom_css) as demo:
    
    gr.HTML(f"""
    <div style='display: flex; align-items: center; margin-bottom: 10px;'>
        {logo_html_str}
        <h1 style='margin: 0; padding: 0;'>Medical Image Diffusion TTA Clinical Dashboard</h1>
    </div>
    """)
    
    gr.Markdown("""
    A prostate cancer risk prediction and image analysis system based on Test-Time Augmentation (TTA).
    
    테스트 시간 증강(TTA) 기법을 기반으로 한 전립선암 위험군 예측 및 이미지 분석 시스템입니다.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 📋 Patient Information")
                # 🚨 수정됨: 추천/대표 케이스 안내 추가
                gr.Markdown(f"**💡 Recommended Cases:** `{', '.join(representative_cases)}`<br>*(Cases where Diffusion TTA successfully predicts while others fail)*")
                
                case_input = gr.Dropdown(choices=all_cases, value=representative_cases[0], label="Patient ID (Anonymized)")
                modality_input = gr.Dropdown(choices=["T2WI", "DWI", "ADC"], value="T2WI", label="Scan Modality")
                run_btn = gr.Button("Analyze Scan (Run)", variant="primary")
                
            gr.Markdown("<br>")
            with gr.Group():
                gr.Markdown("### 🔍 Original Image")
                original_output = gr.Image(label="Input Image", interactive=False, height=450)
                
        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.TabItem("🖼️ Image Analysis"):
                    # 🚨 수정됨: 예측 점수 수치(Score) 표기 내용 삭제하고 결과(Result) 내용만 반영
                    gr.Markdown("""
                    <div style='background-color:#f0f4f8; padding:15px; border-radius:8px; line-height:1.6;'>
                        <b>💡 Malignancy Prediction Result</b><br>
                        Shows the final classification result (High Grade or Low Grade) predicted by the AI model. 
                        <hr style='margin: 10px 0; border: 0; border-top: 1px solid #d1d5db;'>
                        <b>💡 악성도 예측 결과</b><br>
                        AI 모델이 분석한 최종 분류 결과(고위험군 또는 저위험군)를 나타냅니다. 
                    </div>
                    """)
                    gr.Markdown("<br>")
                    
                    gt_output = gr.HTML()
                    
                    with gr.Group(elem_classes="margin-bottom-20"):
                        gr.Markdown("### 1. Baseline (Original)")
                        gr.Markdown("*Prediction results from the original medical image without any augmentation.*<br>*어떠한 증강 기법도 적용하지 않은 원본 의료 영상의 예측 결과입니다.*")
                        baseline_output = gr.Image(label="Baseline", interactive=False)
                        base_result_output = gr.HTML()
                        gr.Markdown("<br>")
                
                    with gr.Group(elem_classes="margin-bottom-20"):
                        gr.Markdown("### 2. Traditional TTA Generated Images")
                        # 🚨 수정됨: "다양성이 다소 부족할 수 있습니다" 문구 삭제
                        gr.Markdown("*Images generated using geometric augmentations (rotation, flip, etc.).*<br>*회전, 반전 등 전통적인 기하학적 증강을 통해 생성된 이미지입니다.*")
                        trad_gallery_output = gr.Gallery(label="Traditional TTA", show_label=False, elem_id="gallery_trad", columns=[6], rows=[1], object_fit="contain", height="auto", interactive=False)
                        trad_result_output = gr.HTML()
                        gr.Markdown("<br>")
                
                    with gr.Group(elem_classes="margin-bottom-20"):
                        gr.Markdown("### 3. Diffusion TTA Images")
                        # 🚨 수정됨: Diffusion TTA 이미지 26장 전체를 갤러리로 표시
                        gr.Markdown("*Images with realistic transformations applied using a Diffusion model. Incorporates strengths 0.2, 0.4, and 0.6.*<br>*디퓨전 모델을 활용하여 미세 구조와 질감에 현실적인 변형을 가한 이미지입니다. (Strength 0.2, 0.4, 0.6 생성 영상 모두 포함)*")
                        diff_gallery_output = gr.Gallery(label="Diffusion TTA", show_label=False, elem_id="gallery_diff", columns=[6], rows=[2], object_fit="contain", height="auto", interactive=False)
                        diff_result_output = gr.HTML()
                        gr.Markdown("<br>")
                
                with gr.TabItem("📊 Clinical Metrics"):
                    # 🚨 수정됨: 3가지 모달리티 융합(Fusion) 결과임을 설명하는 안내 상자 추가
                    gr.Markdown("""
                    <div style='background-color:#fff3cd; padding:15px; border-radius:8px; line-height:1.7; border-left: 4px solid #ffc107; margin-bottom: 15px;'>
                        <b>⚠️ Note on Modality Fusion</b><br>
                        The prediction metrics presented in Table 1 represent the <b>fused results of all three modalities (T2WI, DWI, ADC)</b> combined, unlike the 'Image Analysis' tab which shows individual modality outputs.<br>
                        <hr style='margin: 10px 0; border: 0; border-top: 1px solid #ffeeba;'>
                        <b>⚠️ 모달리티 융합(Fusion) 안내</b><br>
                        표 1에 제시된 예측 성능 지표는 'Image Analysis' 탭의 단일 모달리티 결과와 달리, <b>T2WI, DWI, ADC 세 가지 모달리티의 결과를 모두 합친(융합한) 최종 성능</b>을 나타냅니다.
                    </div>
                    """)
                    
                    gr.Markdown("### Evaluation & Similarity Metrics")
                    
                    with gr.Group():
                        gr.Markdown("#### Table 1. Prediction Metrics")
                        gr.Markdown("""
                        <div style='background-color:#f9f9f9; padding:15px; border-radius:8px; line-height:1.7; border-left: 4px solid #e74c3c;'>
                            <b>[Metric Descriptions]</b> <span style='color:#e74c3c;'><b>(↑ indicates higher values are better)</b></span><br>
                            - <b>Accuracy / Balanced Accuracy</b> : A comprehensive metric showing how accurately the model classifies Low Grade and High Grade cases.<br>
                            - <b>Sensitivity</b> : The ratio of actual High Grade patients correctly identified. The most crucial clinical metric.<br>
                            - <b>Specificity</b> : The ratio of actual Low Grade patients correctly identified.<br>
                            - <b>AUC</b> : Indicates how well the prediction score distribution distinguishes between the two classes.
                            <hr style='margin: 10px 0; border: 0; border-top: 1px solid #d1d5db;'>
                            <b>[지표 설명]</b> <span style='color:#e74c3c;'><b>(↑ 높을수록 좋은 지표)</b></span><br>
                            - <b>Accuracy / Balanced Accuracy</b> : 저위험군과 고위험군을 얼마나 정확하게 분류하는지 나타내는 종합 지표입니다.<br>
                            - <b>Sensitivity (민감도)</b> : 실제 고위험군 환자를 고위험군으로 정확히 찾아낸 비율입니다. 임상적으로 가장 중요한 지표입니다.<br>
                            - <b>Specificity (특이도)</b> : 실제 저위험군 환자를 저위험군으로 정확히 분류한 비율입니다.<br>
                            - <b>AUC</b> : 두 클래스 간의 점수 분포가 얼마나 명확하게 구별되는지 나타내는 지표입니다.
                        </div>
                        """)
                        table1_output = gr.HTML()
                    
                    gr.Markdown("<br>")
                    
                    with gr.Group():
                        gr.Markdown("#### Table 2. Similarity & Distance Metrics")
                        gr.Markdown("""
                        <div style='background-color:#f9f9f9; padding:15px; border-radius:8px; line-height:1.7; border-left: 4px solid #3498db;'>
                            <b>[Metric Descriptions]</b> <br>
                            - <b>SSIM <span style='color:#e74c3c;'>(↑ higher is better)</span></b> : Evaluates how well the generated image preserves the anatomical structure of the original image.<br>
                            - <b>Wasserstein / Radiomics distance <span style='color:#3498db;'>(↓ lower is better)</span></b> : Measures the textural differences between the original and TTA images. If too large, clinical features may be compromised.
                            <hr style='margin: 10px 0; border: 0; border-top: 1px solid #d1d5db;'>
                            <b>[지표 설명]</b> <br>
                            - <b>SSIM <span style='color:#e74c3c;'>(↑ 높을수록 좋은 지표)</span></b> : 생성된 이미지가 원본 이미지의 해부학적 구조를 얼마나 잘 보존하는지 평가합니다.<br>
                            - <b>Wasserstein / Radiomics distance <span style='color:#3498db;'>(↓ 낮을수록 좋은 지표)</span></b> : 원본과 TTA 이미지 간의 픽셀 분포 및 질감 차이를 측정합니다. 수치가 너무 높으면 기존 임상적 특징이 훼손되었음을 의미할 수 있습니다.
                        </div>
                        """)
                        table2_output = gr.HTML()

    # 🚨 수정됨: 출력 인자들을 갤러리와 Result에 맞게 재배열
    run_btn.click(
        fn=process_case,
        inputs=[case_input, modality_input],
        outputs=[
            original_output, 
            gt_output,             
            baseline_output, base_result_output, 
            trad_gallery_output, trad_result_output,
            diff_gallery_output, diff_result_output, 
            table1_output, table2_output
        ]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    demo.launch(server_name="0.0.0.0", server_port=port, theme=gr.themes.Base())
