import gradio as gr
import os
import pandas as pd
import openpyxl
import base64

# --- 0. Patient ID List (case01 ~ case52 auto-generated) ---
case_list = [f"case{i:02d}" for i in range(1, 53)]

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
    
    original_img = find_image("original_image", case_name, modality)
    baseline_img = original_img
    
    diff_02 = find_image(os.path.join("Diffusion_TTA", "strength_0.2"), case_name, modality)
    diff_04 = find_image(os.path.join("Diffusion_TTA", "strength_0.4"), case_name, modality)
    diff_06 = find_image(os.path.join("Diffusion_TTA", "strength_0.6"), case_name, modality)
    
    trad_gallery = find_gallery_images("Traditional_TTA", case_name, modality)

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
                ground_truth = "High Grade (Malignant)" if y_val == 1 else "Low Grade (Benign)"
            
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
    
    # --- Table 1: HTML Table with Colored Arrows and Styles ---
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

    # Formatting function for value output (to prevent errors)
    def fmt(val): return f"{val:.3f}" if isinstance(val, (int, float)) else str(val)

    # --- Table 2: HTML Table with Colored Arrows and Styles ---
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

    def format_score(score):
        if score == "---" or "Error" in str(score).title():
            return f"<div style='font-size:16px; margin-top:5px;'><b>Prediction Score:</b> {score}</div>"
        
        score_val = float(score)
        pred_class = "High Grade" if score_val >= 0.5 else "Low Grade"
        color = "#e74c3c" if score_val >= 0.5 else "#27ae60"
        
        return f"<div style='font-size:16px; margin-top:5px;'><b>Prediction Score:</b> <span style='color:{color}; font-weight:bold;'>{score_val:.3f} ({pred_class})</span></div>"
    
    base_md = format_score(base_score)
    trad_md = format_score(trad_score)
    diff_md = format_score(diff_score)

    if ground_truth != "---":
        gt_color = "#e74c3c" if "High" in ground_truth else "#27ae60"
        bg_color = "#fadbd8" if "High" in ground_truth else "#d5f5e3"
        gt_html = f"<div style='padding: 10px; background-color: {bg_color}; border-left: 5px solid {gt_color}; font-size: 16px; margin-bottom: 15px; border-radius: 4px;'><b>🎯 Ground Truth:</b> <span style='color:{gt_color}; font-weight:bold;'>{ground_truth}</span></div>"
    else:
        gt_html = "<div style='padding: 10px; background-color: #f2f2f2; border-left: 5px solid #ccc; font-size: 16px; margin-bottom: 15px; border-radius: 4px;'><b>🎯 Ground Truth:</b> No Data</div>"

    return original_img, gt_html, baseline_img, base_md, diff_02, diff_04, diff_06, diff_md, trad_gallery, trad_md, table1_html, table2_html


# 💡 Custom CSS to completely block downloads and right-click saves
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
    gr.Markdown("A malignancy prediction and image analysis system based on Test-Time Augmentation (TTA).")
    
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
                original_output = gr.Image(label="Input Image", interactive=False, height=450)
                
        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.TabItem("🖼️ Image Analysis"):
                    gr.Markdown("""
                    <div style='background-color:#f0f4f8; padding:15px; border-radius:8px; line-height:1.6;'>
                        <b>💡 Malignancy Prediction Score</b><br>
                        This is a value between 0 and 1 indicating the probability that the tumor is malignant, as analyzed by the AI model.<br>
                        Based on the experimental control threshold of <b>0.5, a score equal to or higher is classified as High Grade (Malignant), and a lower score is classified as Low Grade (Benign/Low-grade).</b> 
                        In particular, applying Diffusion TTA allows for a more sensitive and accurate detection of High Grade tumors that were previously easy to miss (False Negatives).
                    </div>
                    """)
                    gr.Markdown("<br>")
                    
                    gt_output = gr.HTML()
                    
                    with gr.Group(elem_classes="margin-bottom-20"):
                        gr.Markdown("### 1. Baseline (Original)")
                        gr.Markdown("*Prediction results obtained by inputting the original medical image into the AI model without any augmentation.*")
                        baseline_output = gr.Image(label="Baseline", interactive=False)
                        base_score_output = gr.HTML()
                        gr.Markdown("<br>")
                
                    with gr.Group(elem_classes="margin-bottom-20"):
                        gr.Markdown("### 2. Traditional TTA Generated Images")
                        gr.Markdown("*26 test images (aug01 ~ aug26) generated using traditional geometric augmentation techniques such as rotation, flipping, and translation. Diversity may be somewhat lacking.*")
                        trad_gallery_output = gr.Gallery(label="Traditional TTA", show_label=False, elem_id="gallery", columns=[6], rows=[1], object_fit="contain", height="auto", interactive=False)
                        trad_score_output = gr.HTML()
                        gr.Markdown("<br>")
                
                    with gr.Group(elem_classes="margin-bottom-20"):
                        gr.Markdown("### 3. Diffusion TTA Images")
                        gr.Markdown("*Images with realistic transformations applied to texture and microstructures using a Diffusion generative model. Higher Strength values indicate greater differences from the original.*")
                        with gr.Row():
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
                        gr.Markdown("#### Table 1. Prediction Metrics")
                        gr.Markdown("""
                        <div style='background-color:#f9f9f9; padding:15px; border-radius:8px; line-height:1.7; border-left: 4px solid #e74c3c;'>
                            <b>[Metric Descriptions]</b> <span style='color:#e74c3c;'><b>(↑ indicates higher values are better)</b></span><br>
                            • <b>Accuracy / Balanced Accuracy</b> : A comprehensive metric showing how accurately the model classifies benign and malignant cases.<br>
                            • <b>Sensitivity</b> : The ratio of actual malignant (High Grade) patients correctly identified as malignant. This is the most crucial clinical metric.<br>
                            • <b>Specificity</b> : The ratio of actual benign (Low Grade) patients correctly identified as benign.<br>
                            • <b>AUC</b> : A metric indicating how well the prediction score distribution distinguishes between the two classes. (Closer to 1 is perfect)
                        </div>
                        """)
                        # Render DataFrame component by converting it to an HTML component
                        table1_output = gr.HTML()
                    
                    gr.Markdown("<br>")
                    
                    with gr.Group():
                        gr.Markdown("#### Table 2. Similarity & Distance Metrics")
                        gr.Markdown("""
                        <div style='background-color:#f9f9f9; padding:15px; border-radius:8px; line-height:1.7; border-left: 4px solid #3498db;'>
                            <b>[Metric Descriptions]</b> <br>
                            • <b>SSIM <span style='color:#e74c3c;'>(↑ higher is better)</span></b> : Evaluates how well the generated image preserves the anatomical structure of the original image.<br>
                            • <b>Wasserstein / Radiomics distance <span style='color:#3498db;'>(↓ lower is better)</span></b> : Measures the pixel distribution and textural differences between the original and TTA images. If too large, clinical features may have been compromised.<br>
                            <i>* The Baseline is the original image itself, so comparing it to itself is meaningless; it is set to fixed values (SSIM=1, Distance=0).</i>
                        </div>
                        """)
                        # Render DataFrame component by converting it to an HTML component
                        table2_output = gr.HTML()

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
    demo.launch(server_name="0.0.0.0", server_port=port, theme=gr.themes.Base())
