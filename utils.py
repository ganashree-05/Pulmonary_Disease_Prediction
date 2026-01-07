# utils.py
from fpdf import FPDF
from datetime import datetime
import os

REPORT_DIR = "reports"

def generate_pdf_local(data, lang="en"):
    os.makedirs(REPORT_DIR, exist_ok=True)

    fname = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    path = os.path.join(REPORT_DIR, fname)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Pulmonary Disease Prediction Report", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Patient Name: {data.get('patient_name','')}", ln=True)
    pdf.cell(200, 10, txt=f"Age Category: {data.get('age_category','')}", ln=True)
    pdf.cell(200, 10, txt=f"Risk Level: {data.get('risk','')}", ln=True)
    pdf.multi_cell(0, 8, f"Suggestion: {data.get('suggestion','')}")

    pdf.ln(6)
    pdf.cell(0, 8, "General Symptoms:", ln=True)
    for t in data.get("general_texts", []):
        pdf.multi_cell(0, 6, f"- {t}")

    pdf.ln(4)
    pdf.cell(0, 8, "Age-specific Symptoms:", ln=True)
    for t in data.get("age_texts", []):
        pdf.multi_cell(0, 6, f"- {t}")

    if data.get("xray_filename"):
        pdf.ln(6)
        pdf.multi_cell(0, 6, f"X-ray file: {data.get('xray_filename')}")

    pdf.output(path)
    return path
