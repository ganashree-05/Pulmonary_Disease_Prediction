import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# Directory to save PDFs
REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)


def generate_pdf_full(data):
    """
    Generates a PDF report for pulmonary disease prediction.
    data: dict with keys:
        - risk: str
        - suggestion: str
        - general_texts: list of strings
        - age_texts: list of strings
        - xray_filename: str (optional)
        - ml_pred: str (optional)
        - ml_conf: float (optional)
    Returns: full path of saved PDF
    """
    fname = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    path = os.path.join(REPORT_DIR, fname)

    c = canvas.Canvas(path, pagesize=A4)
    width, height = A4
    y = height - 50  # start from top

    # Title
    c.setFont("Times-Roman", 16)
    c.drawString(50, y, "Pulmonary Disease Prediction Report")
    y -= 30

    # Risk Level
    c.setFont("Times-Roman", 14)
    c.drawString(50, y, f"Risk Level: {data.get('risk', '')}")
    y -= 25

    # Suggestion
    suggestion = data.get('suggestion', '')
    text_obj = c.beginText(50, y)
    text_obj.setFont("Times-Roman", 12)
    text_obj.textLines(f"Suggestion: {suggestion}")
    c.drawText(text_obj)
    y -= (12 * (suggestion.count('\n') + 2) + 10)

    # General Symptoms
    c.setFont("Times-Roman", 12)
    c.drawString(50, y, "General Symptoms:")
    y -= 20
    for t in data.get("general_texts", []):
        c.drawString(70, y, f"- {t}")
        y -= 18

    # Age-specific Symptoms
    y -= 10
    c.drawString(50, y, "Age-specific Symptoms:")
    y -= 20
    for t in data.get("age_texts", []):
        c.drawString(70, y, f"- {t}")
        y -= 18

    # X-ray prediction
    xray_file = data.get("xray_filename")
    if xray_file:
        y -= 20
        c.drawString(50, y, f"X-ray File: {xray_file}")
        y -= 18
        ml_pred = data.get("ml_pred", "")
        ml_conf = data.get("ml_conf", "")
        c.drawString(50, y, f"X-ray Prediction: {ml_pred} (Confidence: {ml_conf}%)")

    c.showPage()
    c.save()
    return path


# ------------------------------
# Example Usage
# ------------------------------

if __name__ == "__main__":
    # Example survey data
    data = {
        "risk": "High",
        "suggestion": "High risk for pneumonia. Upload a recent chest X-ray and seek medical care as soon as possible.",
        "general_texts": [
            "Cough lasting more than a few days",
            "Fever above 101°F",
            "Persistent fatigue"
        ],
        "age_texts": [
            "Shortness of breath",
            "Weakness and low energy"
        ],
        "xray_filename": "chest_xray_001.jpg",
        "ml_pred": "Pneumonia",
        "ml_conf": 92.5
    }

    pdf_path = generate_pdf_full(data)
    print(f"PDF generated successfully at: {pdf_path}")
