# ==========================================
# PDF IMAGE FILTERING USING TRAINED MODEL ONLY
# ==========================================

from inference_model import classify_image
import os
import fitz
import cv2
import numpy as np
from PIL import Image
import glob

# ==========================================
# CONFIG
# ==========================================
PDF_FOLDER = "all pdfs"
PDF_LIST = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))

if not PDF_LIST:
    print("No PDFs found in 'all pdfs' folder.")
    exit()

# ==========================================
# STEP 1: EXTRACT IMAGES
# ==========================================
def extract_images(pdf_path, temp_dir):

    doc = fitz.open(pdf_path)
    count = 0

    for page_index in range(len(doc)):
        page = doc[page_index]
        images = page.get_images(full=True)

        for img_index, img in enumerate(images):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)

                image_bytes = base_image["image"]
                image_ext = base_image.get("ext", "").lower()

                if image_ext not in ["jpeg", "jpg", "png"]:
                    continue

                img_array = np.frombuffer(image_bytes, np.uint8)
                img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if img_cv is None:
                    continue

                file_name = f"page_{page_index+1}_img_{img_index}.jpg"
                out_path = os.path.join(temp_dir, file_name)

                cv2.imwrite(out_path, img_cv)
                count += 1

            except Exception as e:
                continue

    print(f"Extracted {count} images\n")

# ==========================================
# STEP 2: CLASSIFY USING TRAINED MODEL
# ==========================================
def classify_and_filter(temp_dir, final_dir, rejected_dir, filter_dir):

    accepted = 0
    rejected = 0

    for file in os.listdir(temp_dir):

        if not file.lower().endswith(".jpg"):
            continue

        image_path = os.path.join(temp_dir, file)

        label, confidence = classify_image(image_path)

        print(f"{file} → {label.upper()} ({confidence:.2f})")

        image = Image.open(image_path).convert("RGB")

        if label == "reject":
            image.save(os.path.join(rejected_dir, file))
            rejected += 1
        else:
            image.save(os.path.join(final_dir, file))
            image.save(os.path.join(filter_dir, file))
            accepted += 1

    print("\n====================================")
    print("FINAL RESULTS")
    print(f"Accepted: {accepted}")
    print(f"Rejected: {rejected}")
    print("====================================\n")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":

    print("Starting Multi-PDF Image Filtering Pipeline\n")

    for PDF_PATH in PDF_LIST:

        pdf_name = os.path.splitext(os.path.basename(PDF_PATH))[0].strip()
        pdf_dir = os.getcwd()

        BASE_DIR = os.path.join(pdf_dir, pdf_name + "_all_images")
        TEMP_DIR = os.path.join(BASE_DIR, "embedded_images")
        FINAL_DIR = os.path.join(BASE_DIR, "filtered_images")
        REJECTED_DIR = os.path.join(BASE_DIR, "rejected_images")
        FILTER_DIR = os.path.join(pdf_dir, pdf_name)

        os.makedirs(TEMP_DIR, exist_ok=True)
        os.makedirs(FINAL_DIR, exist_ok=True)
        os.makedirs(REJECTED_DIR, exist_ok=True)
        os.makedirs(FILTER_DIR, exist_ok=True)

        print(f"\n Processing: {pdf_name}")
        print(f"Output Folder: {BASE_DIR}\n")

        extract_images(PDF_PATH, TEMP_DIR)
        classify_and_filter(TEMP_DIR, FINAL_DIR, REJECTED_DIR, FILTER_DIR)

    print("All PDFs processed successfully!")