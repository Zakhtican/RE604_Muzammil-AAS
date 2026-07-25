import os
import csv
import base64
import requests
import re
import json

# ==========================================
# KONFIGURASI
# ==========================================
DATASET_BASE = "Indonesian License Plate Recognition Dataset"
IMAGES_DIR = os.path.join(DATASET_BASE, "images", "test")
LABELS_DIR = os.path.join(DATASET_BASE, "labels", "test")
CLASSES_FILE = os.path.join(DATASET_BASE, "classes.names")
OUTPUT_CSV = "hasil_evaluasi_ocr_lmstudio.csv"

LMSTUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
MODEL_NAME = "moondream-2b-2025-04-14"
PROMPT = "Extract the license plate number from this image. Output only the alphanumeric characters, no spaces, no punctuation."

# ==========================================
# FUNGSI GROUND TRUTH & CER
# ==========================================
def load_class_names(classes_path):
    if not os.path.exists(classes_path):
        return []
    with open(classes_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def get_ground_truth_from_label(label_path, class_names):
    if not os.path.exists(label_path):
        return ""
    annotations = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                annotations.append((x_center, class_id))
    annotations.sort(key=lambda item: item[0])
    gt_chars = []
    for _, class_id in annotations:
        if class_id < len(class_names):
            gt_chars.append(class_names[class_id])
    return "".join(gt_chars).upper()

def calculate_cer(ground_truth, prediction):
    n = len(ground_truth)
    m = len(prediction)
    if n == 0:
        return 1.0 if m > 0 else 0.0
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if ground_truth[i-1] == prediction[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[n][m] / n

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def extract_plate_number(text):
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
    pattern = re.compile(r'([A-Z]{1,2})([0-9]{1,4})([A-Z]{0,3})')
    matches = pattern.findall(cleaned)
    if matches:
        best = max(matches, key=lambda x: len(''.join(x)))
        return ''.join(best)
    if 3 <= len(cleaned) <= 10:
        return cleaned
    return ""

# ==========================================
# FALLBACK OCR (EasyOCR)
# ==========================================
def ocr_with_easyocr(img_path):
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False)
        result = reader.readtext(img_path, detail=0, paragraph=False)
        raw = ' '.join(result).upper()
        return extract_plate_number(raw)
    except Exception as e:
        print(f"EasyOCR error: {e}")
        return ""

# ==========================================
# FUNGSI PRINT SUMMARY (TANPA SIMBOL)
# ==========================================
def print_summary(results):
    total = len(results)
    if total == 0:
        print("Tidak ada data.")
        return

    cer_scores = [r['CER_score'] for r in results]
    avg_cer = sum(cer_scores) / total

    perfect = [r for r in results if r['CER_score'] == 0.0]
    success = [r for r in results if r['CER_score'] <= 0.5]
    failed = [r for r in results if r['CER_score'] > 0.5]

    exact_match = len(perfect)
    accuracy_exact = exact_match / total * 100

    sorted_results = sorted(results, key=lambda x: x['CER_score'])
    best_5 = sorted_results[:5]
    worst_5 = sorted_results[-5:][::-1]

    # Ringkasan
    print("\nRINGKASAN EVALUASI OCR")
    print(f"Total gambar               : {total}")
    print(f"Rata-rata CER              : {avg_cer:.4f}")
    print(f"Exact match (CER=0)        : {exact_match}")
    print(f"Accuracy (exact)           : {accuracy_exact:.1f} persen")
    print(f"Jumlah CER <= 0.5 (berhasil): {len(success)} ({len(success)/total*100:.1f} persen)")
    print(f"Jumlah CER > 0.5 (gagal)   : {len(failed)} ({len(failed)/total*100:.1f} persen)")

    # Tabel 5 terbaik
    print("\n5 CONTOH SUKSES (CER terendah)")
    print("No  Gambar               Ground Truth      Prediksi          CER")
    for i, r in enumerate(best_5, 1):
        print(f"{i:<3} {r['image']:<20} {r['ground_truth']:<16} {r['prediction']:<16} {r['CER_score']:.4f}")

    # Tabel 5 terburuk
    print("\n5 CONTOH GAGAL (CER tertinggi)")
    print("No  Gambar               Ground Truth      Prediksi          CER")
    for i, r in enumerate(worst_5, 1):
        print(f"{i:<3} {r['image']:<20} {r['ground_truth']:<16} {r['prediction']:<16} {r['CER_score']:.4f}")

# ==========================================
# MAIN
# ==========================================
def main():
    # Cek koneksi
    try:
        resp = requests.get(f"{LMSTUDIO_BASE_URL}/models", timeout=5)
        if resp.status_code != 200:
            print(f"Server merespon {resp.status_code}. Cek LMStudio.")
        else:
            print("LMStudio server terhubung.")
    except Exception as e:
        print(f"Tidak dapat terhubung ke LMStudio. Error: {e}")

    class_names = load_class_names(CLASSES_FILE)
    if not os.path.exists(IMAGES_DIR):
        print(f"Folder {IMAGES_DIR} tidak ditemukan.")
        return

    image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    image_files.sort()
    results = []
    total = len(image_files)

    # Test satu gambar
    print("Mencoba satu gambar untuk verifikasi...")
    test_img = image_files[0]
    test_path = os.path.join(IMAGES_DIR, test_img)
    base64_img = encode_image_to_base64(test_path)
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                ]
            }
        ],
        "temperature": 0.0,
        "max_tokens": 50
    }
    try:
        response = requests.post(f"{LMSTUDIO_BASE_URL}/chat/completions", json=payload, timeout=60)
        print(f"Status code: {response.status_code}")
        print(f"Response text (first 500 chars): {response.text[:500]}")
        if response.status_code == 200:
            data = response.json()
            print("JSON response:", json.dumps(data, indent=2)[:500])
        else:
            print("Gagal. Lanjutkan dengan fallback EasyOCR.")
    except Exception as e:
        print(f"Error test: {e}")

    use_fallback = input("\nApakah ingin menggunakan EasyOCR sebagai fallback? (y/n): ").lower() == 'y'

    for idx, img_name in enumerate(image_files, 1):
        img_path = os.path.join(IMAGES_DIR, img_name)
        base_name = os.path.splitext(img_name)[0]
        label_path = os.path.join(LABELS_DIR, f"{base_name}.txt")
        ground_truth = get_ground_truth_from_label(label_path, class_names)
        print(f"\n[{idx}/{total}] {img_name} | GT: {ground_truth}")

        prediction = ""
        if not use_fallback:
            base64_img = encode_image_to_base64(img_path)
            payload = {
                "model": MODEL_NAME,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": PROMPT},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                        ]
                    }
                ],
                "temperature": 0.0,
                "max_tokens": 50
            }
            try:
                response = requests.post(f"{LMSTUDIO_BASE_URL}/chat/completions", json=payload, timeout=60)
                if response.status_code == 200:
                    data = response.json()
                    if "choices" in data:
                        raw = data["choices"][0]["message"]["content"].strip()
                        prediction = extract_plate_number(raw)
                        print(f"VLM raw: {raw}")
                        print(f"VLM pred: {prediction}")
                    else:
                        print("VLM response tidak memiliki 'choices'")
                else:
                    print(f"VLM HTTP error: {response.status_code}")
            except Exception as e:
                print(f"VLM error: {e}")

        if not prediction and use_fallback:
            print("Menggunakan EasyOCR fallback...")
            prediction = ocr_with_easyocr(img_path)
            print(f"EasyOCR pred: {prediction}")

        if not prediction:
            prediction = ""
            print("Prediksi kosong, CER akan 1.0")

        cer_score = calculate_cer(ground_truth, prediction) if prediction else 1.0
        print(f"CER: {cer_score:.4f}")

        results.append({
            "image": img_name,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "CER_score": round(cer_score, 4)
        })

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['image', 'ground_truth', 'prediction', 'CER_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    avg_cer = sum(r['CER_score'] for r in results) / len(results) if results else 1.0
    print(f"\nSelesai. Rata-rata CER: {avg_cer:.4f}")
    print(f"Hasil di: {OUTPUT_CSV}")

    # Panggil print_summary di sini
    print_summary(results)

if __name__ == "__main__":
    main()