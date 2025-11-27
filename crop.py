
# 01_crop_from_voc.py
import os
import glob
import xml.etree.ElementTree as ET
from PIL import Image
import kagglehub

def main():
    # 1) Kaggle 데이터 다운로드
    path = kagglehub.dataset_download("andrewmvd/road-sign-detection")
    print("Kaggle dataset path:", path)

    # ❗ 다운로드한 폴더 안에 실제 폴더 이름이 "images", "annotations"이 맞는지 확인 필요
    images_dir = os.path.join(path, "images")
    ann_dir    = os.path.join(path, "annotations")

    assert os.path.isdir(images_dir), f"이미지 폴더 없음: {images_dir}"
    assert os.path.isdir(ann_dir), f"어노테이션 폴더 없음: {ann_dir}"

    # 2) 출력 폴더 생성
    out_root = "D:/Python/simp/traffic_sign_img" 
    class_names = ["stop", "speedlimit", "trafficlight", "crosswalk"]
    for c in class_names:
        os.makedirs(os.path.join(out_root, c), exist_ok=True)

    # 3) XML 파일 파싱해서 각 object만 crop해서 저장
    xml_files = glob.glob(os.path.join(ann_dir, "*.xml"))
    print(f"Found {len(xml_files)} annotation files")

    valid_map = {
        "stop": "stop",
        "speedlimit": "speedlimit",
        "trafficlight": "trafficlight",
        "traffic light": "trafficlight",
        "crosswalk": "crosswalk",
    }

    crop_count = {c: 0 for c in class_names}

    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        filename = root.find("filename").text
        img_path = os.path.join(images_dir, filename)
        if not os.path.exists(img_path):
            print(f"[WARN] 이미지 없음: {img_path}")
            continue

        img = Image.open(img_path).convert("RGB")

        for obj in root.findall("object"):
            raw_name = obj.find("name").text.strip().lower()
            key = raw_name.replace(" ", "")

            if key not in valid_map:
                continue

            cls_name = valid_map[key]

            bbox = obj.find("bndbox")
            xmin = int(float(bbox.find("xmin").text))
            ymin = int(float(bbox.find("ymin").text))
            xmax = int(float(bbox.find("xmax").text))
            ymax = int(float(bbox.find("ymax").text))

            # 범위 체크
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(img.width, xmax)
            ymax = min(img.height, ymax)

            if xmax <= xmin or ymax <= ymin:
                continue

            crop = img.crop((xmin, ymin, xmax, ymax))

            save_dir = os.path.join(out_root, cls_name)
            idx = crop_count[cls_name]
            save_path = os.path.join(save_dir, f"{cls_name}_{idx:05d}.png")

            crop.save(save_path)
            crop_count[cls_name] += 1

    print("===== DONE =====")
    for c in class_names:
        print(f"{c}: {crop_count[c]} images saved")

if __name__ == "__main__":
    main()
