import os
import glob
import xml.etree.ElementTree as ET
from PIL import Image
import kagglehub

def main():
    path = kagglehub.dataset_download("andrewmvd/road-sign-detection")
    print("Kaggle dataset path:", path)

    images_dir = os.path.join(path, "images")
    ann_dir    = os.path.join(path, "annotations")

    assert os.path.isdir(images_dir), f"이미지 폴더 없음: {images_dir}"
    assert os.path.isdir(ann_dir), f"어노테이션 폴더 없음: {ann_dir}"

    out_root = r"D:\Python\simp\sign4_img"
    class_names = ["crosswalk", "speedlimit", "stop", "trafficlight"]
    for c in class_names:
        os.makedirs(os.path.join(out_root, c), exist_ok=True)

    xml_files = glob.glob(os.path.join(ann_dir, "*.xml"))
    print(f"Found {len(xml_files)} annotation files")

    valid_map = {
        "crosswalk": "crosswalk",
        "speedlimit": "speedlimit",
        "stop": "stop",
        "trafficlight": "trafficlight",
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
