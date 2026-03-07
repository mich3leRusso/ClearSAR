import cv2
import os

image_dir = "/home/simone/myprojects/ClearSAR/ClearSAR/yolo_dataset/images/train"
label_dir = "/home/simone/myprojects/ClearSAR/ClearSAR/yolo_dataset/labels/train"

for img_file in sorted(os.listdir(image_dir)):
    if not img_file.endswith((".jpg", ".png")):
        continue

    if not img_file.endswith(("90.png")):
        continue
        
    img_path = os.path.join(image_dir, img_file)
    label_path = os.path.join(label_dir, img_file.replace(".jpg", ".txt").replace(".png",".txt"))

    img = cv2.imread(img_path)
    h, w, _ = img.shape

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f.readlines():
                class_id, x, y, bw, bh = map(float, line.strip().split())

                # Convert YOLO normalized coords → pixel coords
                x1 = int((x - bw/2) * w)
                y1 = int((y - bh/2) * h)
                x2 = int((x + bw/2) * w)
                y2 = int((y + bh/2) * h)

                cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(img, str(int(class_id)), (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    
        print(f"Showing {img_path}")
        cv2.imshow("bbox", img)
        cv2.waitKey(0)

cv2.destroyAllWindows()