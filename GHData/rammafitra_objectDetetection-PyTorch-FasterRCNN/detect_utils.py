import cv2
import numpy as np
import torchvision.transforms as transforms
from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names

# membuat suatu perbedaan warna pada tiap - tiap kelas
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))
# mendefinisikan tranforms image (mengubah image menjadi tensor)
transform = transforms.Compose([transforms.ToTensor()])

def predict(image, model, device, detection_threshold):
    image = transform(image).to(device) # melakukan tranformasi Image ke Tensor
    image = image.unsqueeze(0)          # menambahkan suatu batch dimension
    outputs = model(image)              # memperoleh prediksi pada Image
    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()] # mendapatkan semua prediksi nama - nama kelas
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()                  # mendapatkan semua prediksi objek (confidance value)
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()                   # mendapatkan semua prediksi boundig box                                                                          
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)   # mendapatkan boxes diatas score batas ambang(threshold)
    return boxes, pred_classes, outputs[0]['labels']

    # print the results individually
    # print(f"BOXES: {outputs[0]['boxes']}")
    # print(f"LABELS: {outputs[0]['labels']}")
    # print(f"SCORES: {outputs[0]['scores']}")

def draw_boxes(boxes, classes, labels, image):
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)           # Membaca gambar dari OpenCV
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]                                        # tiap label memiliki warna yang berbeda
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )                                                                 # membuat Bouding Box
        cv2.putText(image, classes[i], (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, 
                    lineType=cv2.LINE_AA)                                 # membuat Text pada Image
    return image

def Counter(classes, nama_kelas):                                         # logika sederhana untuk menghitung jumlah kelas yang sedang diamati
    count = 0
    for i in classes:
        if i == nama_kelas:
            count = count + 1
    print("jumlah " + nama_kelas + " adalah ", count)        