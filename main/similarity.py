import os
import tqdm
import cv2
from skimage import metrics

def load_images_from_folder(folder_path, target_size=(32, 32)):
    """
    Load all images from a folder and resize them to a target size.
    """
    images = []
    filenames = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath):
            image = cv2.imread(filepath)
            if image is not None:
                image = cv2.resize(image, target_size)
                images.append(image)
                filenames.append(filename)
    return images, filenames

image1 = '/home/dj/course-diffusevae-20241212/samples/ddpm-cifar10-cond-stage1-form1-10000/1000/images/output_gpu_0_0_0_388.png'
image_dir = '/home/dj/course-diffusevae-20241212/datasets/cifar10-jpg-train/'

image1 = cv2.imread(image1)
dataset_images, dataset_filenames = load_images_from_folder(image_dir)
best_score = -1

for image2, filename2 in zip(tqdm.tqdm(dataset_images), dataset_filenames):
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    metric_val = metrics.structural_similarity(image1_gray, image2_gray, full=True)[0]
    
    if metric_val > best_score:
        print("Similarity Score:", round(metric_val, 2), "Path:", filename2)
        best_score = metric_val
