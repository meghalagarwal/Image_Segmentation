import os
from pathlib import Path
from joblib import Parallel, delayed
from FaceRecognitionOperations.operations import RecognitionOperations

def main():
    face_detection = RecognitionOperations()
    people_path = "./PersonalImages/"
    dataset_input_dir = Path.cwd()/"DatasetImages"
    dataset_images_list = list(dataset_input_dir.rglob('*.JPG'))
    count = 0
    for _ in os.listdir(people_path):
        count+=1
    
    Parallel(n_jobs=count)(delayed(face_detection.people_images)(people_img=people_img, dataset_img=str(dataset_img), people_path=people_path) for dataset_img in dataset_images_list for people_img in os.listdir(people_path))

    print("completed")

if __name__ == "__main__":
    main()