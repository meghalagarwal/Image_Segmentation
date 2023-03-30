import os
import cv2
import face_recognition
import numpy as np

class RecognitionOperations():

    def save_image_to_directory(self, image, name, image_name):
        path = "./output/"+name
        if os.path.exists(path):
            pass
        else:
            os.mkdir(path)
        cv2.imwrite(path+"/"+image_name, image)

    def create_encodings(self, image):
        #Find face locations for all faces in an image
        face_locations = face_recognition.face_locations(image)
        # Create encodines for all faces in an image
        known_encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)
        return known_encodings, face_locations

    def compare_face_encodings(self, unknown_encoding, known_encodings, known_names):
        duplicate_name = ""
        distance = 0.0
        matches = face_recognition.compare_faces(known_encodings, unknown_encoding, tolerance=0.5)
        face_distances = face_recognition.face_distance(known_encodings, unknown_encoding)
        # print(face_distances)
        best_match_index = np.argmin(face_distances)
        # print(best_match_index)
        distance = face_distances[best_match_index]

        if matches[best_match_index]:
            accept_bool = True
            duplicate_name = known_names
        else:
            accept_bool = False
            duplicate_name = ""
        return accept_bool, duplicate_name, distance

    def people_images(self, people_img: str, dataset_img: str , people_path: str):
        # Read image
        image = cv2.imread(people_path+people_img)
        name = people_img.split(".")[0]
        #Resize
        image = cv2.resize(image, (0,0), fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)        
        # Get locations and encodines
        person_encs, person_locs = self.create_encodings(image)

        #Read image
        image = cv2.imread(dataset_img)
        orig = image.copy()
        # Resize
        image = cv2.resize(image, (780, 540), fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)
        # Get locations and encodines
        encs, locs = self.create_encodings(image)
        known_flag = 0
        for loc in locs:
            top, right, bottom, left = loc
            unknown_encoding = encs[0]
            accept_bool, duplicate_name, distance = self.compare_face_encodings(unknown_encoding, person_encs, name)

            if accept_bool:
                self.save_image_to_directory(orig, duplicate_name, dataset_img.split('/')[-1])
                known_flag = 1
        if known_flag == 1:
            print("Match Found")