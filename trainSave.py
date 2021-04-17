import face_recognition
import cv2
import os
print(cv2.__version__)
 
Encodings=[]
Names=[]
j=0


def scan_known_images():
    known_image_dir='/home/rami/Desktop/test_face/jetson_nano_demo/face_recognition/face_db'
    for root, dirs, files in os.walk(known_image_dir):
        print(files)
        for file in files:
            path=os.path.join(root,file)
            print(path)
            name=os.path.splitext(file)[0]
            print(name)
            person=face_recognition.load_image_file(path)
            encoding=face_recognition.face_encodings(person)[0]
            Encodings.append(encoding)
            Names.append(name)
    print(Names)
    
font=cv2.FONT_HERSHEY_SIMPLEX
 
def scan_unknow_images():
    unknown_image_dir='/home/rami/Desktop/test_face/jetson_nano_demo/face_recognition/unknown'
    vec = []
    for root,dirs, files in os.walk(unknown_image_dir):
        for file in files:
            print(root)
            print(file)
            testImagePath=os.path.join(root,file)
            testImage=face_recognition.load_image_file(testImagePath)
            facePositions=face_recognition.face_locations(testImage)
            allEncodings=face_recognition.face_encodings(testImage,facePositions)
            testImage=cv2.cvtColor(testImage,cv2.COLOR_RGB2BGR)
    
            for (top,right,bottom,left),face_encoding in zip(facePositions,allEncodings):
                name='Known'
                matches=face_recognition.compare_faces(Encodings,face_encoding)
                if True in matches:
                    first_match_index=matches.index(True)
                    name=Names[first_match_index]
                    print("True match")
                    vec.append(1)
            vec.append(0)
        #cv2.imshow('Picture', testImage)
        #cv2.moveWindow('Picture',0,0)
        print(vec)
        confidence = calculate_statistic(vec)
        print(confidence)
        return confidence
        if cv2.waitKey(0)==ord('q'):
            cv2.destroyAllWindows()

def calculate_statistic(vec):
    len_vec = len(vec) # 0 - number of pictures   1 - true result 
    true_results = countOccurrences(vec,len_vec,1)
    number_of_pictures = countOccurrences(vec,len_vec,0)
    confidence = true_results/number_of_pictures
    return confidence
    
def countOccurrences(arr, n, x):
    res = 0
    for i in range(n):
        if x == arr[i]:
            res += 1
    return res
  



