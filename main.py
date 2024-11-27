import cv2

facePrototype = "opencv_face_detector.pbtxt" 
faceModel = "opencv_face_detector_uint8.pb"

agePrototype = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderPrototype = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

faceNet = cv2.dnn.readNet(faceModel,facePrototype) #dnn = deep neural network
ageNet = cv2.dnn.readNet(ageModel,agePrototype)
genderNet = cv2.dnn.readNet(genderModel,genderPrototype)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)','(4-6)','(8-12)','(15-20)','(25-32)','(38-43)','(60-100)']
genderList = ['Male','Female']

video = cv2.VideoCapture(0)

def faceBox(faceNet,frame):
    frameHeight=frame.shape[0]
    frameWidth=frame.shape[1]
    blob=cv2.dnn.blobFromImage(frame, 1.0, (227,227), [104,117,123], swapRB=False)
    faceNet.setInput(blob)
    detection=faceNet.forward()
    container = []
    for i in range(detection.shape[2]):
        confidence=detection[0,0,i,2]
        if confidence>0.7:
            x1=int(detection[0,0,i,3]*frameWidth)
            y1=int(detection[0,0,i,4]*frameHeight)
            x2=int(detection[0,0,i,5]*frameWidth)
            y2=int(detection[0,0,i,6]*frameHeight)
            container.append([x1,y1,x2,y2])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),1)

    return frame, container


while True:
    ret,frame=video.read()
    frame, container=faceBox(faceNet,frame)
    for contain in container:
        face=frame[contain[1]:contain[3], contain[0]:contain[2]]
        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        
        genderNet.setInput(blob)
        genderPredictions=genderNet.forward()
        gender=genderList[genderPredictions[0].argmax()]

        ageNet.setInput(blob)
        agePredictions=ageNet.forward()
        age=ageList[agePredictions[0].argmax()]

        lable = "{},{}".format(gender, age)
        cv2.putText(frame, lable, (contain[0], contain[1]-10), cv2.FONT_HERSHEY_PLAIN, 0.9, (0,255,0), 2)
    cv2.imshow("Age-Gender",frame)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break
video.release()
cv2.destroyAllWindows()