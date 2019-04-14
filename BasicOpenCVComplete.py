import cv2
faceCascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def create_dataset(img,id,img_id):
        cv2.imwrite("data/pic."+str(id)+"."+str(img_id)+".jpg",img)
        

def draw_boundary(img,classifier,scaleFactor,minNeighbors,color,clf):
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        features=classifier.detectMultiScale(gray,scaleFactor,minNeighbors)
        coords=[]
        for (x,y,w,h) in features:
                cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
                id,con= clf.predict(gray[y:y+h,x:x+w])
                
                if con >= 30 and con<=100 :
                    cv2.putText(img,"Cherprang",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
                else :
                    cv2.putText(img,"Bill Gates",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

                if (con < 100):
                    con = "  {0}%".format(round(100 - con))
                else:
                    con = "  {0}%".format(round(100 - con))

                print(str(con))
                    
                coords=[x,y,w,h]
                
        return img,coords 
        
def detect(img,faceCascade,img_id,clf):
        img,coords=draw_boundary(img,faceCascade,1.1,10,(0,255,0),clf)
        if len(coords)== 4 :
                id=1
                result = img[coords[1]:coords[1]+coords[3],coords[0]:coords[0]+coords[2]]
                #create_dataset(result,id,img_id)
        return img

img_id=0       
cap = cv2.VideoCapture("BNK48.mp4")

clf=cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.xml")

while (True):
        ret,frame = cap.read()
        frame=detect(frame,faceCascade,img_id,clf)
        cv2.imshow('frame',frame)
        img_id+=1
        if(cv2.waitKey(1) & 0xFF== ord('q')):
            break
cap.release()
cv2.destroyAllWindows()


