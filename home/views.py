from django.shortcuts import render
from django.http import StreamingHttpResponse, HttpResponse
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
from keras.layers import Input, Dense
from keras.models import Model
import os
from tensorflow.keras.utils import to_categorical

def inFrame(lst):
    if lst[28].visibility > 0.6 and lst[27].visibility > 0.6 and lst[15].visibility>0.6 and lst[16].visibility>0.6:
          return True
    return False

model = load_model("home/model.h5")
label = np.load("home/labels.npy")

holistic = mp.solutions.pose
holis = holistic.Pose()
drawing = mp.solutions.drawing_utils



def gen_frames():
    cap = cv2.VideoCapture(0)
     
    
    
    while True:
        lst = []

        _, frm = cap.read()

        window = np.zeros((940, 940, 3), dtype="uint8")

        frm = cv2.flip(frm, 1)

        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        frm = cv2.blur(frm, (4, 4))
        if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
            for i in res.pose_landmarks.landmark:
                lst.append(i.x - res.pose_landmarks.landmark[0].x)
                lst.append(i.y - res.pose_landmarks.landmark[0].y)

            lst = np.array(lst).reshape(1, -1)

            p = model.predict(lst)
            pred = label[np.argmax(p)]
            

            if p[0][np.argmax(p)] > 0.75:
                cv2.putText(frm, pred, (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
            else:
                cv2.putText(frm, "Asana is either wrong or not trained", (100, 450), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 3)
        else:
            cv2.putText(frm, "Make Sure Full body is visible", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),
                        3)

        drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS,
                                connection_drawing_spec=drawing.DrawingSpec(color=(255, 255, 255), thickness=6),
                                landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3,
                                                                         thickness=3))

        _, buffer = cv2.imencode('.jpg', frm)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    

        #yield frame, pred  # Yield both frame and prediction


def video_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def home(request):
   
    return render(request, 'home.html')
def aasanPage(request):
    return render(request,'aasans.html')
def tracks(request):
    return render(request,'tracks.html')
def sittingAasanList(request):
    return render(request,'sittingAasanList.html')
def padmasana(request):
    return render(request,'padmasana.html')
def lotus(request):
    return render(request,'lotus.html')
def recliningAasanList(request):
    return render(request,'recliningAasanList.html')
def standingAasanList(request):
    return render(request,'standingAasanList.html')
def Virabhadrasana(request):
    return render(request,'Virabhadrasana.html')
def Vrikshasana(request):
    return render(request,'Vrikshasana.html')



# def home(request):
#     frame, pred = next(gen_frames())  # Get the frame and prediction
#     predAasan = {'predName': pred}  # Create context dictionary
#     return render(request, 'home.html', predAasan)

# def collect_data(request):
#     cap = cv2.VideoCapture(0)

#     name = request.POST.get('asana_name')  # Assuming you have a form to input the asana name.

#     holistic = mp.solutions.pose
#     holis = holistic.Pose()
#     drawing = mp.solutions.drawing_utils

#     X = []
#     data_size = 0

#     while True:
#         lst = []

#         _, frm = cap.read()

#         frm = cv2.flip(frm, 1)

#         res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

#         if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
#             for i in res.pose_landmarks.landmark:
#                 lst.append(i.x - res.pose_landmarks.landmark[0].x)
#                 lst.append(i.y - res.pose_landmarks.landmark[0].y)

#             X.append(lst)
#             data_size = data_size + 1

#         else:
#             cv2.putText(frm, "Make Sure Full body visible", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#         drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS)

#         cv2.putText(frm, str(data_size), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         cv2.imshow("window", frm)

#         if cv2.waitKey(1) == 27 or data_size > 80:
#             cv2.destroyAllWindows()
#             cap.release()
#             break

#     np.save(f"home/{name}.npy", np.array(X))
#     print(np.array(X).shape)

#     return render(request, 'data_collection_success.html')

# def train_model(request):
#     is_init = False
#     size = -1

#     label = []
#     dictionary = {}
#     c = 0

#     for i in os.listdir('home/'):
#         if i.split(".")[-1] == "npy" and not (i.split(".")[0] == "labels"):
#             if not (is_init):
#                 is_init = True
#                 X = np.load(f'home/{i}')
#                 size = X.shape[0]
#                 y = np.array([i.split('.')[0]] * size).reshape(-1, 1)
#             else:
#                 X = np.concatenate((X, np.load(f'home/{i}')))
#                 y = np.concatenate((y, np.array([i.split('.')[0]] * size).reshape(-1, 1)))

#             label.append(i.split('.')[0])
#             dictionary[i.split('.')[0]] = c
#             c = c + 1

#     for i in range(y.shape[0]):
#         y[i, 0] = dictionary[y[i, 0]]
#     y = np.array(y, dtype="int32")

#     y = to_categorical(y)

#     X_new = X.copy()
#     y_new = y.copy()
#     counter = 0

#     cnt = np.arange(X.shape[0])
#     np.random.shuffle(cnt)

#     for i in cnt:
#         X_new[counter] = X[i]
#         y_new[counter] = y[i]
#         counter = counter + 1

#     ip = Input(shape=(X.shape[1]))
#     m = Dense(128, activation="tanh")(ip)
#     m = Dense(64, activation="tanh")(m)
#     op = Dense(y.shape[1], activation="softmax")(m)

#     model = Model(inputs=ip, outputs=op)

#     model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

#     model.fit(X_new, y_new, epochs=80)

#     model.save("model.h5")
#     np.save("labels.npy", np.array(label))

#     return render(request, 'training_success.html')