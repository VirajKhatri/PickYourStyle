from PIL import Image,ImageTk
import Tkinter
import tkMessageBox
import os,glob
import cv2
import numpy as np
from utils import CFEVideoConf, image_resize

def run():
    cap = cv2.VideoCapture(0)

    save_path='saved-media/glasses_and_stash.mp4'
    frames_per_seconds=24
    config=CFEVideoConf(cap, filepath=save_path, res='720p')
    out=cv2.VideoWriter(save_path, config.video_type, frames_per_seconds, config.dims)
    face_cascade=cv2.CascadeClassifier('C:\Users\Viraj Khatri\Desktop\Final Project\cascades\data\haarcascade_frontalface_default.xml')
    eyes_cascade=cv2.CascadeClassifier('C:\Users\Viraj Khatri\Desktop\Final Project\cascades\third-party\frontalEyes35x16.xml')
    nose_cascade=cv2.CascadeClassifier('C:\Users\Viraj Khatri\Desktop\Final Project\cascades\third-party\Nose18x15.xml')
    glasses=cv2.imread("images/fun/glasses.png", -1)
    mustache=cv2.imread('images/mustache/mustache.png',-1)
    imgHair=cv2.imread('images/hair/2.png',-1)

    while(True):
        
        ret, frame=cap.read()
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

        for (x, y, w, h) in faces:
            #y=y-70
            roi_gray=gray[y:y+h, x:x+h]
            roi_color=frame[y:y+h, x:x+h]
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 3)
            #try:
            eyes = eyes_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5)
            print eyes
            for (ex, ey, ew, eh) in eyes:
                #cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 3)
                roi_eyes = roi_gray[ey: ey + eh, ex: ex + ew]
                glasses2 = image_resize(glasses.copy(), width=ew)

                gw, gh, gc = glasses2.shape
                for i in range(0, gw):
                    for j in range(0, gh):
                        if glasses2[i, j][3] != 0:
                            roi_color[ey + i, ex + j] = glasses2[i, j]


            nose = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5)
            for (nx, ny, nw, nh) in nose:
                #cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (255, 0, 0), 3)
                roi_nose = roi_gray[ny: ny + nh, nx: nx + nw]
                mustache2 = image_resize(mustache.copy(), width=nw)

                mw, mh, mc = mustache2.shape
                for i in range(0, mw):
                    for j in range(0, mh):
                        if mustache2[i, j][3] != 0:
                            roi_color[ny + int(nh/2.0) + i, nx + j] = mustache2[i, j]

            face = face_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5)
            for (fx, fy, fw, fh) in face:
                fx=fx-40
                fy=fy-120
                #cv2.rectangle(roi_color, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 3)
                roi_face = roi_gray[fy: fy + fh, fx: fx + fw]
                imgHair2 = image_resize(imgHair.copy(), width=fw+int(fw/4))

                hw, hh, hc = imgHair2.shape
                for i in range(0, hw):
                    for j in range(0, hh):
                        if imgHair2[i, j][3] != 0:
                            roi_color[fy+i, fx + j] = imgHair2[i, j]
            #except:
                #continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        out.write(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

root = Tkinter.Tk()
root.title("Choose your style")
root.geometry('{}x{}'.format(500,500))
      
top_frame=Tkinter.Frame(root,bg='lightgrey', width=150,height=150)
mid_frame=Tkinter.Frame(root,bg='darkgrey',width=150,height=150)
btm_frame=Tkinter.Frame(root, bg='gainsboro' ,width=150,height=150)
but_frame=Tkinter.Frame(root, bg='darkgrey' ,width=150,height=50)

root.grid_columnconfigure(1, weight=1)

top_frame.grid(row=0,column=1,sticky=Tkinter.N+Tkinter.W+Tkinter.E)
mid_frame.grid(row=1,column=1,sticky=Tkinter.W+Tkinter.E)
btm_frame.grid(row=2,column=1,sticky=Tkinter.W+Tkinter.E)
but_frame.grid(row=3,column=1)

hair_image_list=[]
os.chdir('C:\Users\Viraj Khatri\Desktop\Final Project\images\hair')
for file in glob.glob('*.png'):
    hair_image_list.append(file)
    
mustache_image_list=[]
os.chdir('C:\Users\Viraj Khatri\Desktop\Final Project\images\mustache')
for file in glob.glob('*.png'):
    mustache_image_list.append(file)

glasses_image_list=[]
os.chdir('C:\Users\Viraj Khatri\Desktop\Final Project\images\glasses')
for file in glob.glob('*.png'):
    glasses_image_list.append(file)

icon_image_list=[]
os.chdir('C:\Users\Viraj Khatri\Desktop\Final Project\images\icon')
for file in glob.glob('*.png'):
    icon_image_list.append(file)
print mustache_image_list
currentH = 0
currentM = 0
currentG = 0

def move(delta):
    global currentG, glasses_image_list
    if not (0 <= currentG + delta < len(glasses_image_list)):
        tkMessageBox.showinfo('End', 'No more image.')
        return
    currentG += delta
    image = Image.open("C:\Users\Viraj Khatri\Desktop\Final Project\images\glasses\\"+glasses_image_list[currentG])
    photo = ImageTk.PhotoImage(image)
    labelB['image'] = photo
    labelB.photo = photo

def move1(delta):
    global currentH, hair_image_list
    if not (0 <= currentH + delta < len(hair_image_list)):
        tkMessageBox.showinfo('End', 'No more image.')
        return
    currentH += delta
    image = Image.open("C:\Users\Viraj Khatri\Desktop\Final Project\images\hair\\"+hair_image_list[currentH])
    photo = ImageTk.PhotoImage(image)
    labelT['image'] = photo
    labelT.photo = photo

def move2(delta):
    global currentM, mustache_image_list
    if not (0 <= currentM + delta < len(mustache_image_list)):
        tkMessageBox.showinfo('End', 'No more image.')
        return
    currentM += delta
    image = Image.open("C:\Users\Viraj Khatri\Desktop\Final Project\images\mustache\\"+mustache_image_list[currentM])
    photo = ImageTk.PhotoImage(image)
    labelM['image'] = photo
    labelM.photo = photo
      
#left
photoB=ImageTk.PhotoImage(file="arow_r.png")
b = Tkinter.Button(btm_frame,bg='grey',image=photoB, command=lambda: move(-1))
b.pack(side=Tkinter.LEFT)
#right
photoB1=ImageTk.PhotoImage(file="arow.png")
b1 = Tkinter.Button(btm_frame,image=photoB1,command=lambda: move(+1))
b1.pack(side=Tkinter.RIGHT)
#picture frame
labelB = Tkinter.Label(btm_frame, bg='grey',compound=Tkinter.TOP)
labelB.pack()
#buttons
frameB = Tkinter.Frame(btm_frame)
frameB.pack()
move(0)

#left
photoM=ImageTk.PhotoImage(file="arow_r.png")
m = Tkinter.Button(mid_frame,bg='grey',image=photoM, command=lambda: move2(-1))
m.pack(side=Tkinter.LEFT)
#right
photoM1=ImageTk.PhotoImage(file="arow.png")
m1 = Tkinter.Button(mid_frame,image=photoM1,command=lambda: move2(+1))
m1.pack(side=Tkinter.RIGHT)
#picture frame
labelM = Tkinter.Label(mid_frame, bg='grey',compound=Tkinter.TOP)
labelM.pack()
#buttons
frameM = Tkinter.Frame(mid_frame)
frameM.pack()
move2(0)

#left
photoT=ImageTk.PhotoImage(file="arow_r.png")
t = Tkinter.Button(top_frame,bg='grey',image=photoT, command=lambda: move1(-1))
t.pack(side=Tkinter.LEFT)
#right
photoT1=ImageTk.PhotoImage(file="arow.png")
t1 = Tkinter.Button(top_frame,image=photoT1,command=lambda: move1(+1))
t1.pack(side=Tkinter.RIGHT)
#picture frame
labelT = Tkinter.Label(top_frame, bg='grey',compound=Tkinter.TOP)
labelT.pack()
#buttons
frameT = Tkinter.Frame(top_frame)
frameT.pack()
move1(0)

#---------------------------------------------------------
clickc=ImageTk.PhotoImage(file="0.PNG")
c = Tkinter.Button(but_frame,bg='grey',image=clickc,command=run)
c.pack(side=Tkinter.LEFT)
#click button---------------------
#------------------------------

root.mainloop()
