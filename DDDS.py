import customtkinter as ctk
import torch
import numpy as np
import cv2
from PIL import Image, ImageTk
import vlc
import random

class DrowsyBoiApp:
    def _init_(self):
        self.app = ctk.CTk()
        self.app.geometry("600x600")
        self.app.title("Drowsy Boi 4.0")
        ctk.set_appearance_mode("dark")

        self.counter = 0

        self.vidFrame = ctk.CTkFrame(master=self.app, height=480, width=600)
        self.vidFrame.pack()
        self.vid = ctk.CTkLabel(master=self.vidFrame)
        self.vid.pack()

        self.counterLabel = ctk.CTkLabel(master=self.app, text=str(self.counter), height=40, width=120, font=("Arial", 20), text_color="white", fg_color="teal")
        self.counterLabel.pack(pady=10)

        self.resetButton = ctk.CTkButton(master=self.app, text="Reset Counter", command=self.reset_counter, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="teal")
        self.resetButton.pack()

        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp9/weights/last.pt', force_reload=True)
        self.cap = cv2.VideoCapture(0)  # Changed to 0 for the primary camera

        self.detect()
        self.app.mainloop()

    def reset_counter(self):
        self.counter = 0
        self.counterLabel.configure(text=str(self.counter))

    def detect(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame")
            self.app.after(10, self.detect)
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(frame_rgb)
        img = np.squeeze(results.render())

        if len(results.xywh[0]) > 0:
            dconf = results.xywh[0][0][4]
            dclass = results.xywh[0][0][5]

            if dconf.item() > 0.85 and dclass.item() == 1.0:
                filechoice = random.choice([1, 2, 3])
                p = vlc.MediaPlayer(f"{filechoice}.wav")
                if p is not None:
                    p.play()
                    self.counter += 1
                else:
                    print(f"Failed to load media file: {filechoice}.wav")

        imgarr = Image.fromarray(img)
        self.imgtk = ImageTk.PhotoImage(imgarr)
        self.vid.configure(image=self.imgtk)
        self.counterLabel.configure(text=str(self.counter))
        self.vid.after(10, self.detect)

# Run the application
DrowsyBoiApp()
