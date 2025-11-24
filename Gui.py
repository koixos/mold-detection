from tkinter import Tk, Label, Button, Scale, filedialog, HORIZONTAL
import cv2
from PIL import Image, ImageTk

class GUI:
    def __init__(self):
        self.img = None
        self.img_gray = None
        self.root = Tk()
        self.lbl = Label(self.root)

    def configure(self):
        self.root.title("Mold Candidates Detection")
        Button(self.root, text="Select Image", command=self.select_file).pack()
        Label(self.root, text="Threshold").pack()
        Scale(self.root, from_=0, to=255, orient=HORIZONTAL, command=self.update_th).pack()
        self.lbl.pack()

    def run(self):
        self.configure()
        self.root.mainloop()

    def select_file(self):
        filepath = filedialog.askopenfilename(filetypes=[('Image Files', '*.png *.jpg *.jpeg')])
        if not filepath:
            return
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.show_img(img_gray)
        return

    def update_th(self, val):
        if self.img_gray is None:
            return
        _, th = cv2.threshold(self.img_gray, int(val), 255, cv2.THRESH_BINARY)
        self.show_img(th)
        return

    def show_img(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
        im = Image.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(image=im)
        self.lbl.config(image=imgtk)
        self.lbl.image = imgtk
