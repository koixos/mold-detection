from ..state import AppState

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2


class Portfolio(tk.Frame):
    def __init__(self, parent, state: AppState):
        super().__init__(parent, bg="#ededed")

        self.state = state

        self._img_refs = {} 
        self._build_ui()


    def refresh(self):
        active_idx = self.state.active_index
        img_st = self.state.active()

        self.cv_orig.delete("all")
        self.cv_pre.delete("all")
        self.cv_res.delete("all")
        
        if img_st:
            self._draw_image(self.cv_orig, img_st.original)
            if img_st.preprocessed.img is not None:
                self._draw_image(self.cv_pre, img_st.preprocessed.img)
            if img_st.detected is not None:
                self._draw_image(self.cv_res, img_st.detected)

        # Update Buttons
        if not self.state.images:
            self.btn_prev.config(state="disabled")
            self.btn_next.config(state="disabled")
        else:
            self.btn_prev.config(state="normal" if active_idx > 0 else "disabled")
            self.btn_next.config(state="normal" if active_idx < len(self.state.images) - 1 else "disabled")


    # ====================== UI ====================== 

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Col 1: Original Image + Buttons
        col1 = tk.Frame(self, bg="#ffffff")
        col1.grid(row=0, column=0, sticky="nsew", padx=4, pady=12)
        
        self._build_header(col1, "Original Image", lambda: self._save_single(lambda s: s.original, lambda s: s.filename))
        
        c1_frame = tk.Frame(col1, bg="#f5f5f5")
        c1_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        
        self.cv_orig = self._build_canvas(c1_frame)
        
        # Navigation Buttons
        nav_frame = tk.Frame(col1, bg="#ffffff")
        nav_frame.pack(fill="x", pady=(0, 8))
        
        btn_font = ("Segoe UI", 12, "bold")
        self.btn_prev = tk.Button(nav_frame, text=" < ", font=btn_font, command=self._prev, relief="flat", bg="#e0e0e0", width=5)
        self.btn_prev.pack(side="left", padx=20)
        
        self.btn_next = tk.Button(nav_frame, text=" > ", font=btn_font, command=self._next, relief="flat", bg="#e0e0e0", width=5)
        self.btn_next.pack(side="right", padx=20)

        # Col 2: Preprocessed
        col2 = tk.Frame(self, bg="#ffffff")
        col2.grid(row=0, column=1, sticky="nsew", padx=4, pady=12)
        self._build_header(col2, "Preprocessed", lambda: self._save_single(lambda s: s.preprocessed.img, lambda s: f"preprocessed_{s.filename}"))
        self.cv_pre = self._build_canvas(col2)

        # Col 3: Result
        col3 = tk.Frame(self, bg="#ffffff")
        col3.grid(row=0, column=2, sticky="nsew", padx=4, pady=12)
        self._build_header(col3, "Result", lambda: self._save_single(lambda s: s.detected, lambda s: f"output_{s.filename}"))
        self.cv_res = self._build_canvas(col3)


    def _build_header(self, parent, text, save_cmd):
        header = tk.Frame(parent, bg="#ffffff")
        header.pack(fill="x", padx=8, pady=6)
        tk.Label(header, text=text, bg="#ffffff", fg="#222", font=("Segoe UI", 11, "bold")).pack(side="left")
        
        btn = tk.Label(header, text="\U0001f4be", bg="#ffffff", fg="#444", cursor="hand2")
        btn.pack(side="right")
        btn.bind("<Button-1>", lambda e: save_cmd())


    def _build_canvas(self, parent):
        canvas = tk.Canvas(parent, bg="#f5f5f5", highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        return canvas


    # ====================== LOGIC ====================== 

    def _prev(self):
        if self.state.images:
            self.state.set_active(max(0, self.state.active_index - 1))


    def _next(self):
        if self.state.images:
            self.state.set_active(min(len(self.state.images)-1, self.state.active_index + 1))


    def _draw_image(self, canvas: tk.Canvas, img_bgr):
        if img_bgr is None: return

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        cw = int(canvas.winfo_width())
        ch = int(canvas.winfo_height())
        if cw <= 1: cw = 1
        if ch <= 1: ch = 1

        scale = min(cw / w, ch / h, 1.0)
        nw, nh = int(w * scale), int(h * scale)

        img_resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        img_pil = Image.fromarray(img_resized)
        img_tk = ImageTk.PhotoImage(img_pil)

        canvas.create_image(cw // 2, ch // 2, image=img_tk, anchor="center")
        self._img_refs[canvas] = img_tk 


    # ====================== SAVE ====================== 

    def _save_single(self, getter, name_fn):
        img_st = self.state.active()
        if img_st is None: return
        img = getter(img_st)
        if img is None: return
        
        default = name_fn(img_st)
        if callable(default): default = default(img_st)

        path = filedialog.asksaveasfilename(
            initialfile=default,
            defaultextension=".png",
            filetypes=[("PNG", "*.png")]
        )
        if path:
            cv2.imwrite(path, img) 