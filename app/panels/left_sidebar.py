from ..defs import EXTS
from ..state import ImageState, AppState

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os


class LeftSidebar(tk.Frame):
    def __init__(self, parent, state: AppState, width=380):
        super().__init__(parent, width=width, bg="#f2f2f2")
        self.pack_propagate(False)

        # Scrollbar Style
        try:
            style = ttk.Style()
            style.layout(
                'NoArrows.Vertical.TScrollbar', [(
                    'Vertical.Scrollbar.trough', {
                        'children': [(
                            'Vertical.Scrollbar.thumb', {
                                'expand': '1', 
                                'sticky': 'nswe'
                            }
                        )], 
                        'sticky': 'nswe'
                    }
                )]
            )
        except: pass

        self.state = state
        self.is_collapsed = False
        self.expanded_width = width
        self.collapsed_width = 70
        self._img_refs = [] 

        self._build_ui()

    
    def refresh(self):
        self._update_header()
        for w in self.inner.winfo_children(): w.destroy()
        self._img_refs = []

        for idx, img_st in enumerate(self.state.images):
            self._add_row(idx, img_st)
    

    # ====================== UI ======================

    def _build_ui(self):
        # Top Header Area
        self.top_frame = tk.Frame(self, bg="#f2f2f2")
        self.top_frame.pack(fill="x", padx=4, pady=8)
        
        # Toggle Button
        self.btn_toggle = tk.Button(
            self.top_frame, 
            text="<<", 
            command=self._toggle_sidebar,
            relief="flat",
            bg="#e0e0e0",
            fg="#555",
            cursor="hand2",
            width=3
        )
        self.btn_toggle.pack(side="left")

        # Title
        self.lbl_title = tk.Label(self.top_frame, text="Images", bg="#f2f2f2", fg="#333", font=("Segoe UI", 12, "bold"))
        self.lbl_title.pack(side="left", padx=8)

        # Load Button
        self.btn_load = tk.Button(
            self.top_frame, 
            text="+", 
            command=self._load_images, 
            relief="flat",
            bg="#e6e6e6",
            fg="#222",
            cursor="hand2",
            width=3
        )
        self.btn_load.pack(side="right")

        # Main Body (Scrollbar + Content)
        self.body_frame = tk.Frame(self, bg="#ffffff")
        self.body_frame.pack(fill="both", expand=True, padx=4, pady=4)

        # Scrollbar
        self.scrollbar = ttk.Scrollbar(self.body_frame, orient="vertical", style='NoArrows.Vertical.TScrollbar')
        self.scrollbar.pack(side="right", fill="y")
        
        # Content Frame (Header + List)
        self.content_frame = tk.Frame(self.body_frame, bg="#ffffff")
        self.content_frame.pack(side="left", fill="both", expand=True)

        # Table Header
        self.tbl_head = tk.Frame(self.content_frame, bg="#e0e0e0")
        self.tbl_head.pack(fill="x", padx=0, pady=0)

        # List
        self.canvas = tk.Canvas(self.content_frame, bg="#ffffff", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.configure(command=self.canvas.yview)

        self.inner = tk.Frame(self.canvas, bg="#ffffff")
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        def _configure_width(event):
            self.canvas.itemconfig(self.canvas.create_window((0,0), window=self.inner, anchor='nw'), width=event.width)
        
        self.canvas.bind("<Configure>", _configure_width)
        
        # Mouse Scroll
        self.bind_all("<MouseWheel>", self._on_mousewheel)

        self._update_header()


    def _update_header(self):
        for w in self.tbl_head.winfo_children(): w.destroy()
        
        if self.is_collapsed:
            self.lbl_title.pack_forget()
            self.btn_load.pack_forget()
            self.btn_toggle.config(text=">>")
        else:
            self.lbl_title.pack(side="left", padx=8)
            self.btn_load.pack(side="right")
            self.btn_toggle.config(text="<<")

        if self.is_collapsed:
            self.tbl_head.grid_columnconfigure(0, weight=1) 
            for c in [1, 2, 3]: self.tbl_head.grid_columnconfigure(c, weight=0, minsize=0)
            
            f = tk.Frame(self.tbl_head, bg="#e0e0e0", height=24, bd=0, highlightbackground="#f0f0f0")
            f.pack_propagate(False)
            f.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
            tk.Label(f, text="Images", bg="#e0e0e0", font=("Segoe UI", 9, "bold")).pack(expand=True)
        else:
            self.tbl_head.grid_columnconfigure(0, weight=1, uniform="h_cols") 
            self.tbl_head.grid_columnconfigure(1, weight=1, uniform="h_cols") 
            self.tbl_head.grid_columnconfigure(2, weight=1, uniform="h_cols")
            self.tbl_head.grid_columnconfigure(3, weight=0, minsize=30) # Delete Col
            
            def h_lbl(text, col):
                f = tk.Frame(self.tbl_head, bg="#e0e0e0", height=24, bd=0, highlightthickness=1, highlightbackground="#f0f0f0")
                f.pack_propagate(False)
                f.grid(row=0, column=col, sticky="nsew", padx=0, pady=0)
                tk.Label(f, text=text, bg="#e0e0e0", font=("Segoe UI", 9, "bold")).pack(expand=True)

            h_lbl("Original", 0)
            h_lbl("Preprocessed", 1)
            h_lbl("Result", 2)
            
            # Delete All Button
            f_del = tk.Frame(self.tbl_head, bg="#e0e0e0", bd=0, highlightthickness=1, highlightbackground="#f0f0f0") 
            f_del.grid(row=0, column=3, sticky="nsew")
            
            btn_del_all = tk.Label(f_del, text="🗑️", bg="#e0e0e0", fg="#555", cursor="hand2", font=("Segoe UI", 8))
            btn_del_all.pack(expand=True, fill="both")
            btn_del_all.bind("<Button-1>", lambda e: self._delete_all())
            
            # Tooltip
            def show_tooltip(e):
                self._tooltip = tk.Toplevel(f_del)
                self._tooltip.wm_overrideredirect(True)
                x, y, _, _ = f_del.bbox("all") 
                root_x = f_del.winfo_rootx() + 20
                root_y = f_del.winfo_rooty() - 20
                self._tooltip.wm_geometry(f"+{root_x}+{root_y}")
                tk.Label(self._tooltip, text="Delete All", bg="#ffffe0", bd=1, relief="solid", font=("Segoe UI", 8)).pack()

            def hide_tooltip(e):
                if hasattr(self, '_tooltip'):
                    self._tooltip.destroy()
                    del self._tooltip

            btn_del_all.bind("<Enter>", show_tooltip)
            btn_del_all.bind("<Leave>", hide_tooltip)


    def _on_mousewheel(self, event):
        try:
            x, y = self.winfo_pointerxy()
            widget = self.winfo_containing(x, y)
            if widget and str(widget).startswith(str(self)):
                self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        except: pass


    # ====================== LOGIC ====================== 

    def _toggle_sidebar(self):
        self.is_collapsed = not self.is_collapsed
        target_w = self.collapsed_width if self.is_collapsed else self.expanded_width
        self.configure(width=target_w)
        self.refresh()


    def _load_image(self, path: str):
        ext = os.path.splitext(path)[1].lower()
        if ext not in EXTS:
            raise ValueError(f"Unsupported image format: {ext}")
        
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        
        return img


    def _load_images(self):
        paths = filedialog.askopenfilenames(title="Select Images", filetypes=[("Images", "*" + " *".join(EXTS))])
        for path in paths:
            try:
                img = self._load_image(path)
                state = ImageState(path=path, filename=os.path.basename(path), original=img)
                self.state.add_image(state)
            except Exception as e:
                messagebox.showerror("Load error", str(e))
        
        if paths and self.state.images:
            self.state.set_active(len(self.state.images) - 1)


    def _delete_all(self):
        if self.state.images:
            if messagebox.askyesno("Confirm", "Delete all images?"):
                self.state.clear_images()


    def _add_row(self, index, img_st: ImageState):
        is_active = (index == self.state.active_index)
        bg = "#dfefff" if is_active else "#ffffff"
        
        row = tk.Frame(self.inner, bg=bg, pady=2, bd=0)
        row.pack(fill="x", pady=1)
        
        def make_cell(col, width=None):
            f = tk.Frame(row, bg=bg, bd=0, highlightthickness=1, highlightbackground="#f0f0f0")
            if width:
                f.configure(width=width)
                f.pack_propagate(False)
            f.grid(row=0, column=col, sticky="nsew", padx=0, pady=0)
            f.bind("<Button-1>", lambda e, i=index: self._set_active(i))
            return f

        def add_icon(parent, img):
            if img is not None:
                try:
                    thumb_bgr = cv2.resize(img, (32, 32))
                    thumb_rgb = cv2.cvtColor(thumb_bgr, cv2.COLOR_BGR2RGB)
                    thumb_pil = Image.fromarray(thumb_rgb)
                    thumb_tk = ImageTk.PhotoImage(thumb_pil)
                    self._img_refs.append(thumb_tk)
                    lbl = tk.Label(parent, image=thumb_tk, bg=bg)
                    lbl.pack(expand=True)
                    lbl.bind("<Button-1>", lambda e, i=index: self._set_active(i))
                except: pass

        if self.is_collapsed:
            row.grid_columnconfigure(0, weight=1)
            c0 = make_cell(0)
            add_icon(c0, img_st.original)
        else:
            row.grid_columnconfigure(0, weight=1, uniform="cols") 
            row.grid_columnconfigure(1, weight=1, uniform="cols") 
            row.grid_columnconfigure(2, weight=1, uniform="cols")
            row.grid_columnconfigure(3, weight=0, minsize=30) 

            c0 = make_cell(0)
            f_icon = tk.Frame(c0, bg=bg)
            f_icon.pack(side="left", padx=2)
            add_icon(f_icon, img_st.original)
            
            lbl_name = tk.Label(c0, text=img_st.filename, bg=bg, fg="#222", anchor="w", font=("Segoe UI", 8))
            lbl_name.pack(side="left", fill="x", expand=True)
            lbl_name.bind("<Button-1>", lambda e, i=index: self._set_active(i))

            c1 = make_cell(1)
            add_icon(c1, img_st.preprocessed.img)

            c2 = make_cell(2)
            add_icon(c2, img_st.detected)

            c3 = make_cell(3)
            btn_del = tk.Label(c3, text="x", bg=bg, fg="#999", cursor="hand2")
            btn_del.pack(expand=True)
            btn_del.bind("<Button-1>", lambda e, i=index: self._delete(i))

        row.bind("<Enter>", lambda e: row.config(bg="#e8e8e8"))
        row.bind("<Leave>", lambda e: row.config(bg=bg))


    def _set_active(self, index):
        self.state.set_active(index)


    def _delete(self, index):
        self.state.remove_image(index)