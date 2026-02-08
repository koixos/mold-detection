from ..state import AppState, ImageState
from ..defs import PreprocessParams, DetectParams, PREPROCESS_METHODS, DETECT_METHODS, TH_MODES
from ..pipeline.processor import Processor

import tkinter as tk
from tkinter import ttk

DEF_PREPROCESS_PARAMS = PreprocessParams()
DEF_DETECT_PARAMS = DetectParams()

class RightSidebar(tk.Frame):
    def __init__(self, parent, state: AppState, processor: Processor, width=340):
        super().__init__(parent, width=width, bg="#f4f4f4")
        self.pack_propagate(False)

        self.state = state
        self.processor = processor
        
        self.state.add_listener(self._update_ui_state)
        
        self._updating_flag = False
        self.custom_var = tk.BooleanVar(value=False)
        self.custom_var.trace_add("write", self._on_custom_toggle)

        self._build_ui()
        self._update_ui_state()


    # ====================== UI ====================== 

    def _build_ui(self):
        # Create a canvas with scrollbar for the entire sidebar
        self.canvas = tk.Canvas(self, bg="#f4f4f4", highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview, width=12)
        self.scrollable_frame = tk.Frame(self.canvas, bg="#f4f4f4")
        
        # Configure canvas scrolling
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack canvas and scrollbar - scrollbar on top (overlay style)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.place(relx=1.0, rely=0, relheight=1.0, anchor="ne")
        
        # Bind mouse wheel scrolling
        self._bind_mousewheel()
        
        # Adjust canvas window width when canvas resizes
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        
        # Now build content inside scrollable_frame
        title = tk.Label(
            self.scrollable_frame,
            text="Settings",
            bg="#f4f4f4",
            fg="#222",
            font=("Segoe UI", 14, "bold")
        )
        title.pack(anchor="w", padx=16, pady=(16, 8))

        # Global Customize Toggle
        self.cbtn_custom = tk.Checkbutton(
            self.scrollable_frame, text="Customize Parameters", variable=self.custom_var,
            bg="#f4f4f4", command=self._update_controls_state
        )
        self.cbtn_custom.pack(anchor="w", padx=16, pady=(0, 8))

        # Main Controls Container
        self.frm_controls = tk.Frame(self.scrollable_frame, bg="#f4f4f4")
        # Initially hidden, will be packed in _update_controls_state
        
        self._preprocess_section()
        self._divider()
        self._detect_section()
        self._advanced_detect_section()
        
        self._main_actions()
        
        self.lbl_info = tk.Label(
            self.scrollable_frame, text="", bg="#f4f4f4", fg="#666", font=("Segoe UI", 9)
        )
        self.lbl_info.pack(side="bottom", pady=8)
    
    def _on_canvas_configure(self, event):
        # Make the canvas window match the canvas width
        self.canvas.itemconfig(self.canvas_window, width=event.width)
    
    def _bind_mousewheel(self):
        # Bind mouse wheel to canvas scrolling
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        # Bind to the main frame and all children
        self.bind_all("<MouseWheel>", _on_mousewheel)
    
    def _unbind_mousewheel(self):
        self.unbind_all("<MouseWheel>")

    def _main_actions(self):
        frame = tk.Frame(self.scrollable_frame, bg="#f4f4f4")
        frame.pack(side="bottom", fill="x", padx=16, pady=16)

        # preprocess/detect all btns
        top_bar = tk.Frame(frame, bg="#f4f4f4")
        top_bar.pack(fill="x", pady=(0, 4))
        
        btn_menu = tk.Button(
            top_bar, text="...", 
            relief="flat", bg="#e0e0e0", fg="#555",
            cursor="hand2", width=3
        )
        btn_menu.pack(side="right")
        
        self.menu_actions = tk.Menu(self, tearoff=0)
        self.menu_actions.add_command(label="Auto Detect All", command=self._auto_detect_all)
        
        def show_menu(e):
            self.menu_actions.post(e.x_root, e.y_root)
        btn_menu.bind("<Button-1>", show_menu)

        btn_font = ("Segoe UI", 12, "bold")
        
        self.actions_container = tk.Frame(frame, bg="#f4f4f4")
        self.actions_container.pack(fill="x")

        # Auto Detect Button
        self.btn_auto = tk.Button(
            self.actions_container,
            text="Auto Detect",
            command=self._auto_detect,
            bg="#673ab7", fg="white",
            relief="flat", font=btn_font,
            cursor="hand2", height=2
        )

        # Preprocess button moved to preprocess section

        # custom detect btn
        self.btn_detect = tk.Button(
            self.actions_container,
            text="Detect Mold",
            command=self._detect_active, 
            bg="#4caf50", fg="white",
            relief="flat", font=btn_font,
            cursor="hand2", height=2
        )


    def _divider(self):
        tk.Frame(self.frm_controls, height=1, bg="#cccccc").pack(fill="x", padx=16, pady=12)


    # ====================== PREPROCESS ====================== 

    def _preprocess_section(self):
        lbl = tk.Label(
            self.frm_controls,
            text="Preprocess Parameters Customization",
            bg="#f4f4f4",
            fg="#444",
            font=("Segoe UI", 11, "bold")
        )
        lbl.pack(anchor="w", padx=16, pady=(8, 4))

        self.gray_method_var = tk.StringVar(value=DEF_PREPROCESS_PARAMS.gray_method)
        self.use_clahe_var = tk.BooleanVar(value=DEF_PREPROCESS_PARAMS.use_clahe)
        self.clahe_clip_var = tk.DoubleVar(value=DEF_PREPROCESS_PARAMS.clahe_clip)
        self.clahe_grid_var = tk.IntVar(value=DEF_PREPROCESS_PARAMS.clahe_grid)
        
        self.cb_gray_method = self._dropdown(
            "Grayscale Method", self.gray_method_var,
            PREPROCESS_METHODS, parent=self.frm_controls
        )
        
        # CLAHE
        self.cbtn_clahe = tk.Checkbutton(
            self.frm_controls, text="Use CLAHE (Contrast Limited Adaptive Histogram Equalization)", variable=self.use_clahe_var,
            bg="#f4f4f4", command=self._on_preprocess_change
        )
        self.cbtn_clahe.pack(anchor="w", padx=16, pady=4)

        self.frm_clahe_container = tk.Frame(self.frm_controls, bg="#f4f4f4")
        self.frm_clahe_container.pack(fill="x")

        self.frm_clahe = tk.Frame(self.frm_clahe_container, bg="#f4f4f4")
        # Managed by visibility
        
        self.sc_clahe_clip = self._slider("Clip Limit", self.clahe_clip_var, 1.0, 10.0, step=0.1, parent=self.frm_clahe)
        self.sc_clahe_grid = self._slider("Grid Size", self.clahe_grid_var, 2, 32, parent=self.frm_clahe)
        
        # Live Update Triggers
        self.gray_method_var.trace_add("write", self._on_preprocess_change)
        self.use_clahe_var.trace_add("write", self._on_preprocess_change)
        self.clahe_clip_var.trace_add("write", self._on_preprocess_change)
        self.clahe_grid_var.trace_add("write", self._on_preprocess_change)

        self._row_buttons(("Restore Default", self._restore_preprocess), parent=self.frm_controls)
        
        # Preprocess Action Buttons with Menu
        preprocess_actions_frame = tk.Frame(self.frm_controls, bg="#f4f4f4")
        preprocess_actions_frame.pack(fill="x", padx=16, pady=(4, 8))
        
        # Menu button for Preprocess All
        btn_preprocess_menu = tk.Button(
            preprocess_actions_frame, text="...", 
            relief="flat", bg="#e0e0e0", fg="#555",
            cursor="hand2", width=3
        )
        btn_preprocess_menu.pack(side="right")
        
        self.menu_preprocess_actions = tk.Menu(self, tearoff=0)
        self.menu_preprocess_actions.add_command(label="Preprocess All", command=self._preprocess_all)
        
        def show_preprocess_menu(e):
            self.menu_preprocess_actions.post(e.x_root, e.y_root)
        btn_preprocess_menu.bind("<Button-1>", show_preprocess_menu)
        
        # Preprocess Button
        btn_font = ("Segoe UI", 11, "bold")
        self.btn_preprocess = tk.Button(
            preprocess_actions_frame,
            text="Preprocess",
            command=self._preprocess_active,
            bg="#2196f3", fg="white",
            relief="flat", font=btn_font,
            cursor="hand2", height=2
        )
        self.btn_preprocess.pack(side="left", fill="x", expand=True, padx=(0, 4))


    # ====================== DETECT ====================== 

    def _detect_section(self):
        lbl = tk.Label(
            self.frm_controls,
            text="Mold Detection Parameters Customization",
            bg="#f4f4f4",
            fg="#444",
            font=("Segoe UI", 11, "bold")
        )
        lbl.pack(anchor="w", padx=16, pady=(8, 4))

        self.method_var = tk.StringVar(value=DEF_DETECT_PARAMS.method)
        self.elemsize_var = tk.IntVar(value=DEF_DETECT_PARAMS.elemsize)
        self.open_iter_var = tk.IntVar(value=DEF_DETECT_PARAMS.open_iter)
        self.close_iter_var = tk.IntVar(value=DEF_DETECT_PARAMS.close_iter)

        self.cb_detect_method = self._dropdown(
            "Detection Method", self.method_var,
            DETECT_METHODS, parent=self.frm_controls
        )
        
        # Container for Dynamic Detect Options
        self.frm_detect_options = tk.Frame(self.frm_controls, bg="#f4f4f4")
        self.frm_detect_options.pack(fill="x")

        # -- Common --
        self.frm_common = tk.Frame(self.frm_detect_options, bg="#f4f4f4")
        
        self.sc_elemsize = self._slider("Morph. Size", self.elemsize_var, 1, 31, parent=self.frm_common)
        self.sc_opener = self._slider("Open Iter", self.open_iter_var, 0, 10, parent=self.frm_common)
        self.sc_closer = self._slider("Close Iter", self.close_iter_var, 0, 10, parent=self.frm_common)

        # Triggers
        all_vars = [
            self.method_var,
            self.elemsize_var, self.open_iter_var, self.close_iter_var,
        ]
        for v in all_vars:
            v.trace_add("write", self._on_detect_change)
        
        self.btn_hist = tk.Button(
            self.frm_controls,
            text="Plot Histogram",
            command=self._plot_histogram,
            bg="#e0e0e0", 
            relief="flat",
            cursor="hand2"
        )
        self.btn_hist.pack(anchor="w", padx=16, pady=4, fill="x")

        self._row_buttons(("Restore Default", self._restore_detect), parent=self.frm_controls)

    
    def _advanced_detect_section(self):
        # variance method
        self.th_mode_var = tk.StringVar(value=DEF_DETECT_PARAMS.th_mode)
        self.fixed_th_var = tk.IntVar(value=DEF_DETECT_PARAMS.fixed_th)
        self.zk_var = tk.DoubleVar(value=DEF_DETECT_PARAMS.z_k)
        self.percentile_var = tk.DoubleVar(value=DEF_DETECT_PARAMS.percentile)        
        self.min_area_var = tk.DoubleVar(value=DEF_DETECT_PARAMS.min_area)
        self.max_area_var = tk.DoubleVar(value=DEF_DETECT_PARAMS.max_area)
        # lbp optional
        self.use_lbp_var = tk.BooleanVar(value=DEF_DETECT_PARAMS.use_lbp)
        self.lbp_rad_var = tk.IntVar(value=DEF_DETECT_PARAMS.lbp_rad)
        self.lbp_points_var = tk.IntVar(value=DEF_DETECT_PARAMS.lbp_points)
        self.lbp_uniform_th_var = tk.DoubleVar(value=DEF_DETECT_PARAMS.lbp_uniform_th)

        # adaptive th method
        self.block_size_var = tk.IntVar(value=DEF_DETECT_PARAMS.block_size)
        self.c_var = tk.IntVar(value=DEF_DETECT_PARAMS.c)

        # edge density method
        self.edge_t1_var = tk.IntVar(value=DEF_DETECT_PARAMS.edge_t1)
        self.edge_t2_var = tk.IntVar(value=DEF_DETECT_PARAMS.edge_t2)
        self.edge_kernel_var = tk.IntVar(value=DEF_DETECT_PARAMS.edge_kernel)
        self.edge_density_th_var = tk.IntVar(value=DEF_DETECT_PARAMS.edge_density_th)

        # -- Variance Vars --
        self.frm_variance_opts = tk.Frame(self.frm_detect_options, bg="#f4f4f4")
        
        self.cb_th_mode = self._dropdown("Threshold Mode", self.th_mode_var, TH_MODES, parent=self.frm_variance_opts)
        
        # Create separate containers for each threshold mode parameter
        self.frm_fixed_th = tk.Frame(self.frm_variance_opts, bg="#f4f4f4")
        self.sc_fixed_th = self._slider("Fixed Threshold", self.fixed_th_var, 0, 255, parent=self.frm_fixed_th)
        
        self.frm_zk = tk.Frame(self.frm_variance_opts, bg="#f4f4f4")
        self.sc_zk = self._slider("Z-Score K", self.zk_var, 0.1, 10.0, step=0.1, parent=self.frm_zk)
        
        self.frm_percentile = tk.Frame(self.frm_variance_opts, bg="#f4f4f4")
        self.sc_percentile = self._slider("Percentile", self.percentile_var, 1.0, 99.9, step=0.5, parent=self.frm_percentile)
        
        self.sc_min_area = self._slider("Min Area Ratio", self.min_area_var, 0.0, 0.1, step=0.0001, parent=self.frm_variance_opts)
        self.sc_max_area = self._slider("Max Area Ratio", self.max_area_var, 0.0, 1.0, step=0.01, parent=self.frm_variance_opts)

        self.cbtn_lbp = tk.Checkbutton(
            self.frm_variance_opts, text="Use LBP (Local Binary Pattern) Filter", variable=self.use_lbp_var,
            bg="#f4f4f4", command=self._on_detect_change
        )
        self.cbtn_lbp.pack(anchor="w", padx=16, pady=4)
        
        self.frm_lbp_opts = tk.Frame(self.frm_variance_opts, bg="#f4f4f4")
        
        self.sc_lbp_rad = self._slider("LBP Radius", self.lbp_rad_var, 1, 10, parent=self.frm_lbp_opts)
        self.sc_lbp_points = self._slider("LBP Points", self.lbp_points_var, 4, 32, parent=self.frm_lbp_opts)
        self.sc_lbp_th = self._slider("LBP Uniformity Th", self.lbp_uniform_th_var, 0.0, 1.0, step=0.01, parent=self.frm_lbp_opts)

        # -- Adaptive Vars --
        self.frm_adaptive_opts = tk.Frame(self.frm_detect_options, bg="#f4f4f4")

        self.sc_block_size = self._slider("Block Size", self.block_size_var, 3, 99, parent=self.frm_adaptive_opts)
        self.sc_c = self._slider("C Constant", self.c_var, -20, 20, parent=self.frm_adaptive_opts)

        # -- Edge Vars --
        self.frm_edge_opts = tk.Frame(self.frm_detect_options, bg="#f4f4f4")

        self.sc_edge_t1 = self._slider("Canny T1", self.edge_t1_var, 0, 255, parent=self.frm_edge_opts)
        self.sc_edge_t2 = self._slider("Canny T2", self.edge_t2_var, 0, 255, parent=self.frm_edge_opts)
        self.sc_edge_k = self._slider("Filter Kernel", self.edge_kernel_var, 3, 31, parent=self.frm_edge_opts)
        self.sc_edge_dth = self._slider("Density Th", self.edge_density_th_var, 0, 255, parent=self.frm_edge_opts)

        self.lbl_advanced = tk.Label(
            self.frm_detect_options,
            text="Advanced Settings",
            bg="#f4f4f4",
            fg="#666",
            font=("Segoe UI", 10, "bold")
        )

        # Triggers
        all_vars = [
            self.th_mode_var, self.fixed_th_var, self.zk_var, self.percentile_var,
            self.min_area_var, self.max_area_var, #self.scales_var
            self.use_lbp_var, self.lbp_rad_var, self.lbp_points_var, self.lbp_uniform_th_var,
            self.block_size_var, self.c_var,
            self.edge_t1_var, self.edge_t2_var, self.edge_kernel_var, self.edge_density_th_var
        ]
        for v in all_vars:
            v.trace_add("write", self._on_detect_change)


    # ====================== WIDGET HELPERS ====================== 

    def _dropdown(self, label, var, values, parent=None):
        frame = tk.Frame(parent or self, bg="#f4f4f4")
        frame.pack(fill="x", padx=16, pady=4)

        tk.Label(frame, text=label, bg="#f4f4f4").pack(anchor="w")
        cb = ttk.Combobox(
            frame,
            textvariable=var,
            values=values,
            state="readonly"
        )
        cb.pack(fill="x")
        return cb
    

    def _slider(self, label, var, minv, maxv, step=1, parent=None):
        frame = tk.Frame(parent or self, bg="#f4f4f4")
        frame.pack(fill="x", padx=16, pady=4)

        tk.Label(frame, text=label, bg="#f4f4f4").pack(anchor="w")
        sc = tk.Scale(
            frame,
            from_=minv,
            to=maxv,
            resolution=step,
            orient="horizontal",
            variable=var,
            bg="#f4f4f4",
            highlightthickness=0
        )
        sc.pack(fill="x")
        return sc
    

    def _row_buttons(self, *buttons, parent=None):
        frame = tk.Frame(parent or self, bg="#f4f4f4")
        frame.pack(fill="x", padx=16, pady=6)

        for text, cmd in buttons:
            tk.Button(
                frame,
                text=text,
                command=cmd,
                relief="flat",
                bg="#e6e6e6",
                cursor="hand2"
            ).pack(side="left", expand=True, fill="x", padx=2)


    # ====================== LOGIC ====================== 

    def _active(self):
        return self.state.active()
    
    
    def _update_ui_state(self):
        self._update_button_states()
        
        # Sync Custom Toggle
        active_img = self._active()
        if active_img:
            self._updating_flag = True
            try:
                self.custom_var.set(active_img.custom)
            finally:
                self._updating_flag = False

        self._update_controls_state()
        self._update_info_label()


    def _update_button_states(self):
        has_active = (self.state.images and self.state.active_index != -1)
        
        self.btn_auto.pack_forget()
        self.btn_detect.pack_forget()

        if self.custom_var.get():
            self.btn_detect.pack(fill="x")
        else:
            self.btn_auto.pack(fill="x")

        if not has_active:
            self._set_btn_state(self.btn_auto, False, "#673ab7")
            self._set_btn_state(self.btn_preprocess, False, "#2196f3")
            self._set_btn_state(self.btn_detect, False, "#4caf50")
            return

        self._set_btn_state(self.btn_auto, True, "#673ab7")
        self._set_btn_state(self.btn_preprocess, True, "#2196f3")
        
        active_img = self._active()
        if active_img and active_img.preprocessed.img is not None:
            self._set_btn_state(self.btn_detect, True, "#4caf50")
        else:
            self._set_btn_state(self.btn_detect, False, "#4caf50")


    def _set_btn_state(self, btn, enabled, color):
        if enabled:
            btn.config(state="normal", bg=color, cursor="hand2")
        else:
            btn.config(state="disabled", bg="#cccccc", cursor="arrow")


    def _update_controls_state(self):
        if self._updating_flag: return
        is_custom = self.custom_var.get()
        
        # Main Container Visibility
        if is_custom:
            self.frm_controls.pack(fill="both", expand=True)
        else:
            self.frm_controls.pack_forget()
            return # Skip unpacking inside if hidden

        active_img = self._active()
        has_prep = (active_img is not None and active_img.preprocessed.img is not None)

        # Preprocess State
        # Since we just hide everything if not custom, we can assume 'normal' here? 
        # Actually logic kept 'is_custom' check for states. 
        # We can keep updating states just in case for robustness.
        
        # CLAHE Visibility
        if self.use_clahe_var.get():
            self.frm_clahe.pack(fill="x")
        else:
            self.frm_clahe.pack_forget()

        # Detect State
        det_enabled = has_prep 
        det_state = "normal" if det_enabled else "disabled"
        det_cb_state = "readonly" if det_enabled else "disabled"
        
        self.cb_detect_method.config(state=det_cb_state)
        # Verify btn_hist state
        self.btn_hist.config(state=det_state)

        # Reset packing of dynamic detect options
        for f in (self.frm_variance_opts, self.frm_adaptive_opts, self.frm_edge_opts, self.frm_common):
            f.pack_forget()
        
        self.lbl_advanced.pack_forget()
        self.frm_lbp_opts.pack_forget()
        self.frm_fixed_th.pack_forget()
        self.frm_zk.pack_forget()
        self.frm_percentile.pack_forget()
        
        # NOTE: Because sliders are inside frames that we pack/unpack, we don't need to unpack individual sliders 
        # unless they are conditional inside their frames (like fixed_th vs zk).
        # Variance opts has conditional children
        
        method = self.method_var.get()
        
        # Packing Order Logic inside frm_detect_options
        
        # 1. Common
        self.frm_common.pack(fill="x")
        
        # 2. Advanced Header
        self.lbl_advanced.pack(anchor="w", padx=16, pady=(12, 4))

        # 3. Method Specific
        if method == "variance":
            self.frm_variance_opts.pack(fill="x")
            
            # Logic for Variance internal packing - show only the relevant threshold parameter
            th_mode = self.th_mode_var.get()
            if th_mode == "fixed":
                self.frm_fixed_th.pack(fill="x")
            elif th_mode == "zscore":
                self.frm_zk.pack(fill="x")
            else:
                self.frm_percentile.pack(fill="x")

            if self.use_lbp_var.get():
                self.frm_lbp_opts.pack(fill="x")
        elif method == "adaptive":
            self.frm_adaptive_opts.pack(fill="x")
        elif method == "edge":
            self.frm_edge_opts.pack(fill="x")
        elif method == "saturation":
            self.frm_edge_opts.pack(fill="x")


        # Enable/Disable logic RECURSIVE
        for frm in (self.frm_variance_opts, self.frm_adaptive_opts, self.frm_edge_opts, self.frm_common):
            for child in frm.winfo_children():
                if isinstance(child, (tk.Scale, tk.Checkbutton, ttk.Combobox)):
                    child.configure(state=det_state)
                elif isinstance(child, tk.Entry):
                    child.configure(state=det_state)
                elif isinstance(child, tk.Frame):
                    for sub in child.winfo_children():
                        if isinstance(sub, (tk.Scale, tk.Entry, tk.Checkbutton)):
                            try: sub.configure(state=det_state)
                            except: pass

        self._update_all_vars_from_model()


    def _on_custom_toggle(self, *args):
        if self._updating_flag: return
        img = self._active()
        if img:
            img.custom = self.custom_var.get()
        self._update_ui_state()


    def _update_info_label(self):
        img = self._active()
        if not img:
            self.lbl_info.config(text="")
            return
        
        txt = ""
        if self.custom_var.get():
            txt = f"Custom | Gray: {self.gray_method_var.get()} | Det: {self.method_var.get()}"
        else:
            if img.preprocessed.img is not None:
                txt = getattr(img, 'auto_info', f"Auto Mode | Texture: {getattr(img.preprocessed, 'texture', '?')}")
            else:
                txt = "Auto Detect Mode"
        
        self.lbl_info.config(text=txt)


    def _on_preprocess_change(self, *args):
        img = self._active()
        if img is None: return
        self._write_preprocess_params(img)
        self._update_controls_state()

    
    def _on_detect_change(self, *args):
        img = self._active()
        if img is None: return
        self._write_detect_params(img)
        self._update_controls_state()


    def _update_all_vars_from_model(self):
        img = self._active()
        if not img or self._updating_flag: return

        self._updating_flag = True
        try:
            p = img.preprocess_params
            self.gray_method_var.set(p.gray_method)
            self.use_clahe_var.set(p.use_clahe)
            self.clahe_clip_var.set(p.clahe_clip)
            self.clahe_grid_var.set(p.clahe_grid)

            d = img.detect_params
            self.method_var.set(d.method)
            self.th_mode_var.set(d.th_mode)
            self.fixed_th_var.set(d.fixed_th)
            self.zk_var.set(d.z_k)
            self.percentile_var.set(d.percentile)
            self.use_lbp_var.set(d.use_lbp)
            self.lbp_rad_var.set(d.lbp_rad)
            self.lbp_points_var.set(d.lbp_points)
            self.lbp_uniform_th_var.set(d.lbp_uniform_th)
            self.min_area_var.set(d.min_area)
            self.max_area_var.set(d.max_area)
            self.elemsize_var.set(d.elemsize)
            self.open_iter_var.set(d.open_iter)
            self.close_iter_var.set(d.close_iter)
            self.block_size_var.set(d.block_size)
            self.c_var.set(d.c)
            self.edge_t1_var.set(d.edge_t1)
            self.edge_t2_var.set(d.edge_t2)
            self.edge_kernel_var.set(d.edge_kernel)
            self.edge_density_th_var.set(d.edge_density_th)
            # scales -> string
            '''
            sc = d.scales
            if sc:
                self.scales_var.set(",".join(map(str, sc)))
            else:
                self.scales_var.set("")
            '''
        except Exception as e:
            print(f"Update VARS error: {e}")
        finally:
            self._updating_flag = False


    def _write_preprocess_params(self, img: ImageState):
        if self._updating_flag: return
        img.preprocess_params.gray_method = self.gray_method_var.get()
        img.preprocess_params.use_clahe = self.use_clahe_var.get()
        img.preprocess_params.clahe_clip = self.clahe_clip_var.get()
        img.preprocess_params.clahe_grid = self.clahe_grid_var.get()
    

    def _write_detect_params(self, img: ImageState):
        if self._updating_flag: return
        d = img.detect_params
        d.method = self.method_var.get()
        d.th_mode = self.th_mode_var.get()
        d.fixed_th = self.fixed_th_var.get()
        d.z_k = self.zk_var.get()
        d.percentile = self.percentile_var.get()
        d.use_lbp = self.use_lbp_var.get()
        d.lbp_rad = self.lbp_rad_var.get()
        d.lbp_points = self.lbp_points_var.get()
        d.lbp_uniform_th = self.lbp_uniform_th_var.get()
        d.min_area = self.min_area_var.get()
        d.max_area = self.max_area_var.get()
        d.elemsize = self.elemsize_var.get()
        d.open_iter = self.open_iter_var.get()
        d.close_iter = self.close_iter_var.get()
        d.block_size = self.block_size_var.get()
        d.c = self.c_var.get()
        d.edge_t1 = self.edge_t1_var.get()
        d.edge_t2 = self.edge_t2_var.get()
        d.edge_kernel = self.edge_kernel_var.get()
        d.edge_density_th = self.edge_density_th_var.get()
        
        # parse scales
        '''
        sc_str = self.scales_var.get().strip()
        if not sc_str:
            d.scales = None
        else:
            try:
                valid = []
                for p in sc_str.split(','):
                    val = int(p.strip())
                    if val > 0: valid.append(val)
                d.scales = valid if valid else None
            except:
                d.scales = None
        '''

    def _restore_preprocess(self):
        self._updating_flag = True
        self.gray_method_var.set(DEF_PREPROCESS_PARAMS.gray_method)
        self.use_clahe_var.set(DEF_PREPROCESS_PARAMS.use_clahe)
        self.clahe_clip_var.set(DEF_PREPROCESS_PARAMS.clahe_clip)
        self.clahe_grid_var.set(DEF_PREPROCESS_PARAMS.clahe_grid)
        self._updating_flag = False
        self._on_preprocess_change()
    

    def _restore_detect(self):
        self._updating_flag = True
        self.method_var.set(DEF_DETECT_PARAMS.method)
        self.th_mode_var.set(DEF_DETECT_PARAMS.th_mode)
        self.fixed_th_var.set(DEF_DETECT_PARAMS.fixed_th)
        self.zk_var.set(DEF_DETECT_PARAMS.z_k)
        self.percentile_var.set(DEF_DETECT_PARAMS.percentile)
        self.use_lbp_var.set(DEF_DETECT_PARAMS.use_lbp)
        self.lbp_rad_var.set(DEF_DETECT_PARAMS.lbp_rad)
        self.lbp_points_var.set(DEF_DETECT_PARAMS.lbp_points)
        self.lbp_uniform_th_var.set(DEF_DETECT_PARAMS.lbp_uniform_th)
        #self.scales_var.set("") # Default is None logic
        self.min_area_var.set(DEF_DETECT_PARAMS.min_area)
        self.max_area_var.set(DEF_DETECT_PARAMS.max_area)
        self.elemsize_var.set(DEF_DETECT_PARAMS.elemsize)
        self.open_iter_var.set(DEF_DETECT_PARAMS.open_iter)
        self.close_iter_var.set(DEF_DETECT_PARAMS.close_iter)
        self.block_size_var.set(DEF_DETECT_PARAMS.block_size)
        self.c_var.set(DEF_DETECT_PARAMS.c)
        self.edge_t1_var.set(DEF_DETECT_PARAMS.edge_t1)
        self.edge_t2_var.set(DEF_DETECT_PARAMS.edge_t2)
        self.edge_kernel_var.set(DEF_DETECT_PARAMS.edge_kernel)
        self.edge_density_th_var.set(DEF_DETECT_PARAMS.edge_density_th)
        self._updating_flag = False
        self._on_detect_change()


    def _preprocess_active(self):
        img = self._active()
        if img is None: return
        self._run_preprocess(img)
        self.state._notify() # Refresh UI


    def _preprocess_all(self):
        for img in self.state.images:
            self._run_preprocess(img)
        self.state._notify()


    def _run_preprocess(self, img: ImageState):
        self._write_preprocess_params(img)
        if self.processor:
            try:
                res, txt = self.processor.preprocess(img)
                img.preprocessed.img = res
                img.preprocessed.texture = txt
                img.detected = None # Invalidate detection if re-processed
            except Exception as e:
                print(f"Preprocess Error: {e}")


    def _detect_active(self):
        img = self._active()
        if img is None: return
        self._run_detect(img)
        self.state._notify()


    def _detect_all(self):
        for img in self.state.images:
            self._run_detect(img)
        self.state._notify()


    def _auto_detect(self):
        self._preprocess_active()
        self._detect_active()

    def _auto_detect_all(self):
        """Run auto detect (preprocess + detect) on all images"""
        for img in self.state.images:
            self._run_preprocess(img)
            self._run_detect(img)
        self.state._notify()


    def _plot_histogram(self):
        img = self._active()
        if img and self.processor:
            self.processor.show_variance_histogram(img)

    
    def _run_detect(self, img: ImageState):
        if img.preprocessed.img is None: return 
        self._write_detect_params(img)
        if self.processor:
            try:        
                img.detected = self.processor.detect(img)
            except Exception as e:
                print(f"Detect Error: {e}")
                # Print stack trace
                import traceback
                traceback.print_exc()
