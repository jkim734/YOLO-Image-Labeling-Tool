import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import os
from PIL import Image, ImageTk
import numpy as np
from rotation_utils import RotationUtils
from labeling_utils import LabelingUtils

class YOLOLabeler:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Image Labeler")
        self.root.geometry("1200x800")
        self.crosshair_lines = []

        # 변수 초기화
        self.original_image_cv2 = None
        self.display_image_cv2 = None # 회전을 포함하여 화면에 표시될 이미지
        self.current_image = None
        self.image_path = None
        self.image_list = []
        self.current_index = 0
        self.scale_factor = 1.0
        self.canvas_width = 800
        self.canvas_height = 600

        # 바운딩 박스 관련
        self.start_x = None
        self.start_y = None
        self.current_bbox = None
        self.bboxes = []
        self.bbox_rects = []

        # 클래스 정보
        self.classes = []
        self.current_class = 0

        # 모드 관리
        self.mode = 'labeling' # 'labeling' vs 'rotation'
        self.image_angle = 0
        self.rotation_dirty = False # 회전 후 저장되지 않은 변경사항

        # 마우스 드래그로 섬세한 회전 관련 변수 추가
        self.image_angle_float = 0.0  # 실수형 누적 회전 각도 (마우스 드래그용)
        self.drag_start_x = None
        self.drag_start_y = None
        self.start_angle = 0.0

        # 자동 저장 모드 설정
        self.auto_save_enabled = True

        # 리사이즈 작업 핸들러
        self.resize_job = None

        self.setup_ui()
        self.load_classes()
        self.toggle_mode() # 초기 UI 상태 설정
        self.thumbnail_size = 100
        self.thumbnails = []
        self.thumb_labels = []

        self.rotation_utils = RotationUtils(self)
        self.labeling_utils = LabelingUtils(self)

    def setup_ui(self):
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 왼쪽 썸네일 영역
        thumb_frame = tk.Frame(main_frame, width=120)
        thumb_frame.pack(side=tk.LEFT, fill=tk.Y)
        thumb_frame.pack_propagate(False)

        # 썸네일용 캔버스 + 스크롤바 조합
        self.thumb_canvas = tk.Canvas(thumb_frame, width=120)
        self.thumb_scrollbar = tk.Scrollbar(thumb_frame, orient=tk.VERTICAL, command=self.thumb_canvas.yview)
        self.thumb_scrollable_frame = tk.Frame(self.thumb_canvas)

        self.thumb_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.thumb_canvas.configure(
                scrollregion=self.thumb_canvas.bbox("all")
            )
        )

        self.thumb_canvas.create_window((0, 0), window=self.thumb_scrollable_frame, anchor="nw")
        self.thumb_canvas.configure(yscrollcommand=self.thumb_scrollbar.set)

        self.thumb_canvas.pack(side=tk.LEFT, fill=tk.Y, expand=True)
        self.thumb_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 마우스 휠, 트랙패드, 방향키 스크롤 이벤트 바인딩 (윈도우/맥/리눅스 호환)
        def _on_mousewheel(event):
            if os.name == 'nt':
                self.thumb_canvas.yview_scroll(-1 * int(event.delta / 120), "units")
            else:
                self.thumb_canvas.yview_scroll(-1 * int(event.delta), "units")
        self.thumb_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self.thumb_canvas.bind_all("<Button-4>", lambda e: self.thumb_canvas.yview_scroll(-1, "units"))
        self.thumb_canvas.bind_all("<Button-5>", lambda e: self.thumb_canvas.yview_scroll(1, "units"))
        # 방향키(Up/Down)로 썸네일 스크롤
        self.thumb_canvas.bind_all("<Up>", lambda e: self.thumb_canvas.yview_scroll(-3, "units"))
        self.thumb_canvas.bind_all("<Down>", lambda e: self.thumb_canvas.yview_scroll(3, "units"))
        # 트랙패드 두 손가락 스와이프(Shift+MouseWheel)도 지원
        self.thumb_canvas.bind_all("<Shift-MouseWheel>", _on_mousewheel)

        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(left_frame, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_button_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_button_release)
        self.canvas.bind("<Button-3>", lambda event: self.labeling_utils.delete_bbox(event))
        self.canvas.bind("<Button-2>", lambda event: self.labeling_utils.delete_bbox(event))

        nav_frame = tk.Frame(left_frame)
        nav_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

        tk.Button(nav_frame, text="이전", command=self.prev_image).pack(side=tk.LEFT, padx=5)
        tk.Button(nav_frame, text="다음", command=self.next_image).pack(side=tk.LEFT, padx=5)

        self.image_info_label = tk.Label(nav_frame, text="이미지 없음")
        self.image_info_label.pack(side=tk.LEFT, padx=20)

        right_frame = tk.Frame(main_frame, width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        right_frame.pack_propagate(False)

        file_frame = tk.LabelFrame(right_frame, text="파일 관리")
        file_frame.pack(fill=tk.X, pady=5)
        tk.Button(file_frame, text="이미지 폴더 선택", command=self.load_images).pack(fill=tk.X, padx=5, pady=2)
        tk.Button(file_frame, text="이미지 파일 선택", command=self.load_single_image).pack(fill=tk.X, padx=5, pady=2)
        tk.Button(file_frame, text="클래스 설정", command=self.setup_classes).pack(fill=tk.X, padx=5, pady=2)

        mode_frame = tk.LabelFrame(right_frame, text="모드 설정")
        mode_frame.pack(fill=tk.X, pady=5)
        self.mode_var = tk.StringVar(value=self.mode)
        tk.Radiobutton(mode_frame, text="라벨링", variable=self.mode_var, value="labeling", command=self.toggle_mode).pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(mode_frame, text="회전", variable=self.mode_var, value="rotation", command=self.toggle_mode).pack(side=tk.LEFT, padx=10)

        rotation_frame = tk.LabelFrame(right_frame, text="이미지 회전")
        rotation_frame.pack(fill=tk.X, pady=5)
        self.rotate_left_btn = tk.Button(rotation_frame, text="왼쪽 회전 (90도)", command=self.rotate_image_left)
        self.rotate_left_btn.pack(fill=tk.X, padx=5, pady=2)
        self.rotate_right_btn = tk.Button(rotation_frame, text="오른쪽 회전 (90도)", command=self.rotate_image_right)
        self.rotate_right_btn.pack(fill=tk.X, padx=5, pady=2)

        class_frame = tk.LabelFrame(right_frame, text="클래스 선택")
        class_frame.pack(fill=tk.X, pady=5)
        self.calculate_class_frame_height()
        canvas_frame = tk.Canvas(class_frame, height=self.class_frame_height)
        scrollbar_class = tk.Scrollbar(class_frame, orient="vertical", command=canvas_frame.yview)
        self.scrollable_frame = tk.Frame(canvas_frame)
        self.scrollable_frame.bind("<Configure>", lambda e: canvas_frame.configure(scrollregion=canvas_frame.bbox("all")))
        canvas_frame.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas_frame.configure(yscrollcommand=scrollbar_class.set)
        canvas_frame.pack(side="left", fill="both", expand=True, padx=(5, 0), pady=5)
        scrollbar_class.pack(side="right", fill="y", pady=5)
        self.canvas_frame = canvas_frame
        self.class_var = tk.IntVar(value=0)
        self.class_radiobuttons = []
        self.update_class_radiobuttons()

        label_frame = tk.LabelFrame(right_frame, text="현재 라벨")
        label_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        list_frame = tk.Frame(label_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.label_listbox = tk.Listbox(list_frame)
        scrollbar = tk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.label_listbox.yview)
        self.label_listbox.config(yscrollcommand=scrollbar.set)
        self.label_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.label_listbox.bind("<Double-Button-1>", self.delete_selected_label)

        save_frame = tk.LabelFrame(right_frame, text="저장")
        save_frame.pack(fill=tk.X, pady=5)
        self.auto_save_var = tk.BooleanVar(value=self.auto_save_enabled)
        auto_save_checkbox = tk.Checkbutton(save_frame, text="자동 저장 모드", variable=self.auto_save_var, command=self.toggle_auto_save)
        auto_save_checkbox.pack(fill=tk.X, padx=5, pady=2)
        tk.Button(save_frame, text="저장", command=self.save_changes).pack(fill=tk.X, padx=5, pady=2)

        self.thumbnails = []  # PIL 이미지들
        self.thumb_labels = []  # Label 위젯들

        self.root.bind('<Left>', lambda event: self.prev_image())
        self.root.bind('<Right>', lambda event: self.next_image())
        # 라벨링 모드 단축키
        for key in ['w', 'W']:
            self.root.bind(f'<{key}>', lambda event: self.set_mode_labeling())
        # 회전 모드 단축키
        for key in ['r', 'R']:
            self.root.bind(f'<{key}>', lambda event: self.set_mode_rotation())

    def on_canvas_resize(self, event):
        if self.resize_job: self.root.after_cancel(self.resize_job)
        self.resize_job = self.root.after(200, self.perform_resize)

    def perform_resize(self):
        if self.display_image_cv2 is None: return
        h, w = self.display_image_cv2.shape[:2]
        self.canvas_width, self.canvas_height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if self.canvas_width < 2 or self.canvas_height < 2: return
        self.scale_factor = min(self.canvas_width / w, self.canvas_height / h)
        new_w, new_h = int(w * self.scale_factor), int(h * self.scale_factor)
        if new_w < 1 or new_h < 1: return
        img_resized = cv2.resize(self.display_image_cv2, (new_w, new_h))
        self.current_image = Image.fromarray(img_resized)
        self.photo = ImageTk.PhotoImage(self.current_image)
        self.canvas.delete("all")
        self.canvas.create_image(self.canvas_width // 2, self.canvas_height // 2, image=self.photo)
        self.draw_all_bboxes()
        if self.mode == 'rotation':
            self.draw_crosshair_lines()

    def load_classes(self):
        if os.path.exists("classes.txt"):
            with open("classes.txt", 'r', encoding='utf-8') as f:
                self.classes = [line.strip() for line in f.readlines() if line.strip()]
        else:
            self.classes = ["person", "car", "bike", "dog", "cat"]
        self.update_class_radiobuttons()

    def setup_classes(self):
        class_window = tk.Toplevel(self.root)
        class_window.title("클래스 설정")
        text_widget = tk.Text(class_window, height=15)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        text_widget.insert(tk.END, '\n'.join(self.classes))
        def save_classes():
            self.classes = [cls.strip() for cls in text_widget.get(1.0, tk.END).strip().split('\n') if cls.strip()]
            with open("classes.txt", 'w', encoding='utf-8') as f: f.write('\n'.join(self.classes))
            self.update_class_radiobuttons()
            class_window.destroy()
        tk.Button(class_window, text="저장", command=save_classes).pack(pady=5)

    def update_class_radiobuttons(self):
        for rb in self.class_radiobuttons: rb.destroy()
        self.class_radiobuttons.clear()
        self.calculate_class_frame_height()
        if hasattr(self, 'canvas_frame'): self.canvas_frame.config(height=self.class_frame_height)
        self.class_var.set(0)
        for i, class_name in enumerate(self.classes):
            rb = tk.Radiobutton(self.scrollable_frame, text=class_name, variable=self.class_var, value=i, indicatoron=0, anchor="w", command=self.on_class_selected)
            rb.pack(fill=tk.X, padx=5, pady=2)
            self.class_radiobuttons.append(rb)
        if self.classes: self.current_class = 0
        self.toggle_mode()

    def on_class_selected(self, event=None): self.current_class = self.class_var.get()

    def load_single_image(self):
        if not self.check_unsaved_rotation(): return
        file_path = filedialog.askopenfilename(filetypes=(('Image Files', '*.jpg *.jpeg *.png *.bmp'), ('All Files', '*.*')))
        if file_path:
            self.image_list = [file_path]
            self.current_index = 0
            self.load_current_image()

    def load_images(self):
        if not self.check_unsaved_rotation(): return
        folder_path = filedialog.askdirectory()
        if folder_path:
            ext = ('.jpg', '.jpeg', '.png', '.bmp')
            self.image_list = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(ext)])
            if self.image_list:
                self.current_index = 0
                self.load_current_image()
                self.load_thumbnails()  # 추가: 썸네일 생성
            else:
                messagebox.showwarning("경고", "이미지 파일을 찾을 수 없습니다.")

    def load_current_image(self):
        if not self.image_list: return
        self.image_path = self.image_list[self.current_index]
        img = cv2.imread(self.image_path)
        if img is None:
            messagebox.showerror("오류", f"이미지를 읽을 수 없습니다: {self.image_path}")
            return
        self.original_image_cv2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.display_image_cv2 = self.original_image_cv2.copy()
        self.image_angle = 0
        self.image_angle_float = 0.0
        self.rotation_dirty = False
        self.bboxes, self.bbox_rects = [], []
        label_path = self.image_path.rsplit('.', 1)[0] + '.txt'
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        self.bboxes.append(tuple(map(float, parts)))
        self.perform_resize()
        self.update_image_info()
        self.update_label_list()

    def load_thumbnails(self):
        # 기존 썸네일 삭제
        for lbl in self.thumb_labels:
            lbl.destroy()
        self.thumb_labels.clear()
        self.thumbnails.clear()

        for i, img_path in enumerate(self.image_list):
            img = Image.open(img_path)
            img.thumbnail((self.thumbnail_size, self.thumbnail_size))
            thumb = ImageTk.PhotoImage(img)
            self.thumbnails.append(thumb)  # 참조 유지 중요!

            lbl = tk.Label(self.thumb_scrollable_frame, image=thumb, cursor="hand2", borderwidth=2, relief="groove")
            lbl.pack(padx=5, pady=5)

            def click_handler(event, idx=i):
                if self.auto_save_enabled:
                    if self.rotation_dirty:
                        self.save_rotation()
                    self.save_current_labels()
                if not self.check_unsaved_rotation():
                    return
                self.current_index = idx
                self.load_current_image()

            lbl.bind("<Button-1>", click_handler)
            self.thumb_labels.append(lbl)

    def apply_rotation_and_redraw(self):
        self.rotation_utils.apply_rotation_and_redraw()

    def rotate_image_left(self):
        self.rotation_utils.rotate_image_left()

    def rotate_image_right(self):
        self.rotation_utils.rotate_image_right()

    def apply_smooth_rotation(self):
        self.rotation_utils.apply_smooth_rotation()

    def draw_all_bboxes(self):
        self.labeling_utils.draw_all_bboxes()

    def start_bbox(self, event):
        self.labeling_utils.start_bbox(event)

    def draw_bbox(self, event):
        self.labeling_utils.draw_bbox(event)

    def end_bbox(self, event):
        self.labeling_utils.end_bbox(event)

    def delete_bbox(self, event):
        self.labeling_utils.delete_bbox(event)

    def delete_selected_label(self, event):
        self.labeling_utils.delete_selected_label(event)

    def update_label_list(self):
        self.label_listbox.delete(0, tk.END)
        for i, (cid, xc, yc, w, h) in enumerate(self.bboxes):
            c_name = self.classes[int(cid)] if int(cid) < len(self.classes) else f"Class{int(cid)}"
            self.label_listbox.insert(tk.END, f"{i+1}. {c_name} ({w:.3f}x{h:.3f})")

    def update_image_info(self):
        text = f"{self.current_index + 1}/{len(self.image_list)} - {os.path.basename(self.image_path)}" if self.image_list else "이미지 없음"
        self.image_info_label.config(text=text)

    def prev_image(self):
        if not self.check_unsaved_rotation(): return
        if self.image_list and self.current_index > 0:
            if self.auto_save_enabled:
                if self.rotation_dirty:
                    self.save_rotation()
                self.save_current_labels()
            self.current_index -= 1
            self.load_current_image()

    def next_image(self):
        if self.image_list and self.current_index < len(self.image_list) - 1:
            if self.auto_save_enabled:
                if self.rotation_dirty:
                    self.save_rotation()
                # Save labels and remove txt file if no labels
                if not self.bboxes:
                    label_path = self.image_path.rsplit('.', 1)[0] + '.txt'
                    if os.path.exists(label_path):
                        os.remove(label_path)
                else:
                    self.save_current_labels()
            self.current_index += 1
            self.load_current_image()

    def save_current_labels(self):
        if not self.image_path or not self.bboxes:
            return
        label_path = self.image_path.rsplit('.', 1)[0] + '.txt'
        with open(label_path, 'w') as f:
            for cid, xc, yc, w, h in self.bboxes:
                f.write(f"{int(cid)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    def save_changes(self):
        if self.mode == 'rotation':
            if self.rotation_dirty:
                self.save_rotation()
            else:
                messagebox.showinfo("정보", "회���된 내용이 없습니다.")
        elif self.mode == 'labeling':
            if not self.image_path:
                messagebox.showwarning("경고", "저장할 이미지가 없습니다.")
                return
            self.save_current_labels()

    def calculate_class_frame_height(self):
        self.class_frame_height = max(80, min(300, len(self.classes) * 30 + 20))

    def toggle_auto_save(self): self.auto_save_enabled = self.auto_save_var.get()

    def toggle_mode(self):
        self.mode = self.mode_var.get()
        is_rotation_mode = self.mode == 'rotation'

        self.rotate_left_btn.config(state=tk.NORMAL if is_rotation_mode else tk.DISABLED)
        self.rotate_right_btn.config(state=tk.NORMAL if is_rotation_mode else tk.DISABLED)

        for rb in self.class_radiobuttons:
            rb.config(state=tk.DISABLED if is_rotation_mode else tk.NORMAL)

        self.canvas.config(cursor="arrow" if is_rotation_mode else "crosshair")

        if is_rotation_mode:
            self.draw_crosshair_lines()
        else:
            for line_id in self.crosshair_lines:
                self.canvas.delete(line_id)
            self.crosshair_lines.clear()


    def on_canvas_button_press(self, event):
        if self.mode == 'rotation':
            self.drag_start_x = event.x
            self.drag_start_y = event.y
            self.start_angle = self.image_angle_float
        elif self.mode == 'labeling':
            self.start_bbox(event)

    def on_canvas_mouse_drag(self, event):
        if self.mode == 'rotation' and self.drag_start_x is not None:
            delta_x = event.x - self.drag_start_x
            sensitivity = 0.5  # 1픽셀당 0.5도 회전, 필요시 조절 가능
            delta_angle = delta_x * sensitivity
            new_angle = (self.start_angle + delta_angle) % 360
            self.image_angle_float = new_angle
            self.apply_smooth_rotation()
        elif self.mode == 'labeling':
            self.draw_bbox(event)

    def on_canvas_button_release(self, event):
        if self.mode == 'rotation' and self.drag_start_x is not None:
            self.image_angle_float = self.image_angle_float % 360
            self.image_angle = int(round(self.image_angle_float))
            self.rotation_dirty = True
            self.drag_start_x = None
            self.drag_start_y = None
            self.start_angle = 0.0
        elif self.mode == 'labeling':
            self.end_bbox(event)


    def save_rotation(self):
        img_to_save = cv2.cvtColor(self.display_image_cv2, cv2.COLOR_RGB2BGR)
        try:
            cv2.imwrite(self.image_path, img_to_save)
            self.original_image_cv2 = self.display_image_cv2.copy()
            self.rotation_dirty = False
            self.image_angle = 0
            self.image_angle_float = 0.0
        except Exception as e:
            messagebox.showerror("오류", f"이미지 저장에 실패했습니다: {e}")

    def check_unsaved_rotation(self):
        if self.rotation_dirty:
            response = messagebox.askyesnocancel("저장 확인", "회전된 이미지가 저장되지 않았습니다. 저장하시겠습니까?")
            if response is True:
                self.save_rotation()
                return True
            elif response is False:
                return True
            else:
                return False
        return True

    def draw_crosshair_lines(self):
        if self.display_image_cv2 is None:
            return
        # 기존 보조선 삭제
        for line_id in self.crosshair_lines:
            self.canvas.delete(line_id)
        self.crosshair_lines = []

        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        center_x, center_y = w // 2, h // 2

        # 가로/세로 파선 보조선 추가
        h_line = self.canvas.create_line(0, center_y, w, center_y, fill='blue', dash=(4, 4))
        v_line = self.canvas.create_line(center_x, 0, center_x, h, fill='blue', dash=(4, 4))

        self.crosshair_lines.extend([h_line, v_line])

    def set_mode_labeling(self):
        self.mode_var.set('labeling')
        self.toggle_mode()

    def set_mode_rotation(self):
        self.mode_var.set('rotation')
        self.toggle_mode()

def main():
    root = tk.Tk()
    app = YOLOLabeler(root)
    root.mainloop()

if __name__ == "__main__":
    main()
