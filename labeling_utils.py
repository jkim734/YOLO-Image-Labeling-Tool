import tkinter as tk

class LabelingUtils:
    def __init__(self, labeler):
        self.labeler = labeler
        self.labeler.bbox_rects = []
        self.labeler.bbox_texts = []  # Track class text items

    def draw_all_bboxes(self):
        for rect in self.labeler.bbox_rects:
            self.labeler.canvas.delete(rect)
        for text in self.labeler.bbox_texts:
            self.labeler.canvas.delete(text)
        self.labeler.bbox_rects.clear()
        self.labeler.bbox_texts.clear()
        if not self.labeler.current_image:
            return
        img_w, img_h = self.labeler.current_image.size
        off_x = (self.labeler.canvas_width - img_w) // 2
        off_y = (self.labeler.canvas_height - img_h) // 2
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
        for cid, xc, yc, w, h in self.labeler.bboxes:
            x1, y1 = off_x + (xc - w / 2) * img_w, off_y + (yc - h / 2) * img_h
            x2, y2 = off_x + (xc + w / 2) * img_w, off_y + (yc + h / 2) * img_h
            color = colors[int(cid) % len(colors)]
            rect = self.labeler.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2)
            self.labeler.bbox_rects.append(rect)
            if int(cid) < len(self.labeler.classes):
                text = self.labeler.canvas.create_text(x1, y1 - 10, text=self.labeler.classes[int(cid)], fill=color, anchor=tk.W)
                self.labeler.bbox_texts.append(text)

    def start_bbox(self, event):
        if self.labeler.mode != 'labeling' or not self.labeler.current_image:
            return
        self.labeler.start_x, self.labeler.start_y = event.x, event.y
        if self.labeler.current_bbox:
            self.labeler.canvas.delete(self.labeler.current_bbox)
        self.labeler.current_bbox = None

    def draw_bbox(self, event):
        if self.labeler.mode != 'labeling' or self.labeler.start_x is None:
            return
        if self.labeler.current_bbox:
            self.labeler.canvas.delete(self.labeler.current_bbox)
        self.labeler.current_bbox = self.labeler.canvas.create_rectangle(
            self.labeler.start_x, self.labeler.start_y, event.x, event.y, outline='red', width=2, dash=(5, 5))

    def end_bbox(self, event):
        if self.labeler.mode != 'labeling' or self.labeler.start_x is None:
            return
        if abs(event.x - self.labeler.start_x) < 5 or abs(event.y - self.labeler.start_y) < 5:
            if self.labeler.current_bbox:
                self.labeler.canvas.delete(self.labeler.current_bbox)
            self.labeler.start_x = self.labeler.current_bbox = None
            return
        img_w, img_h = self.labeler.current_image.size
        off_x, off_y = (self.labeler.canvas_width - img_w) // 2, (self.labeler.canvas_height - img_h) // 2
        x1, y1 = max(0, min(self.labeler.start_x, event.x) - off_x), max(0, min(self.labeler.start_y, event.y) - off_y)
        x2, y2 = min(img_w, max(self.labeler.start_x, event.x) - off_x), min(img_h, max(self.labeler.start_y, event.y) - off_y)
        if x1 >= x2 or y1 >= y2:
            return
        xc, yc = (x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h
        w, h = (x2 - x1) / img_w, (y2 - y1) / img_h
        self.labeler.bboxes.append((self.labeler.current_class, xc, yc, w, h))
        if self.labeler.current_bbox:
            self.labeler.canvas.delete(self.labeler.current_bbox)
        self.labeler.start_x = self.labeler.current_bbox = None
        self.draw_all_bboxes()
        self.labeler.update_label_list()

    def save_labels_to_txt(self):
        # Save current bboxes to the corresponding txt file
        if not hasattr(self.labeler, 'image_path') or not self.labeler.image_path:
            return
        label_path = self.labeler.image_path.rsplit('.', 1)[0] + '.txt'
        if not self.labeler.bboxes:
            import os
            if os.path.exists(label_path):
                os.remove(label_path)
            return
        with open(label_path, 'w') as f:
            for bbox in self.labeler.bboxes:
                f.write(' '.join(map(str, bbox)) + '\n')

    def delete_bbox(self, event):
        if self.labeler.mode != 'labeling' or not self.labeler.bboxes:
            return
        if not self.labeler.current_image:
            return
        img_w, img_h = self.labeler.current_image.size
        off_x, off_y = (self.labeler.canvas_width - img_w) // 2, (self.labeler.canvas_height - img_h) // 2
        click_x, click_y = event.x - off_x, event.y - off_y
        min_dist = float('inf')
        del_index = None
        for i, (cid, xc, yc, w, h) in enumerate(self.labeler.bboxes):
            center_x = xc * img_w
            center_y = yc * img_h
            dist = (center_x - click_x) ** 2 + (center_y - click_y) ** 2
            if dist < min_dist:
                min_dist = dist
                del_index = i
        if del_index is not None:
            del self.labeler.bboxes[del_index]
            self.draw_all_bboxes()
            self.labeler.update_label_list()
            self.save_labels_to_txt()

    def delete_selected_label(self, event):
        if self.labeler.mode != 'labeling' or not self.labeler.label_listbox.curselection():
            return
        sel = self.labeler.label_listbox.curselection()
        if sel:
            idx = sel[0]
            if 0 <= idx < len(self.labeler.bboxes):
                del self.labeler.bboxes[idx]
                self.draw_all_bboxes()
                self.labeler.update_label_list()
                self.save_labels_to_txt()
