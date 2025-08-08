import cv2

class RotationUtils:
    def __init__(self, labeler):
        self.labeler = labeler

    def apply_rotation_and_redraw(self):
        if self.labeler.original_image_cv2 is None:
            return
        if self.labeler.image_angle == 90:
            self.labeler.display_image_cv2 = cv2.rotate(self.labeler.original_image_cv2, cv2.ROTATE_90_CLOCKWISE)
        elif self.labeler.image_angle == 180:
            self.labeler.display_image_cv2 = cv2.rotate(self.labeler.original_image_cv2, cv2.ROTATE_180)
        elif self.labeler.image_angle == 270:
            self.labeler.display_image_cv2 = cv2.rotate(self.labeler.original_image_cv2, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            self.labeler.display_image_cv2 = self.labeler.original_image_cv2.copy()
        self.labeler.perform_resize()

    def rotate_image_left(self):
        if self.labeler.original_image_cv2 is None:
            return
        self.labeler.image_angle = (self.labeler.image_angle - 90 + 360) % 360
        self.labeler.image_angle_float = float(self.labeler.image_angle)
        self.labeler.rotation_dirty = True
        self.apply_rotation_and_redraw()

    def rotate_image_right(self):
        if self.labeler.original_image_cv2 is None:
            return
        self.labeler.image_angle = (self.labeler.image_angle + 90) % 360
        self.labeler.image_angle_float = float(self.labeler.image_angle)
        self.labeler.rotation_dirty = True
        self.apply_rotation_and_redraw()

    def apply_smooth_rotation(self):
        if self.labeler.original_image_cv2 is None:
            return

        h, w = self.labeler.original_image_cv2.shape[:2]
        angle = self.labeler.image_angle_float

        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, -angle, 1.0)

        abs_cos = abs(M[0, 0])
        abs_sin = abs(M[0, 1])

        new_w = int(h * abs_sin + w * abs_cos)
        new_h = int(h * abs_cos + w * abs_sin)

        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        rotated = cv2.warpAffine(
            self.labeler.original_image_cv2, M, (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)  # 흰색 배경
        )

        self.labeler.display_image_cv2 = rotated
        self.labeler.perform_resize()
