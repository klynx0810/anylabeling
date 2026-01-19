import logging
import os

import cv2
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .model import Model
from .types import AutoLabelingResult
from .engines.build_onnx_engine import OnnxBaseModel


class SwinUNETR(Model):
    """Semantic segmentation model using Swin-UNETR (ONNX)"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
            "classes",
        ]
        widgets = [
            "output_label",
            "output_select_combobox",
            "button_add_point",
            "button_remove_point",
            "button_add_rect",
            "button_clear",
            "button_finish_object",
            "button_run",
            "mask_fineness_slider",
            "mask_fineness_value_label",
        ]
        output_modes = {
            "polygon": QCoreApplication.translate("Model", "Polygon"),
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
            "rotation": QCoreApplication.translate("Model", "Rotation"),
        }
        default_output_mode = "polygon"

    def __init__(self, model_config, on_message) -> None:
        super().__init__(model_config, on_message)

        self.marks = []
        self.mask_fineness = float(self.config.get("poly_epsilon_ratio", 0.002))
        model_name = self.config["type"]
        model_abs_path = self.get_model_abs_path(self.config, "model_path")
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    f"Could not download or initialize {model_name} model.",
                )
            )

        # self.net = OnnxBaseModel(model_abs_path, __preferred_device__)
        print("[DBG] config provider =", self.config.get("provider"))
        import onnxruntime as ort
        print("[DBG] ORT available providers =", ort.get_available_providers())
        self.net = OnnxBaseModel(model_abs_path, self.config.get("provider", "cpu"))
        print("[DBG] ORT session providers:", self.net.ort_session.get_providers())
        self.classes = self.config["classes"]
        self.input_shape = self.net.get_input_shape()[-2:]  # (H, W)

    # def preprocess(self, input_image):
    #     input_h, input_w = self.input_shape
    #     image = cv2.resize(input_image, (input_w, input_h))
    #     image = np.transpose(image, (2, 0, 1))
    #     image = image.astype(np.float32) / 255.0
    #     image = (image - 0.5) / 0.5
    #     image = np.expand_dims(image, axis=0)
    #     return image

    def preprocess(self, input_image):
        input_h, input_w = self.input_shape  # (512,512)
        h, w = input_image.shape[:2]

        # LongestMaxSize
        scale = min(input_w / w, input_h / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        resized = cv2.resize(input_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # PadIfNeeded to (512,512)
        pad_w = input_w - new_w
        pad_h = input_h - new_h

        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        print("[DBG] padded dtype:", padded.dtype,
            "shape:", padded.shape,
            "min/max:", float(padded.min()), float(padded.max()),
            "mean:", float(padded.mean()),
            "new_size:", (new_h, new_w),
            "orig:", (h, w))

        # Normalize(mean=0,std=1,max_pixel_value=255) == scale to 0..1
        # x = padded.astype(np.float32) / 255.0
        x = padded.astype(np.float32)
        mx = float(x.max())
        if mx > 1.5:     # ảnh dạng 0..255 (hoặc hơn)
            x = x / 255.0
        # nếu đã 0..1 thì giữ nguyên

        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, axis=0)

        meta = {
            "scale": scale,
            "new_size": (new_h, new_w),
            "pad": (top, bottom, left, right),
            "orig_size": (h, w),
        }
        return x, meta

    # def postprocess(self, orig_image, outputs, resized_hw):
    #     # outputs: (1,C,512,512)
    #     new_h, new_w = resized_hw
    #     H, W = orig_image.shape[:2]

    #     pred = np.argmax(outputs, axis=1)[0]  # (512,512)
    #     uniq, cnt = np.unique(pred, return_counts=True)
    #     print("[DBG] pred unique:", list(zip(uniq.tolist(), cnt.tolist())))
    #     print("[DBG] outputs stats min/max/mean:",
    #         float(outputs.min()), float(outputs.max()), float(outputs.mean()))

    #     pred = pred[:new_h, :new_w]           # remove pad

    #     # resize back to original image size (undo longestmaxsize)
    #     pred = cv2.resize(pred.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)

    #     results = []
    #     for cls_id in range(len(self.classes)):
    #         if self.classes[cls_id] == "_background_":
    #             continue

    #         mask = (pred == cls_id).astype(np.uint8)
    #         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #         polys = []
    #         for cnt in contours:
    #             cnt = np.squeeze(cnt, axis=1)
    #             if cnt.ndim == 2 and cnt.shape[0] >= 3:
    #                 polys.append(cnt.tolist())

    #         if polys:
    #             results.append((self.classes[cls_id], polys))

    #     return results
    # def postprocess(self, orig_image, outputs, meta):
    #     # outputs: (1,C,512,512)
    #     top, bottom, left, right = meta["pad"]
    #     new_h, new_w = meta["new_size"]
    #     H, W = orig_image.shape[:2]

    #     pred = np.argmax(outputs, axis=1)[0]  # (512,512)

    #     uniq, cnt = np.unique(pred, return_counts=True)
    #     print("[DBG] pred unique:", list(zip(uniq.tolist(), cnt.tolist())))
    #     print("[DBG] outputs stats min/max/mean:",
    #         float(outputs.min()), float(outputs.max()), float(outputs.mean()))

    #     # ✅ remove pad đúng kiểu center
    #     pred = pred[top:top + new_h, left:left + new_w]

    #     # resize back to original crop size
    #     pred = cv2.resize(pred.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)

    #     results = []
    #     for cls_id in range(len(self.classes)):
    #         if self.classes[cls_id] == "_background_":
    #             continue

    #         mask = (pred == cls_id).astype(np.uint8)
    #         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #         polys = []
    #         for cnt in contours:
    #             cnt = np.squeeze(cnt, axis=1)
    #             if cnt.ndim == 2 and cnt.shape[0] >= 3:
    #                 polys.append(cnt.tolist())

    #         if polys:
    #             results.append((self.classes[cls_id], polys))

    #     return results

    def postprocess(self, orig_image, outputs, meta):
        """
        orig_image: ảnh crop/ROI trước preprocess (H,W,3)
        outputs: logits (1,C,512,512)
        meta: dict từ preprocess
            meta["new_size"]=(new_h,new_w)
            meta["pad"]=(top,bottom,left,right)
            meta["orig_size"]=(H,W)
        """
        H, W = orig_image.shape[:2]
        new_h, new_w = meta["new_size"]
        top, bottom, left, right = meta["pad"]

        # 1) argmax -> (512,512)
        pred = np.argmax(outputs, axis=1)[0].astype(np.uint8)

        # 2) bỏ pad (vì pad giữa)
        pred = pred[top: top + new_h, left: left + new_w]

        # 3) resize về size gốc của orig_image
        pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST)

        results = []

        # 4) tham số mượt từ slider
        eps_ratio = float(getattr(self, "mask_fineness", 0.002))  # vd 0.0016
        min_area = int(getattr(self, "min_contour_area", 200))    # tùy chỉnh

        for cls_id, cls_name in enumerate(self.classes):
            if cls_name == "_background_":
                continue

            mask = (pred == cls_id).astype(np.uint8)  # 0/1

            # Lấy contour
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )

            polys = []
            for cnt in contours:
                if cv2.contourArea(cnt) < min_area:
                    continue

                # approxPolyDP để giảm điểm
                peri = cv2.arcLength(cnt, True)
                eps = max(1e-6, eps_ratio) * peri
                approx = cv2.approxPolyDP(cnt, eps, True)

                approx = np.squeeze(approx, axis=1)  # (N,2)
                if approx.ndim == 2 and approx.shape[0] >= 3:
                    polys.append(approx.tolist())

            if polys:
                results.append((cls_name, polys))

        return results




    def predict_shapes(self, image, image_path=None):
        # IMPORTANT: always return AutoLabelingResult
        if image is None:
            return AutoLabelingResult([], replace=False)

        try:
            image = qt_img_to_rgb_cv_img(image, image_path)
            print("[DBG] cv_img dtype:", image.dtype,
            "shape:", image.shape,
            "min/max:", float(image.min()), float(image.max()),
            "mean:", float(image.mean()))

        except Exception as e:
            logging.warning(e)
            return AutoLabelingResult([], replace=False)

        img_h, img_w = image.shape[:2]

        # ROI nếu có marks, không có thì full image
        roi = None
        if getattr(self, "marks", None):
            roi = self._roi_from_marks(img_h, img_w, pad=10)

        if roi is None:
            x0, y0, x1, y1 = 0, 0, img_w, img_h
            crop = image
        else:
            x0, y0, x1, y1 = roi
            crop = image[y0:y1, x0:x1].copy()

        # blob, resized_hw = self.preprocess(crop)
        # outputs = self.net.get_ort_inference(blob)
        # results = self.postprocess(crop, outputs, resized_hw)
        
        blob, meta = self.preprocess(crop)
        outputs = self.net.get_ort_inference(blob)

        # meta["new_size"] = (new_h, new_w)
        # results = self.postprocess(crop, outputs, meta["new_size"])
        results = self.postprocess(crop, outputs, meta)


        shapes = []
        for label, polys in results:
            for pts in polys:
                if len(pts) < 3:
                    continue
                shape = Shape(flags={})
                pts_arr = np.asarray(pts, dtype=np.float32)
                # Chuẩn hoá về (N,2)
                if pts_arr.ndim == 3 and pts_arr.shape[1] == 1 and pts_arr.shape[2] == 2:
                    pts_arr = pts_arr[:, 0, :]
                elif pts_arr.ndim != 2 or pts_arr.shape[1] != 2:
                    # bỏ qua polygon lỗi format
                    continue

                for x, y in pts_arr:
                    shape.add_point(QtCore.QPointF(float(x + x0), float(y + y0)))
                shape.shape_type = "polygon"
                shape.closed = True
                shape.label = label
                shape.selected = False
                shapes.append(shape)

        # interactive mode nên replace=False để không xoá annotation cũ
        return AutoLabelingResult(shapes, replace=False)


    def unload(self):
        del self.net
    
    def set_auto_labeling_marks(self, marks):
        """Set auto labeling marks"""
        self.marks = marks or []
        print("DEBUG marks:", self.marks[:1])

    def _clamp(self, v, lo, hi):
        return max(lo, min(int(v), hi))

    def _roi_from_marks(self, image_h, image_w, pad=10):
        """
        Try to parse common mark formats:
        - dict with 'rect' or 'bbox'
        - dict with 'points'
        - tuple/list of 4 numbers
        Return (x0,y0,x1,y1) or None
        """
        if not self.marks:
            return None

        m = self.marks[0]

        # Case 1: tuple/list [x0,y0,x1,y1]
        if isinstance(m, (list, tuple)) and len(m) == 4 and all(isinstance(x, (int, float)) for x in m):
            x0, y0, x1, y1 = m

        # Case 2: dict
        elif isinstance(m, dict):
            # X-AnyLabeling rectangle mark format (bạn debug thấy đúng dạng này)
            if m.get("type") == "rectangle" and "data" in m and isinstance(m["data"], (list, tuple)) and len(m["data"]) == 4:
                x0, y0, x1, y1 = m["data"]

            elif "rect" in m and isinstance(m["rect"], (list, tuple)) and len(m["rect"]) == 4:
                x0, y0, x1, y1 = m["rect"]

            elif "bbox" in m and isinstance(m["bbox"], (list, tuple)) and len(m["bbox"]) == 4:
                x0, y0, x1, y1 = m["bbox"]

            elif "points" in m and m["points"]:
                pts = m["points"]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)

            else:
                return None
        else:
            return None

        # normalize ordering
        if x0 > x1: x0, x1 = x1, x0
        if y0 > y1: y0, y1 = y1, y0

        # padding + clamp
        x0 = self._clamp(x0 - pad, 0, image_w - 1)
        y0 = self._clamp(y0 - pad, 0, image_h - 1)
        x1 = self._clamp(x1 + pad, 0, image_w - 1)
        y1 = self._clamp(y1 + pad, 0, image_h - 1)

        # minimum size
        if (x1 - x0) < 5 or (y1 - y0) < 5:
            return None

        return x0, y0, x1, y1

    def set_mask_fineness(self, epsilon: float):
        """
        epsilon từ slider (ví dụ 0.0016)
        Dùng epsilon làm tỉ lệ simplify polygon.
        """
        try:
            self.mask_fineness = float(epsilon)
            print("[DBG] mask_fineness =", self.mask_fineness)
        except Exception as e:
            print("[DBG] set_mask_fineness error:", e)