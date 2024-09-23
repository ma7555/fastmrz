import numpy as np
import cv2
import pytesseract
from datetime import datetime
import os
import cv2
import numpy as np
from binarize import binarize


class FastMRZ:

    def __init__(self, tesseract_path=""):
        self.tesseract_path = tesseract_path
        self.net = cv2.dnn.readNetFromONNX(
            os.path.join(os.path.dirname(__file__), "model/mrz_det.onnx")
        )
        self.INPUT_WIDTH = 320
        self.INPUT_HEIGHT = 320
        self.NMS_THRESHOLD = 0.7
        self.CONFIDENCE_THRESHOLD = 0.5
        self.CLASESS_YOLO = ["mrz"]

    def resize_with_padding(self, image, target_size, padding_color=(0, 0, 0)):
        """
        Resize an image while keeping the aspect ratio intact and adding padding to reach the target size.

        Args:
        - image: Input image to be resized.
        - target_size: Tuple of (width, height) for the target dimensions.
        - padding_color: Tuple (B, G, R) to set padding color. Default is black (0, 0, 0).

        Returns:
        - Padded image with the aspect ratio maintained and scaling factors + padding.
        """
        h, w = image.shape[:2]
        target_w, target_h = target_size
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized_image = cv2.resize(image, (new_w, new_h))
        padded_image = np.full((target_h, target_w, 3), padding_color, dtype=np.uint8)
        top = (target_h - new_h) // 2
        left = (target_w - new_w) // 2
        padded_image[top : top + new_h, left : left + new_w] = resized_image
        return padded_image, scale, left, top

    def _process_image(self, image_path, return_pad=False):
        orig_image = (
            cv2.imread(image_path, cv2.IMREAD_COLOR)
            if isinstance(image_path, str)
            else image_path
        )

        processed_image, scale, left_pad, top_pad = self.resize_with_padding(
            orig_image, (self.INPUT_WIDTH, self.INPUT_HEIGHT)
        )

        return processed_image, orig_image, scale, left_pad, top_pad

    def _get_roi(self, output_data, image, scale=1, left_pad=0, top_pad=0):
        if self.tesseract_path != "":
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_path

        class_ids, confs, boxes = [], [], []

        rows = output_data[0].shape[0]
        for i in range(rows):
            row = output_data[0][i]
            conf = row[4]

            classes_score = row[4:]
            minVal, maxVal, min_idx, max_idx = cv2.minMaxLoc(classes_score)
            class_id = max_idx[1]
            if classes_score[class_id] > self.CONFIDENCE_THRESHOLD:
                confs.append(maxVal)
                label = self.CLASESS_YOLO[int(class_id)]
                class_ids.append(label)
                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = int(((x - 0.5 * w) - left_pad) / scale)
                top = int(((y - 0.5 * h) - top_pad) / scale)
                width = int(w / scale)
                height = int(h / scale)
                box = np.array([left, top, width, height])
                boxes.append(box)

        r_class_ids, r_confs, r_boxes = [], [], []

        indexes = cv2.dnn.NMSBoxes(
            boxes, confs, self.CONFIDENCE_THRESHOLD, self.NMS_THRESHOLD
        )

        for i in indexes:
            r_class_ids.append(class_ids[i])
            r_confs.append(confs[i])
            r_boxes.append(boxes[i])

        if not r_confs:
            return ""
        i = np.argmax(r_confs)
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        roi_arr = image[y : y + h, x : x + w].copy()
        if roi_arr.shape[0] == 0 or roi_arr.shape[1] == 0:
            return ""
        roi_arr = cv2.cvtColor(roi_arr, cv2.COLOR_BGR2GRAY)
        binary_roi = binarize(roi_arr)
        return pytesseract.image_to_string(
            roi_arr, lang="mrz", config="--psm 6 -c tosp_min_sane_kn_sp=3"
        )

    def _cleanse_roi(self, raw_text):
        input_list = raw_text.replace(" ", "").split("\n")
        selection_length = next(
            (
                len(item)
                for item in input_list
                if "<" in item and len(item) in {30, 36, 44}
            ),
            None,
        )
        if selection_length is None:
            return ""
        new_list = [item for item in input_list if len(item) >= selection_length]
        return "\n".join(new_list)

    def _get_final_check_digit(self, input_string, input_type):
        if input_type == "TD3":
            return self._get_check_digit(
                input_string[:10] + input_string[13:20] + input_string[21:43]
            )
        elif input_type == "TD2":
            return self._get_check_digit(
                input_string[:10] + input_string[13:20] + input_string[21:35]
            )
        else:
            return self._get_check_digit(
                input_string[0][5:]
                + input_string[1][:7]
                + input_string[1][8:15]
                + input_string[1][18:29]
            )

    def _get_check_digit(self, input_string):
        weights_pattern = [7, 3, 1]

        total = 0
        for i, char in enumerate(input_string):
            if char.isdigit():
                value = int(char)
            elif char.isalpha():
                value = ord(char.upper()) - ord("A") + 10
            else:
                value = 0
            total += value * weights_pattern[i % len(weights_pattern)]

        check_digit = total % 10

        return str(check_digit)

    def _format_date(self, input_date):
        formatted_date = str(datetime.strptime(input_date, "%y%m%d").date())
        return formatted_date

    def _is_valid(self, image):
        if isinstance(image, str):
            return bool(os.path.isfile(image))
        elif isinstance(image, np.ndarray):
            return image.shape[-1] == 3

    def _get_raw_mrz(self, image):
        (processed_image, orig_image, scale, left_pad, top_pad) = self._process_image(
            image
        )
        blob = cv2.dnn.blobFromImage(
            processed_image,
            1 / 255.0,
            (self.INPUT_WIDTH, self.INPUT_HEIGHT),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        output_data = self.net.forward()
        output_data = output_data.transpose((0, 2, 1))
        raw_roi = self._get_roi(output_data, orig_image, scale, left_pad, top_pad)
        return self._cleanse_roi(raw_roi)

    def get_mrz(self, image, raw=False):
        if not self._is_valid(image):
            return {"status": "FAILURE", "message": "Invalid input image"}
        mrz_text = self._get_raw_mrz(image)
        return mrz_text if raw else self._parse_mrz(mrz_text)

    def _get_date_of_birth(self, date_of_birth_str, date_of_expiry_str):
        birth_year = int(date_of_birth_str[:4])
        expiry_year = int(date_of_expiry_str[:4])

        if expiry_year > birth_year:
            return date_of_birth_str
        adjusted_year = birth_year - 100
        return f"{adjusted_year}-{date_of_birth_str[5:]}"

    def _parse_mrz(self, mrz_text):
        if not mrz_text:
            return {"status": "FAILURE", "message": "No MRZ detected"}
        mrz_lines = mrz_text.strip().split("\n")
        if len(mrz_lines) not in [2, 3]:
            return {"status": "FAILURE", "message": "Invalid MRZ format"}

        mrz_code_dict = {}
        if len(mrz_lines) == 2:
            mrz_code_dict["mrz_type"] = "TD2" if len(mrz_lines[0]) == 36 else "TD3"

            # Line 1
            mrz_code_dict["document_type"] = mrz_lines[0][:2].strip("<")
            mrz_code_dict["country_code"] = mrz_lines[0][2:5]
            if not mrz_code_dict["country_code"].isalpha():
                return {"status": "FAILURE", "message": "Invalid MRZ format"}
            names = mrz_lines[0][5:].split("<<")
            mrz_code_dict["surname"] = names[0].replace("<", " ")
            mrz_code_dict["given_name"] = names[1].replace("<", " ")

            # Line 2
            mrz_code_dict["document_number"] = mrz_lines[1][:9].replace("<", "")
            if (
                self._get_check_digit(mrz_code_dict["document_number"])
                != mrz_lines[1][9]
            ):
                return {
                    "status": "FAILURE",
                    "message": "document number checksum is not matching",
                }
            mrz_code_dict["nationality"] = mrz_lines[1][10:13]
            if not mrz_code_dict["nationality"].isalpha():
                return {"status": "FAILURE", "message": "Invalid MRZ format"}
            mrz_code_dict["date_of_birth"] = mrz_lines[1][13:19]
            if (
                self._get_check_digit(mrz_code_dict["date_of_birth"])
                != mrz_lines[1][19]
            ):
                return {
                    "status": "FAILURE",
                    "message": "date of birth checksum is not matching",
                }
            mrz_code_dict["date_of_birth"] = self._format_date(
                mrz_code_dict["date_of_birth"]
            )
            mrz_code_dict["sex"] = mrz_lines[1][20]
            mrz_code_dict["date_of_expiry"] = mrz_lines[1][21:27]
            if (
                self._get_check_digit(mrz_code_dict["date_of_expiry"])
                != mrz_lines[1][27]
            ):
                return {
                    "status": "FAILURE",
                    "message": "date of expiry checksum is not matching",
                }
            mrz_code_dict["date_of_expiry"] = self._format_date(
                mrz_code_dict["date_of_expiry"]
            )
            mrz_code_dict["date_of_birth"] = self._get_date_of_birth(
                mrz_code_dict["date_of_birth"], mrz_code_dict["date_of_expiry"]
            )
            if mrz_code_dict["mrz_type"] == "TD3":
                mrz_code_dict["optional_data"] = mrz_lines[1][28:35].strip("<")

            mrz_code_dict["optional_data"] = (
                mrz_lines[1][28:35].strip("<")
                if mrz_code_dict["mrz_type"] == "TD2"
                else mrz_lines[1][28:42].strip("<")
            )
            if mrz_lines[1][-1] != self._get_final_check_digit(
                mrz_lines[1], mrz_code_dict["mrz_type"]
            ):
                return {
                    "status": "FAILURE",
                    "message": "final checksum is not matching",
                }

        else:
            mrz_code_dict["mrz_type"] = "TD1"

            # Line 1
            mrz_code_dict["document_type"] = mrz_lines[0][:2].strip("<")
            mrz_code_dict["country_code"] = mrz_lines[0][2:5]
            if not mrz_code_dict["country_code"].isalpha():
                return {"status": "FAILURE", "message": "Invalid MRZ format"}
            mrz_code_dict["document_number"] = mrz_lines[0][5:14]
            if (
                self._get_check_digit(mrz_code_dict["document_number"])
                != mrz_lines[0][14]
            ):
                return {
                    "status": "FAILURE",
                    "message": "document number checksum is not matching",
                }
            mrz_code_dict["optional_data_1"] = mrz_lines[0][15:].strip("<")

            # Line 2
            mrz_code_dict["date_of_birth"] = mrz_lines[1][:6]
            if self._get_check_digit(mrz_code_dict["date_of_birth"]) != mrz_lines[1][6]:
                return {
                    "status": "FAILURE",
                    "message": "date of birth checksum is not matching",
                }
            mrz_code_dict["date_of_birth"] = self._format_date(
                mrz_code_dict["date_of_birth"]
            )
            mrz_code_dict["sex"] = mrz_lines[1][7]
            mrz_code_dict["date_of_expiry"] = mrz_lines[1][8:14]
            if (
                self._get_check_digit(mrz_code_dict["date_of_expiry"])
                != mrz_lines[1][14]
            ):
                return {
                    "status": "FAILURE",
                    "message": "date of expiry checksum is not matching",
                }
            mrz_code_dict["date_of_expiry"] = self._format_date(
                mrz_code_dict["date_of_expiry"]
            )
            mrz_code_dict["date_of_birth"] = self._get_date_of_birth(
                mrz_code_dict["date_of_birth"], mrz_code_dict["date_of_expiry"]
            )
            mrz_code_dict["nationality"] = mrz_lines[1][15:18]
            if not mrz_code_dict["nationality"].isalpha():
                return {"status": "FAILURE", "message": "Invalid MRZ format"}
            mrz_code_dict["optional_data_2"] = mrz_lines[0][18:29].strip("<")
            if mrz_lines[1][-1] != self._get_final_check_digit(
                mrz_lines, mrz_code_dict["mrz_type"]
            ):
                return {
                    "status": "FAILURE",
                    "message": "final checksum is not matching",
                }

            # Line 3
            names = mrz_lines[2].split("<<")
            mrz_code_dict["surname"] = names[0].replace("<", " ")
            mrz_code_dict["given_name"] = names[1].replace("<", " ")

        # Final status
        mrz_code_dict["status"] = "SUCCESS"
        return mrz_code_dict
