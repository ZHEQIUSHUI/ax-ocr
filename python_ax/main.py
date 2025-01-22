import os
import argparse
import cv2
import numpy as np
from text_det import TextDetector
from text_angle_cls import TextClassifier
from text_rec import TextRecognizer

from PIL import Image, ImageDraw, ImageFont

def putTextChinese(image, text, position, font_path, font_size, color=(255, 255, 255)):
    """
    在 OpenCV 图像上绘制中文文字。
    
    :param image: OpenCV 图像对象
    :param text: 要绘制的中文文字
    :param position: 绘制文字的起始位置 (x, y)
    :param font_path: 字体文件路径（例如 simhei.ttf）
    :param font_size: 字体大小
    :param color: 字体颜色 (B, G, R)，默认白色
    :return: 带中文文字的 OpenCV 图像
    """
    # 将 OpenCV 图像转换为 PIL 图像
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # 创建 PIL 绘图对象
    draw = ImageDraw.Draw(img_pil)
    
    # 加载字体
    font = ImageFont.truetype(font_path, font_size)
    
    # 绘制文字
    draw.text(position, text, font=font, fill=color)
    
    # 将 PIL 图像转换回 OpenCV 图像
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('imgpath', type=str, default='images/1.jpg', help="image path")
    args = parser.parse_args()

    detect_model = TextDetector()
    angle_model = TextClassifier()
    rec_model = TextRecognizer()

    srcimg = cv2.imread(args.imgpath)
    # srcimg = cv2.rotate(srcimg, 1)
    box_list, score_list = detect_model.detect(srcimg)
    text = ''
    if len(box_list) > 0:
        for point, score in zip(box_list, score_list):
            if score < 0.9:
                continue
            point = detect_model.order_points_clockwise(point)
            textimg = detect_model.get_rotate_crop_image(srcimg, point.astype(np.float32))
            # angle = angle_model.predict(textimg)
            # if angle=='180':
            #     print('rotate')
            #     textimg = cv2.rotate(textimg, 1)
            text = rec_model.predict_text(textimg)

            point = point.astype(int)
            cv2.polylines(srcimg, [point], True, (0, 0, 255), thickness=2)
            for i in range(4):
                cv2.circle(srcimg, tuple(point[i, :]), 3, (0, 255, 0), thickness=-1)
            print(text)
            srcimg = putTextChinese(srcimg, text, (point[0, 0], point[0, 1]+100), 'models/simfang.ttf', 100, color=(255, 0, 0))
    
    if not os.path.exists('result'):
        os.mkdir('result')
    filename = os.path.basename(args.imgpath)
    cv2.imwrite(f'result/result_{filename}', srcimg)