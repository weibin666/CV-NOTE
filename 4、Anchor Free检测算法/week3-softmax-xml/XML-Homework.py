'''
xml作业
作业内容：
1. 【必做 XML-Homework.py】
2.  了解目标检测xml的标注文件，提取目标框和图片长宽，用红色框把图像上的人脸框出来,并且写上类别名称 最后保存画框后的图像。

xml文件示例：
<annotations>
    <filename>test_00000335.jpg</filename>
    <size>
        <width>305</width>
        <height>458</height>
        <depth>3</depth>
    </size>
    <object>
        <name>face_mask</name>
        <bndbox>
            <xmin>138</xmin>
            <ymin>126</ymin>
            <xmax>207</xmax>
            <ymax>203</ymax>
        </bndbox>
    </object>
    <object>
        <name>face</name>
        <bndbox>
            <xmin>123</xmin>
            <ymin>316</ymin>
            <xmax>187</xmax>
            <ymax>379</ymax>
        </bndbox>
    </object>
</annotation>
'''


import cv2
from xml.dom.minidom import parse
import xml.dom.minidom
import os


def img_draw_bndbox(img_path, xml_path, out_path, img_draw_name):
    '''

    :param img_path: str, 原图像的地址
    :param xml_path: str, xml文件的地址
    :param out_path: 保存图像的地址
    :param img_draw_name:保存图像的名称
    :return: None
    '''
    image = cv2.imread(img_path)
    # 使用minidom解析器打开 XML 文档
    DOMTree = xml.dom.minidom.parse(xml_path)
    collection = DOMTree.documentElement
    # print(collection)

    '''
    获得左上角坐标(xmin, ymin)，右下角坐标(xmax, ymax)
    x.firstchild.data:获取元素第一个子节点的数据；
    x.childNodes[0]：:获取元素第一个子节点;
    x.childNodes[0].nodeValue.:也是获取元素第一个子节点值的意思
    '''
    # 示例：提取图片名称、宽、高
    filename = collection.getElementsByTagName('filename')[0].firstChild.data
    width = collection.getElementsByTagName('width')[0].firstChild.data
    height = collection.getElementsByTagName('height')[0].childNodes[0].nodeValue
    # print(filename)
    # print(width)
    # print(height)

    object_elements =
    for object_element in object_elements:
        # 获得类别名称
        object_name =
        # print('object name: ', object_name)
        # 获得 bndbox 集合
        bndbox_element =
        # print(bndbox_element)
        xmin =
        ymin =
        xmax =
        ymax =
        # 用红框把图像中的人脸框出,红色 (0, 0, 255)。
        '''
        import cv2
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
        
(xmin,ymin) -----------
           |          |
           |          |
           |          |
           ------------(xmax,ymax)
        '''
        image = cv2.rectangle(.......)
        font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
        # cv2.putText()参数依次是：图像，文字内容，坐标(左上角坐标) ，字体，大小，颜色，字体厚度
        # 用黄色字体在图像中写出类别名称，黄色 (0, 255, 255)
        image = cv2.putText(.......)
    cv2.imwrite(os.path.join(out_path, img_draw_name), image)
    return



if __name__ ==  '__main__':
    img_path = './test_00000335.jpg'
    xml_path = './test_00000335.xml'
    out_path = './'
    img_draw_name = 'test_00000335_draw.jpg'
    img_draw_bndbox(img_path, xml_path, out_path, img_draw_name)
