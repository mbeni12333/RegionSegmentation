import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import xml.etree.ElementTree as ET
import os

def findIsolatedLocalMaxima(greyScaleImage, threshold=1):
    squareDiameterLog3 = 2  # 27x27

    total = greyScaleImage
    for axis in range(2):
        d = 1
        for i in range(squareDiameterLog3):
            total = np.maximum(total, np.roll(total, d, axis))
            total = np.maximum(total, np.roll(total, -d, axis))
            d *= 2

    maxima = total == greyScaleImage
    h, w = greyScaleImage.shape

    result = []
    for j in range(h):
        for i in range(w):
            if maxima[j][i] and greyScaleImage[j, i] > threshold:
                result.append((i, j))
    return result

 

class IcyXml(object):
    def __init__(self, root_dir, ann_file):
        """
        Args:
        """
        self.root_dir = root_dir
        self.ann_file = ann_file


        xml = ET.parse(os.path.join(self.root_dir, self.ann_file))
        root = xml.getroot()
        self.name = root.find("name").text

        positionX = root.find("meta").find("positionX").text
        positionY = root.find("meta").find("positionY").text
        pixelSizeX = root.find("meta").find("pixelSizeX").text
        pixelSizeY = root.find("meta").find("pixelSizeY").text
        self.createMetadata(positionX, positionY, pixelSizeX, pixelSizeY)

        self.rois = []

        
    def createMetadata(self, positionX, positionY, pixelSizeX, pixelSizeY):
        """
        """
        self.meta = f"<meta>\
            <positionX>{positionX}</positionX>\
            <positionY>{positionY}</positionY>\
            <positionZ>0</positionZ>\
            <positionT>0</positionT>\
            <pixelSizeX>{pixelSizeX}</pixelSizeX>\
            <pixelSizeY>{pixelSizeY}</pixelSizeY>\
            <pixelSizeZ>1</pixelSizeZ>\
            <timeInterval>1</timeInterval>\
            <channelName0>ch 0</channelName0>\
            <channelName1>ch 1</channelName1>\
            <channelName2>ch 2</channelName2>\
        </meta>"


    def addRoiPoint(self, x, y, name, color):
        """
        """
        point = f"<roi>\
            <classname>plugins.kernel.roi.roi2d.ROI2DPoint</classname>\
            <name>{name}</name>\
            <selected>false</selected>\
            <readOnly>false</readOnly>\
            <color>{color}</color>\
            <stroke>2</stroke>\
            <opacity>0.3</opacity>\
            <showName>false</showName>\
            <z>-1</z>\
            <t>-1</t>\
            <c>-1</c>\
            <position>\
                <pos_x>{x}</pos_x>\
                <pos_y>{y}</pos_y>\
            </position>\
        </roi>"
        
        self.rois.append(point)

    def addPolygon(self, points, name, color):
        """
        """

        polygonPoints = [f"<point><pos_x>{point[0]}</pos_x><pos_y>{point[1]}</pos_y></point>" for point in points]

        polygon = "<roi>\
            <classname>plugins.kernel.roi.roi2d.ROI2DPolygon</classname>\
            <name>{}</name>\
            <selected>false</selected>\
            <readOnly>false</readOnly>\
            <color>{}</color>\
            <stroke>2</stroke>\
            <opacity>0.3</opacity>\
            <showName>false</showName>\
            <z>-1</z>\
            <t>-1</t>\
            <c>-1</c>\
            <position>\
                <pos_x>0</pos_x>\
                <pos_y>0</pos_y>\
            </position>\
            <points>\
                {}\
            </points>\
            </roi>".format(name, color[0]<<16 | color[1]<<8 | color[2]<<0, '\n'.join(polygonPoints))


        self.rois.append(polygon)

    def save(self):
        """
        """
        xml = f"<root>\
                    <name>{self.name}</name>\
                    <meta>{self.meta}</meta>\
                    <rois>{self.rois}</rois>\
                </root>"

        dataset_name = self.ann_file.split("/")[0]
        filename = self.ann_file.split("/")[1]
        destination = os.path.join(self.root_dir, "results", dataset_name)
        os.makedirs(destination, exist_ok=True)
        with open(os.path.join(destination, filename), "w") as f:
            f.write(xml)

        
