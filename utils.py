import xml.etree.ElementTree as ET

def parse_xml(path):
    root = ET.parse(path).getroot()
    w = int(root.find("size/width").text)
    h = int(root.find("size/height").text)
    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        difficult = int(obj.find("difficult").text)
        b = obj.find("bndbox")
        xmin = int(b.find("xmin").text)
        ymin = int(b.find("ymin").text)
        xmax = int(b.find("xmax").text)
        ymax = int(b.find("ymax").text)
        boxes.append((name, xmin, ymin, xmax, ymax, difficult))
    return w, h, boxes
