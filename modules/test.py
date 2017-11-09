import xml.etree.ElementTree as ET


file = '/home/ichida/dev/ml/datasets/assin/assin-ptbr-dev.xml'

tree = ET.parse(file)
root = tree.getroot()

for child in root:
    print child.attrib
    print child.attrib['similarity']
    print child.find('t').text
    print child.find('h').text
