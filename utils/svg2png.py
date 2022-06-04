""" convert svg file to png file """
import cairosvg
import os

def exportsvg(fromDir, targetDir, exportType):
    num = 0
    for a, f, c in os.walk(fromDir):
        for fileName in c:
            path = os.path.join(a, fileName)
            if os.path.isfile(path) and fileName[-3:] == "svg":
                num += 1
                fileHandle = open(path)
                svg = fileHandle.read()
                fileHandle.close()
                exportPath = os.path.join(targetDir, fileName[:-3] + exportType)
                exportFileHandle = open(exportPath, 'w')
                if exportType == "png":
                    try:
                        cairosvg.svg2png(bytestring=svg, write_to=exportPath)
                    except:
                        print("error in convert svg file : %s to png." % path)
                exportFileHandle.close()

def main(svgDir, exportDir):
    if not os.path.exists(exportDir):
        os.mkdir(exportDir)
    exportsvg(svgDir, exportDir, 'png')



