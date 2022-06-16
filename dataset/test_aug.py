import os
import six
import numpy as np
from six.moves import xrange
import svgwrite

def get_bounds(data):
    """Return bounds of data."""
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0

    abs_x = 0
    abs_y = 0
    for i in range(len(data)):
        x = float(data[i, 0])
        y = float(data[i, 1])
        abs_x += x
        abs_y += y
        min_x = min(min_x, abs_x)
        min_y = min(min_y, abs_y)
        max_x = max(max_x, abs_x)
        max_y = max(max_y, abs_y)

    return (min_x, max_x, min_y, max_y)

def draw_strokes(data, svg_filename='sample.svg', width=48, margin=1.5, color='black'):
    """ convert sequence data to svg format """
    min_x, max_x, min_y, max_y = get_bounds(data)
    if max_x - min_x > max_y - min_y:
        norm = max_x - min_x
        border_y = (norm - (max_y - min_y)) * 0.5
        border_x = 0
    else:
        norm = max_y - min_y
        border_x = (norm - (max_x - min_x)) * 0.5
        border_y = 0

    # normalize data
    norm = max(norm, 10e-6)
    scale = (width - 2 * margin) / norm
    dx = 0 - min_x + border_x
    dy = 0 - min_y + border_y

    abs_x = (0 + dx) * scale + margin
    abs_y = (0 + dy) * scale + margin

    # start converting
    dwg = svgwrite.Drawing(svg_filename, size=(width, width))
    dwg.add(dwg.rect(insert=(0, 0), size=(width, width), fill='white'))
    lift_pen = 1
    p = "M%s,%s " % (abs_x, abs_y)
    command = "m"
    for i in xrange(len(data)):
        if (lift_pen == 1):
            command = "m"
        elif (command != "l"):
            command = "l"
        else:
            command = ""
        x = float(data[i, 0]) * scale
        y = float(data[i, 1]) * scale
        lift_pen = data[i, 2]
        p += command + str(x) + "," + str(y) + " "
    the_color = color  # "black"
    stroke_width = 1
    dwg.add(dwg.path(p).stroke(the_color, stroke_width).fill("none"))
    dwg.save()

data_seq_dir = '/home/lizenan/cs420/CS420-Proj/dataset/data/dataset_raw'
seq_path = os.path.join(data_seq_dir, 'dog.npz')
if six.PY3:
    seq_data = np.load(seq_path, encoding='latin1', allow_pickle=True)
else:
    seq_data = np.load(seq_path, allow_pickle=True)

draw_strokes(seq_data[0])

