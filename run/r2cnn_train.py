import os.path
import sys
import warnings

from .base_train import SketchR2CNNTrain

if __name__ == '__main__':
    with SketchR2CNNTrain() as app:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            app.run()