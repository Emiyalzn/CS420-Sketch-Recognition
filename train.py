import warnings
from run.cnn_train import SketchCNNTrain
# from run.r2cnn_train import SketchR2CNNTrain
from run.rnn_train import SketchRNNTrain
from run.sketchmate_train import SketchMateTrain

if __name__ == '__main__':
    # with SketchCNNTrain() as app:
    #     with warnings.catch_warnings():
    #         warnings.simplefilter('ignore')
    #         app.run()

    # with SketchR2CNNTrain() as app:
    #     with warnings.catch_warnings():
    #         warnings.simplefilter('ignore')
    #         app.run()

    # with SketchRNNTrain() as app:
    #     with warnings.catch_warnings():
    #         warnings.simplefilter('ignore')
    #         app.run()

    with SketchMateTrain() as app:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            app.run()