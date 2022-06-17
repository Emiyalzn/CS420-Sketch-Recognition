import warnings
from run.cnn_runner import SketchCNNRunner
# from run.r2cnn_runner import SketchR2CNNRunner
# from run.rnn_runner import SketchRNNRunner
# from run.sketchmate_runner import SketchMateRunner

if __name__ == '__main__':
    with SketchCNNRunner() as app:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            app.evaluate()

    # with SketchR2CNNRunner() as app:
    #     with warnings.catch_warnings():
    #         warnings.simplefilter('ignore')
    #         app.evaluate()

    # with SketchRNNRunner() as app:
    #     with warnings.catch_warnings():
    #         warnings.simplefilter('ignore')
    #         app.evaluate()

    # with SketchMateRunner() as app:
    #     with warnings.catch_warnings():
    #         warnings.simplefilter('ignore')
    #         app.evaluate()
    #
    # with Trans2CNNRunner() as app:
    #     with warnings.catch_warnings():
    #         warnings.simplefilter('ignore')
    #         app.train()