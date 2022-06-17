import warnings
# from run.cnn_runner import SketchCNNRunner
from run.trans2cnn_runner import Trans2CNNRunner
# from run.r2cnn_runner import SketchR2CNNRunner
# from run.rnn_runner import SketchRNNRunner
# from run.sketchmate_runner import SketchMateRunner

if __name__ == '__main__':
    # with SketchCNNRunner() as app:
    #     with warnings.catch_warnings():
    #         warnings.simplefilter('ignore')
    #         app.train()

    # with SketchR2CNNRunner() as app:
    #     with warnings.catch_warnings():
    #         warnings.simplefilter('ignore')
    #         app.train()

    # with SketchRNNRunner() as app:
    #     with warnings.catch_warnings():
    #         warnings.simplefilter('ignore')
    #         app.train()

    # with SketchMateRunner() as app:
    #     with warnings.catch_warnings():
    #         warnings.simplefilter('ignore')
    #         app.train()

    with Trans2CNNRunner() as app:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            app.train()