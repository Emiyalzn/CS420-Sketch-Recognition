import warnings
from run.cnn_runner import SketchCNNRunner
# from run.r2cnn_runner import SketchR2CNNRunner
from run.rnn_runner import SketchRNNRunner
from run.sketchmate_runner import SketchMateRunner
# from run.trans2cnn_runner import Trans2CNNRunner

if __name__ == '__main__':
    # with SketchCNNRunner(local_dir='res/efficientnet') as app:
    #     with warnings.catch_warnings():
    #         warnings.simplefilter('ignore')
    #         app.visualize_emb()

    # with SketchR2CNNRunner(local_dir='res/r2cnn') as app:
    #     with warnings.catch_warnings():
    #         warnings.simplefilter('ignore')
    #         app.visualize_emb()

    with SketchRNNRunner(local_dir="res/gru") as app:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            app.visualize_emb()

    # with SketchMateRunner() as app:
    #     with warnings.catch_warnings():
    #         warnings.simplefilter('ignore')
    #         app.visualize_emb()

    # with Trans2CNNRunner(local_dir='res/trans2cnn_full') as app:
    #     with warnings.catch_warnings():
    #         warnings.simplefilter('ignore')
    #         app.visualize_emb()