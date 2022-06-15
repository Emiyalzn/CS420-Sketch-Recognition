import warnings
# from run.cnn_train import SketchCNNTrain
from run.r2cnn_train import SketchR2CNNTrain

if __name__ == '__main__':
    # * This set of args corresponds to original SketchR2CNNTrain paper.
    args = [
        '--dropout', '0.5',
        '--intensity_channels', '8',

        '--model_fn', 'sketchanet',
        '--data_seq_dir', '/home/lizenan/cs420/CS420-Proj/dataset/data/dataset_raw/',
        # '--data_img_dir', '~/cs420/dataset/data/dataset_processed_28',

        '--categories', str(['dog', 'bear', 'kangaroo', 'whale', 'crocodile',
                             'rhinoceros', 'penguin', 'camel', 'flamingo', 'giraffe',
                             'pig', 'cat', 'cow', 'panda', 'lion',
                             'tiger', 'raccoon', 'monkey', 'hedgehog', 'zebra',
                             'horse', 'owl', 'elephant', 'squirrel', 'sheep']),

        '--batch_size', str(48),
        '--num_epoch', str(20),
        '--seed', str(42),

        '--disable_augmentation'
    ]

    # with SketchCNNTrain(args) as app:
    #     with warnings.catch_warnings():
    #         warnings.simplefilter('ignore')
    #         app.run()

    with SketchR2CNNTrain(args) as app:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            app.run()