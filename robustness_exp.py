import warnings
import os
from copy import deepcopy

if __name__ == '__main__':
    stroke_removal_probs=[0.0, 0.1, 0.2, 0.3, 0.4, 
                          0.5, 0.6, 0.7, 0.8, 0.9]
    stroke_deformation_settings=[(0.0, 00.0), (0.1, 10.0), (0.2, 20.0), (0.3, 30.0), (0.4, 40.0), 
                                 (0.5, 50.0), (0.6, 60.0), (0.7, 70.0), (0.8, 80.0), (0.9, 90.0)]
    # # efficientnet
    # from run.cnn_runner import SketchCNNRunner
    # args = [
    #     '--model_fn', 'efficientnet_b0',
    #     '--data_seq_dir', '/home/lizenan/cs420/CS420-Proj/dataset/data/dataset_seq_r2cnn',
    #     # '--data_img_dir', '/home/lizenan/cs420/CS420-Proj/dataset/data/dataset_processed_28',
    #     '--batch_size', '48',
    #     '--seed', '[44]',
    # ]
    # with SketchCNNRunner(args=args, local_dir='/home/lizenan/cs420/CS420-Proj/res/efficientnet') as app:
    #     with warnings.catch_warnings():
    #         warnings.simplefilter('ignore')
    #         app.robustness_experiment(stroke_removal_probs=[0.0, 0.2, 0.4, 0.6, 0.8],
    #                                   stroke_deformation_settings=[(0.0, 0.0), (0.2, 20.0), (0.4, 40.0), (0.6, 60.0)])
    
    # gru
    from run.rnn_runner import SketchRNNRunner
    args = [
        '--model_fn', 'gru',
        '--data_seq_dir', '/home/lizenan/cs420/CS420-Proj/dataset/data/dataset_seq_r2cnn',
        # '--data_img_dir', '/home/lizenan/cs420/CS420-Proj/dataset/data/dataset_processed_28',
        '--batch_size', '48',
        '--seed', '[44]',
    ]
    with SketchRNNRunner(args=args, local_dir='/home/lizenan/cs420/CS420-Proj/res/gru') as app:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            app.robustness_experiment(stroke_removal_probs=stroke_removal_probs,
                                      stroke_deformation_settings=stroke_deformation_settings)
    
    # r2cnn
    from run.r2cnn_runner import SketchR2CNNRunner
    # backbone: sketchanet
    args = [
        '--model_fn', 'efficientnet_b0',
        '--data_seq_dir', '/home/lizenan/cs420/CS420-Proj/dataset/data/dataset_seq_r2cnn',
        # '--data_img_dir', '/home/lizenan/cs420/CS420-Proj/dataset/data/dataset_processed_28',
        '--batch_size', '64',
        '--seed', '[44]',
        '--intensity_channels', '8',
    ]
    with SketchR2CNNRunner(args=args, local_dir='/home/lizenan/cs420/CS420-Proj/res/r2cnn') as app:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            app.robustness_experiment(stroke_removal_probs=stroke_removal_probs,
                                      stroke_deformation_settings=stroke_deformation_settings)

    # trans2cnn_full
    from run.trans2cnn_runner import Trans2CNNRunner
    args = [
        '--model_fn', 'efficientnet_b0',
        '--dropout', '0.100',
        '--intensity_channels', '8',
        '--data_seq_dir', '/home/lizenan/cs420/CS420-Proj/dataset/data/dataset_seq_r2cnn',
        '--data_img_dir', '/home/lizenan/cs420/CS420-Proj/dataset/data/dataset_processed_28',
        '--batch_size', '128',
        '--seed', '[42]',
        '--do_reconstruction',
    ]
    print('trans2cnn_full')
    with Trans2CNNRunner(args=args, local_dir='/home/lizenan/cs420/CS420-Proj/res/trans2cnn_full') as app:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            app.robustness_experiment(stroke_removal_probs=stroke_removal_probs,
                                      stroke_deformation_settings=stroke_deformation_settings)