import random
import numpy as np
import torch
from texttable import Texttable
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def args_print(args, logger):
    _dict = args
    table = Texttable()
    table.add_row(["Parameter", "Value"])
    for k in _dict:
        table.add_row([k, _dict[k]])
    logger.info(table.draw())

def tsne_vis(feats, labels, filename, num_class):
    fig = plt.figure(figsize=(4.5, 4.5))

    tsne = TSNE(perplexity=50,
                verbose=False,
                n_components=2,
                init='pca',
                early_exaggeration=12,
                learning_rate=1000,
                n_iter=3000)

    feature_2d = tsne.fit_transform(feats)
    x_min, x_max = np.min(feature_2d, 0), np.max(feature_2d, 0)
    feature_2d = (feature_2d - x_min) / (x_max - x_min)

    df = pd.DataFrame()
    df['pca-one'] = feature_2d[:, 0]
    df['pca-two'] = feature_2d[:, 1]
    df['label'] = labels

    sns.scatterplot(
        x='pca-one', y='pca-two',
        hue='label',
        palette=sns.color_palette("hls", num_class),
        data=df,
        legend="full",
        alpha=0.7
    )

    plt.xlabel("")
    plt.ylabel("")
    fig.tight_layout()
    fig.savefig(filename)