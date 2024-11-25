import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import img_as_ubyte
from cellpose import models, plot, io
from typing import List, Tuple


def load_cellpose_model(model_name: str) -> models.Cellpose:
    """_summary_

    Parameters
    ----------
    model_name : str
        _description_

    Returns
    -------
    models.Cellpose
        _description_
    """
    model = models.Cellpose(model_type=model_name)
    return model


def load_img_tiles(tile_dir: str) -> List[str]:
    """_summary_

    Parameters
    ----------
    tile_dir : str
        _description_

    Returns
    -------
    List[str]
        _description_
    """
    imgs = []
    for i in os.listdir(tile_dir):
       img = io.imread(f"{tile_dir}/{i}")
       imgs.append(img)
    return imgs


def cell_seg_inference(model:models.Cellpose, img:np.ndarray) -> Tuple:
    """_summary_

    Parameters
    ----------
    model : models.Cellpose
        _description_
    img : np.ndarray
        _description_

    Returns
    -------
    Tuple
        _description_
    """
    masks, flows, styles, diams = model.eval(img)
    return (masks, flows, styles, diams)


def vsualize_cell_seg(seg_out: Tuple, img: np.ndarray) -> None:
    """_summary_

    Parameters
    ----------
    seg_out : Tuple
        _description_
    img : np.ndarray
        _description_
    """
    if img.dtype != "uint8":
        img = img_as_ubyte(img)
    fig = plt.figure(figsize=(24,8))
    plot.show_segmentation(fig, img, seg_out[0], seg_out[1][0], channels=0)
    plt.tight_layout()
    plt.show()


def main(tile_dir:str, model_name: str, img_index: int=0) -> None:
    """_summary_

    Parameters
    ----------
    tile_dir : str
        _description_
    model_name : str
        _description_
    img_index : int, optional
        _description_, by default 0
    """
    assert img_index < len(os.listdir(tile_dir))
    imgs = load_img_tiles(tile_dir=tile_dir)
    model = load_cellpose_model(model_name=model_name)
    seg_out = cell_seg_inference(model=model, img=imgs[img_index])
    vsualize_cell_seg(seg_out=seg_out, img=imgs[img_index])


if __name__ =="__main__":
    main(tile_dir="data", model_name="cyto2", img_index=2)