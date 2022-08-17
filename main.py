"""
  This is main module to use for the project and preidct in production.

Usage:
    $ python main.py --image img.png

Code example:
    from PIL import Image
    import numpy as np

    from main import load_model, predict

    img = np.array(Image.open('img.png'))
    model, device = load_model()

    pred = predict(model, device, img)
    print(pred)
"""
from torch_utils import *
from sys_utils import *
from pathlib import Path
import torch
from general import non_max_suppression


def load_model(
    cfg="./yolov5n.yaml", weights_dic="./best_yolo5n_iris_pupil-dict.pt", nc=2
):
    """Load model from weights and config"""
    from yolo5.models.yolo import Model
    import torch

    device = torch.device("cpu")
    model = Model(cfg, ch=3, nc=2).float()
    model.fuse()
    model.load_state_dict(torch.load(weights_dic))
    model.eval()

    return model, device


def load_model_all(weights="./best_yolo5n_iris_pupil.pt"):
    """[DO NOT USE IT!!!] This is slow version of load model, it loads entire model like in training.

    Args:
        weights (str, optional):  exported wereights of the model. Defaults to "./best_yolo5n_iris_pupil.pt".

    Returns:
        model: model
        device: device
    """
    device = torch.device("cpu")
    model = torch.load(weights)["model"].float()
    model.eval()
    return model, device


IRIS_MM = 11.4


def convert_pupil_px2mm_item(pupil_diameter, iris_diameter, iris_mm=IRIS_MM):
    if iris_diameter <= 0:
        iris_diameter = 220
    return pupil_diameter * iris_mm / iris_diameter


def predict(model, device, img, prob_treshold=0.5):
    img = as_model_image(img, device)
    pred = predict_xywh(model, img).detach().numpy()

    iris = None
    pupil = None

    for p in pred:
        if p[1] < 1.0:  # iris
            iris = {
                "prob": p[0],
                "x": p[2],
                "y": p[3],
                "w": p[4],
                "h": p[5],
            }
        else:
            pupil = {
                "prob": p[0],
                "x": p[2],
                "y": p[3],
                "w": p[4],
                "h": p[5],
            }
    if iris is None:
        iris_prob = 0.0
        iris_px = 220.0
    else:
        iris_prob = iris["prob"]
        iris_px = max(iris["w"], iris["h"])

    if pupil is None:
        pupil_prob = 0.0
        pupil_px = 0.0
        pupil_mm = 0.0
    else:
        pupil_prob = pupil["prob"]
        pupil_px = (pupil["w"] + pupil["h"]) / 2.0

        if pupil_prob > prob_treshold and iris_prob > prob_treshold:
            pupil_mm = convert_pupil_px2mm_item(pupil_px, iris_px)
        else:
            pupil_mm = 0.0
    return {
        "prob_treshold": prob_treshold,
        "iris_prob": iris_prob,
        "iris_px": iris_px,
        "pupil_prob": pupil_prob,
        "pupil_px": pupil_px,
        "pupil_mm": pupil_mm,
        "pupil": pupil,
        "iris": iris,
    }


def as_model_image(im, device):
    assert im.shape == (640, 640, 3), "Image shape must be 640,640,3"

    img = torch.from_numpy(im).to(device, non_blocking=True).permute(2, 0, 1)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    return img.unsqueeze(dim=0)


def predict_xywh(self, im, conf_thres=0.25, iou_thres=0.45):
    classes = None
    agnostic_nms = False
    max_det = 2  # maximum detections per image
    pred = self(im)[0]

    # NMS
    pred = non_max_suppression(
        pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det
    )[0]

    w = abs(pred[:, 0] - pred[:, 2])
    h = abs(pred[:, 1] - pred[:, 3])

    pred[:, 0] = pred[:, 0] + w / 2.0
    pred[:, 1] = pred[:, 1] + h / 2.0
    pred[:, 2] = w
    pred[:, 3] = h

    pred = pred[:, [4, 5, 0, 1, 2, 3]]
    return pred


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(epilog="\n----------------------------\n")
    parser.add_argument("--image", type=str, default="./img.png")
    parser.add_argument("--loop", type=int, default=5)
    parser.add_argument("--prob", type=float, default=0.5)

    args, _ = parser.parse_known_args()
    img_path = args.image
    prob = args.prob

    from PIL import Image
    import datetime
    import numpy as np

    img = np.array(Image.open(img_path))
    model, device = load_model()

    st = datetime.datetime.now()
    from tqdm.auto import trange

    for i in trange(args.loop):
        pred = predict(model, device, img, prob)
    end = datetime.datetime.now() - st

    print(
        f"\n\nAvg Time predict: {end.total_seconds() / i / 60.0:0.4f} s\nPredictions:\n"
    )
    for k in pred.keys():
        print(f"{k}: {pred[k]}")
