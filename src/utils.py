import cv2
from prettytable import PrettyTable
import torch
import onnx


COLOURS = [
            (144, 238, 144),
            (43, 34, 144),
            (178, 34, 34),
            (221, 160, 221),
            (0, 255, 0),
            (0, 128, 0),
            (210, 105, 30),
            (220, 20, 60),
            (192, 192, 192),
            (255, 228, 196),
            (50, 205,50),
            (139, 0, 139),
            (100, 149, 237),
            (138, 43, 226),
            (238, 130, 238),
            (255, 0, 255),
            (0, 100, 0),
            (127, 255, 0),
            (255, 0,255),
            (0, 0, 205),
            (255, 140, 0)
            ]


def export_onnx(module, input, filepath):
    
    torch.onnx.export(module, 
                        args=input, 
                        f=filepath,
                        opset_version=12,
                        input_names = ['input'],
                        output_names = ['output'],
                        do_constant_folding=False)
    
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(filepath)), filepath)


def resize_image(image, width, height):
    return cv2.resize(image,
                        (width, height), 
                        interpolation = cv2.INTER_LINEAR)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        # if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Params: {human_format(total_params)}")
    return total_params


def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


