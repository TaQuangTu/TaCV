import torch
from torch.utils.data import random_split, Dataset
import cv2
from tacv.detection import CenterNetTrainer, CenterNet
import torch

from tacv.detection import load_centernet_model_with_config
from tacv.detection import infer

from tacv.fileUtils import ThreadedDownload

# class MockDataset(Dataset):
#     def __init__(self, max_objs):
#         self.max_objs = max_objs
#
#     def __getitem__(self, item):
#         image = torch.rand(3, 224, 448)  # Shape = (3, H, W)
#         annos = torch.rand(self.max_objs, 5)  # Shape = (MaxObjs x 5) , each row presents for (x,y,w,h,class_id)
#         masks = torch.zeros(
#             self.max_objs)  # Shape = (MaxObjs,)  each value is False or True (1 indicates having object)
#         masks[0:3] = True
#         return {"image": image, "annos": annos, "masks": masks}
#
#     def __len__(self):
#         return 1000


if __name__ == "__main__":
    # multi thread downloader
    urls = [
        'http://localhost:11223/210207000111980_3_124202.jpg',
        'http://localhost:11223/210430000373373_3_407200.jpg',
        'http://localhost:11223/210426000324979_3_356188.jpg',
        'http://localhost:11223/200819000060994_3_78314.png'
    ]
    downloader = ThreadedDownload(urls, "/home/tu/Desktop/LabelStudio", False, 3, 3)

    print(f'Downloading {len(urls)} files')
    downloader.run()
    print(f'Downloaded {len(downloader.report["success"])} of {len(urls)}')

    config_path = "/home/tu/Projects/PycharmProjects/TaCV/tacv/detection/sample_config.yml"
    dataset = MockDataset(max_objs=16)  # Replace with your custom dataset
    train_set, val_set = random_split(dataset, [900, 100])

    trainer = CenterNetTrainer(train_set, val_set, config_path)
    # trainer.train()
    # debug = 0

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = load_centernet_model_with_config(config_path, load_bbone_pretrained=False)
    model.load_state_dict("your_checkpoint.pth")
    model.eval()
    model.to(device)


    image = cv2.imread("your_image.png")
    bboxes = infer(model, image, device)
    print(bboxes)
