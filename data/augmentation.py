import random
import torchvision.transforms as transforms
from PIL import ImageFilter, ImageOps, Image
from data.randaugment import rand_augment_transform
from data.random_erasing import RandomErasing

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize():
    def __init__(self, threshold=128):
        self.threshold = threshold
    def __call__(self, sample):
        return ImageOps.solarize(sample, self.threshold)


def get_deit_test_aug(res=224):
    
    eval_res = 256 if res == 224 else 384
    self.transform = transforms.Compose([
        transforms.Resize(eval_res),
        transforms.CenterCrop(res),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
    ])
    return transforms.Compose(augs)



def get_deit_aug(res=224, erase=True):
    mean = (0.485, 0.456, 0.406)
    aa_params = dict(
            translate_const=int(res * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
    augs = [
        transforms.RandomResizedCrop(res, interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        rand_augment_transform('rand-m9-mstd0.5-inc1', aa_params),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    if erase:
        augs.append(RandomErasing(0.25, mode='pixel', max_count=1, num_splits=0, device='cpu'))
    return transforms.Compose(augs)

    
def get_deit_xaa_xcj_aug(res=224, erase=True):
    mean = (0.485, 0.456, 0.406)
    augs = [
        transforms.RandomResizedCrop(res, interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    if erase:
        augs.append(RandomErasing(0.25, mode='pixel', max_count=1, num_splits=0, device='cpu'))
    return transforms.Compose(augs)




weak_aug = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


eval_aug = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
])
