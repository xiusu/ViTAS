import torch.utils.data as data
import os
import torchvision.transforms as transforms
from PIL import Image
import mc
import io
from data.randaugment import rand_augment_transform
import torch


class DatasetCache(data.Dataset):
    def __init__(self):
        super().__init__()
        self.initialized = False
    

    def _init_memcached(self):
        if not self.initialized:
            server_list_config_file = "/mnt/lustreold/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustreold/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def load_image(self, filename):
        self._init_memcached()
        value = mc.pyvector()
        self.mclient.Get(filename, value)
        value_str = mc.ConvertBuffer(value)
        
        buff = io.BytesIO(value_str)
        with Image.open(buff) as img:
            img = img.convert('RGB')
        return img



class BaseDataset(DatasetCache):
    def __init__(self, mode='train', max_class=1000, aug=None, use_randaugment=False, default_res=224, imagenet22k=False):
        super().__init__()
        self.initialized = False


        if not imagenet22k:
            prefix = '/mnt/lustreold/share/images/meta'
            image_folder_prefix = '/mnt/lustreold/share/images'
            if mode == 'train':
                image_list = os.path.join(prefix, 'train.txt')
                self.image_folder = os.path.join(image_folder_prefix, 'train')
            elif mode == 'test':
                image_list = os.path.join(prefix, 'test.txt')
                self.image_folder = os.path.join(image_folder_prefix, 'test')
            elif mode == 'val':
                image_list = os.path.join(prefix, 'val.txt')
                self.image_folder = os.path.join(image_folder_prefix, 'val')
            else:
                raise NotImplementedError('mode: ' + mode + ' does not exist please select from [train, test, eval]')
                
            self.samples = []
            with open(image_list) as f:
                for line in f:
                    name, label = line.split()
                    label = int(label)
                    if label < max_class:
                        self.samples.append((label, name))
        
        else:
            prefix = '/mnt/lustreold/share/imagenet22k'
            label_path =  os.path.join(prefix, 'label_list.txt')
            image_list = os.path.join(prefix, 'list.txt')

            label_dict = {}
            cnt = 0
            with open(label_path,'r') as f:
                for line in f:
                    linestr = line.strip()
                    label_dict[linestr] = cnt
                    cnt += 1

            self.samples = []
            self.image_folder = prefix
            with open(image_list) as f:
                for line in f:
                    linestr = line.strip().split('/')
                    label = label_dict[linestr[2]]
                    name = '/'.join(linestr[1:])
                    self.samples.append((label, name))


        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(
            translate_const=int(224 * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
        )

        if aug is None:
            if mode == 'train':
                aug = [
                    transforms.RandomResizedCrop(default_res),
                    transforms.RandomHorizontalFlip(),
                ]
                if use_randaugment:
                    aug.append(rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 9), ra_params))
                aug.extend([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
                ])
                self.transform = transforms.Compose(aug)

            else:
                eval_res = 256 if default_res == 224 else 384
                self.transform = transforms.Compose([
                    transforms.Resize(eval_res),
                    transforms.CenterCrop(default_res),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]),
                ])

        else:
            self.transform = aug


class Imagenet(BaseDataset):
    def __init__(self, mode='train', max_class=1000, aug=None, use_randaugment=False, default_res=224, imagenet22k=False):
        super().__init__(mode, max_class, aug, use_randaugment, default_res, imagenet22k)

    def __len__(self):
        return self.samples.__len__()

    def __getitem__(self, index):
        label, name = self.samples[index]
        filename = os.path.join(self.image_folder, name)
        img = self.load_image(filename)
        return self.transform(img), label



class ImagenetContrastive(BaseDataset):
    def __init__(self, mode='train', max_class=1000, aug=None):
        super().__init__(mode, max_class, aug)

    def __len__(self):
        return self.samples.__len__()

    def __getitem__(self, index):
        _, name = self.samples[index]
        filename = os.path.join(self.image_folder, name)
        img = self.load_image(filename)
        if isinstance(self.transform, list):
            return self.transform[0](img), self.transform[1](img)
        return self.transform(img), self.transform(img)



class ImagenetContrastiveWithIndex(BaseDataset):
    def __init__(self, mode='train', max_class=1000, topk=4, aug=None, crop_aug=None, knn=None):
        super().__init__(mode, max_class, aug)

        self.topk = topk
        self.knn = knn
        self.multi_crop = crop_aug

    def __len__(self):
        return self.samples.__len__()

    def __getitem__(self, index):
        _, name = self.samples[index]
        filename = os.path.join(self.image_folder, name)
        img = self.load_image(filename)

        similar = []
        if not self.knn is None:
            for i in self.knn[index]:
                _, name = self.samples[i]
                filename = os.path.join(self.image_folder, name)
                similar_img = self.load_image(filename)
                similar.append(self.multi_crop(similar_img))
        else:
            for _ in range(self.topk):
                similar.append(self.multi_crop(img))

        similar = torch.stack(similar)
        return self.transform(img), self.transform(img), index, similar




