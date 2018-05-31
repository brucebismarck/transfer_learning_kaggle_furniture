#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 21:36:41 2018

@author: wenyue
"""
import random
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image
import collections
_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
}



def ds_trans(): # play around here
    ds_trans = transforms.Compose([transforms.Resize((224,224)), # Resize the input PIL Image to given size
                                   transforms.CenterCrop((224,224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                   ]) 
    # Normalization method is std according to torchvision tutorial
    # http://pytorch.org/docs/master/torchvision/models.html
    return ds_trans

def ds_trans_aug():
    ds_trans = transforms.Compose([RandomResize(0.6, 0.7, (256, 256), (291, 291) ),
                                   transforms.CenterCrop((224,224)),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation= 0.3),
                                   transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                   ])
    return ds_trans

#%%

class RandomResize(object):
    def __init__(self, p1, p2, size1, size2, interpolation = Image.BILINEAR):
        assert isinstance(size1, int) or (isinstance(size1, collections.Iterable) and len(size1) == 2)
        assert isinstance(size2, int) or (isinstance(size2, collections.Iterable) and len(size2) == 2)
        self.p1 = p1
        self.p2 = p2
        self.size1 = size1
        self.size2 = size2
        self.interpolation = interpolation
        
    def __call__(self, img):
        random_num = random.random()
        if  random_num < self.p1:
            return F.resize(img, self.size1, self.interpolation)
        elif random_num < self.p2: 
            return F.resize(img, self.size2, self.interpolation)
        else:
            return img
    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class RandomTransforms(object):
    """Base class for a list of transformations with randomness
    Args:
        transforms (list or tuple): list of transformations
    """

    def __init__(self, transforms):
        assert isinstance(transforms, (list, tuple))
        self.transforms = transforms

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

    
class RandomApply(RandomTransforms):
    """Apply randomly a list of transformations with a given probability
    Args:
        transforms (list or tuple): list of transformations
        p (float): probability
    """

    def __init__(self, transforms, p=0.5):
        super(RandomApply, self).__init__(transforms)
        self.p = p

    def __call__(self, img):
        if self.p < random.random():
            return img
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n    p={}'.format(self.p)
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
    
class RandomChoice(RandomTransforms):
    """Apply single transformation randomly picked from a list
    """
    def __call__(self, img):
        t = random.choice(self.transforms)
        return t(img)
    
class HorizontalFlip(object):
    """Horizontally flip the given PIL Image."""

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Flipped image.
        """
        return F.hflip(img)

class Brightness(object):
    def __init__(self, brightness = 1 ):
        self.brightness = brightness
    def __call__(self, img):        
        return F.adjust_brightness(img, self.brightness)

class Contrast(object):
    def __init__(self, contrast = 1 ):
        self.contrast = contrast
    def __call__(self, img):
        return F.adjust_contrast(img, self.contrast)

class Saturation(object):
    def __init__(self, saturation_factor = 1 ):
        self.saturation_factor = saturation_factor
    def __call__(self, img):
        return F.adjust_contrast(img, self.saturation_factor)        

class Gamma(object):
    def __init__(self, gamma_factor = 1 ):
        self.gamma_factor = gamma_factor
    def __call__(self, img):
        return F.adjust_gamma(img, self.gamma_factor, 1)

preprocess_aug = ds_trans_aug()

preprocess = ds_trans()

preprocess_hflip = transforms.Compose([
    transforms.Resize((224, 224)), #if size is an int,  the image will keep shape.
    HorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
    
preprocess_scale_1 = transforms.Compose([  # rescale
        transforms.Resize((256,256)),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
preprocess_scale_2 = transforms.Compose([  # rescale
        transforms.Resize((256)),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
preprocess_brightness_darker = transforms.Compose([
        transforms.Resize((224,224)),
        Brightness(0.9),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

preprocess_brightness_lighter = transforms.Compose([
        transforms.Resize((224,224)),
        Brightness(1.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
    
preprocess_contrast_blur = transforms.Compose([
        transforms.Resize((224,224)),
        Contrast(0.9),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

preprocess_contrast_sharp = transforms.Compose([
        transforms.Resize((224,224)),
        Contrast(1.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

preprocess_saturation_pale = transforms.Compose([
        transforms.Resize((224,224)),
        Saturation(0.9),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

preprocess_saturation_saturated = transforms.Compose([
        transforms.Resize((224,224)),
        Saturation(1.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])    

preprocess_gamma_small = transforms.Compose([
        transforms.Resize((224,224)),
        Gamma(0.8),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])    

preprocess_gamma_large = transforms.Compose([
        transforms.Resize((224,224)),
        Gamma(1.25),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])      
    
    
    