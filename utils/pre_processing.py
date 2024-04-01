### >>> Import ###
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
### <<< Import ###

### Histogram equalization ###
def histogram_equalization(image, plot = False):
    if image is None:
        return None
    if isinstance(image, torch.Tensor):
       Tensor_tracker = True
    else:
        Tensor_tracker = False
    if not isinstance(image, np.ndarray):
        image = image.numpy()
    if not image.dtype == np.uint8:
        image = image.astype(np.uint8)

    colorimage_b = cv2.equalizeHist(image[:, :, 0])
    colorimage_g = cv2.equalizeHist(image[:, :, 1])
    colorimage_r = cv2.equalizeHist(image[:, :, 2])

    new_image = np.stack((colorimage_b, colorimage_g, colorimage_r), axis=2)

    if plot:
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            hist, bins = np.histogram(image[:, :, i], bins=256, range=[0, 256])
            plt.bar(bins[:-1], hist, color=col, width=1)
            plt.title('Before histogram equalization')
            plt.ylim([0, max(hist[hist != 0]) * 1.2])
            plt.xlim([0, 256])
        plt.show()

        for i, col in enumerate(color):
            hist, bins = np.histogram(new_image[:, :, i], bins=256, range=[0, 256])
            plt.bar(bins[:-1], hist, color=col, width=1)
            plt.title('After histogram equalization')
            plt.ylim([0, max(hist[hist != 0]) * 1.2])
            plt.xlim([0, 256])
        plt.show()

    if Tensor_tracker == True:
        new_image = torch.Tensor(new_image)

    return new_image

### Augmentation ###
def data_augmentation(image, rep = 1, rotation = 2, brightness = 0.2,
                      contrast = 0.1, saturation = 0.1, hue = 0.01, scale = 0.15,
                      height = None, width = None, px_w = None, px_h = None, seed = 123):

    torch.manual_seed(seed)
    if not isinstance(image, torch.Tensor):
        image = transforms.ToTensor()(image)

    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert to PIL Image
        transforms.RandomRotation(degrees=rotation),
        transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
        transforms.RandomAffine(degrees=0, translate=(0, 0), scale=((1-scale), (1+scale))),
        transforms.ToTensor()
    ])

    if px_w is not None:
        h, w = height, width
        px_h, px_w = px_h / 2, px_w / 2
        h, w = h + px_h, w + px_w
        px_h = int(px_h)
        px_w = int(px_w)
        h = int(h)
        w = int(w)

    augmented_images = []
    for _ in range(rep):
        transformed_image = transform(image.clone())
        transformed_image = transformed_image.permute(1, 2, 0).numpy()
        transformed_image = (transformed_image * 255).astype(np.uint8)

        if px_w is not None:
            transformed_image = transformed_image[px_h:h, px_w:w, :]

        augmented_images.append(transformed_image)

    return image, augmented_images

