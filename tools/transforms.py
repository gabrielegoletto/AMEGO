import torchvision.transforms as transforms


def default_mean_std():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return mean, std

def default_transform(split, is_torch=False):
    mean, std = default_mean_std()

    if is_torch:
        
        if split=='train':
            transform = transforms.Compose([
                            transforms.Lambda(lambda x: x/255),
                            transforms.Resize(256, antialias=True),
                            transforms.RandomCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.Normalize(mean, std)
                        ])


        elif split=='val':
            transform = transforms.Compose([
                            transforms.Lambda(lambda x: x/255),
                            transforms.Resize(256, antialias=True),
                            transforms.CenterCrop(224),
                            transforms.Normalize(mean, std)
                ])

        elif split=='val224':
            transform = transforms.Compose([
                        transforms.Lambda(lambda x: x/255),
                        transforms.Resize(224, antialias=True),
                        transforms.CenterCrop(224),
                        transforms.Normalize(mean, std)
                ])
            
    else:

        if split=='train':
            transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.RandomCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)
                        ])


        elif split=='val':
            transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)
                ])

        elif split=='val224':
            transform = transforms.Compose([
                        transforms.Resize(224),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std)
                ])


    return transform
