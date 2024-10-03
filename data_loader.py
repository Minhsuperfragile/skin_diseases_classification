from torchvision import datasets, transforms

data_set_path = "/mnt/g/Code/Dataset/skin_illness"

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_dataset = {
    x: datasets.ImageFolder(data_set_path, data_transforms[x])
    for x in ['train', 'test']
}

if __name__ == "__main__":
    from os.path import exists
    print(exists(data_set_path))
    print(image_dataset)