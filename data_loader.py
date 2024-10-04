from torchvision import datasets, transforms

train_set_path = "/mnt/g/Code/Dataset/skin_illness"
test_set_path = "/mnt/g/Code/Dataset/skin_illness_test"

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
    "train": datasets.ImageFolder(train_set_path, data_transforms['train']),
    "test" : datasets.ImageFolder(test_set_path, data_transforms["test"])
}