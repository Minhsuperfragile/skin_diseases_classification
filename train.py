from torch.utils.data import DataLoader
import model as md
from data_loader import image_dataset
import json

batch_size = 32
num_workers = 4

data_loaders = {x: DataLoader(image_dataset[x], shuffle=True, batch_size=batch_size, num_workers=4)
    for x in ['train', 'test']
}

epoch_loss = md.train(data_loaders, iters=50)

with open(r"model_eval/overall_loss.json", "w") as f:
    json.dump(epoch_loss, f)