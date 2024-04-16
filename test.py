import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms.transforms import ToPILImage
from math import ceil

# write here or import from train.py
class MLP_Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
                
        self.input_fc = nn.Linear(input_dim, 250)
        self.hidden_fc = nn.Linear(250, 100)
        self.output_fc = nn.Linear(100, output_dim)
        
    def forward(self, x):       
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        h_1 = F.relu(self.input_fc(x))
        h_2 = F.relu(self.hidden_fc(h_1))
        y_pred = self.output_fc(h_2)
        return y_pred, h_2

INPUT_DIMENSION = 150 * 150
OUTPUT_DIMENSION = 25

loaded_model = torch.load('tut1-model.pt',map_location=lambda storage, loc: storage)
model = MLP_Classifier(INPUT_DIMENSION, OUTPUT_DIMENSION)
model.load_state_dict(loaded_model)
model.eval()

mean = 0.8
std = 0.2

def test_function(test_images):
    test_images_int = torch.from_numpy(test_images).float()
    test_images_int = transforms.functional.normalize(test_images_int/255, mean, std)
    test_images_input = transforms.functional.normalize(test_images_int, mean, std)
    top_predication = []
    test_iterator = [test_images_input[x*44:(x+1)*44] for x in range(ceil(len(test_images_input)/44))]
    with torch.no_grad():
        for x in test_iterator:
            y_predication, _ = model(x.unsqueeze(1))            
            y_prob = F.softmax(y_predication, dim = -1)
            top_predication.extend(y_prob.argmax(1, keepdim = True).squeeze(1).numpy())
    return np.array(top_predication)

test_images = np.load("Images_final.npy")
predicted_labels = test_function(test_images)