import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, input_height, input_width, num_classes):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
       
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        

        self.feature_height = input_height // 4
        self.feature_width = input_width // 4
        
       
        self.fc1 = nn.Linear(64 * self.feature_height * self.feature_width, 128)
       
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        
        x = F.relu(self.conv1(x))  
        x = self.pool(x)            
        x = F.relu(self.conv2(x))   
        x = self.pool(x)           
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))     
        x = self.fc2(x)             
        return x

# Example usage
if __name__ == '__main__':
   
    input_height, input_width = 28, 28  
    num_classes = 10  
    
   
    model = SimpleCNN(input_height, input_width, num_classes)

    sample_input = torch.randn(1, 1, input_height, input_width)

    output = model(sample_input)
    
    print("Output shape:", output.shape)  