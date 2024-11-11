import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import openvino as ov
import comet_ml 
from time import time
import numpy as np

num_images = 1000
batch_size = 1


# Load the MNIST dataset with the necessary transformations
transform = transforms.Compose([
    transforms.Grayscale(3),  # Convert to 3 channels
    transforms.Resize((224, 224)),  # Resize to 224x224, expected input size for MobileNetV3
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize images
])

dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
mnist_dataset = torch.utils.data.Subset(dataset,list(range(0, 1000)))
mnist_loader = DataLoader(mnist_dataset, batch_size=1, shuffle=False)

# Load the pre-trained MobileNetV3 model
model = models.mobilenet_v3_small(pretrained=True)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 10)  # Adjust for 10 MNIST classes

compiled_model = ov.compile_model(ov.convert_model(model, example_input=torch.rand([1, 3, 224, 224])))

# Set model to evaluation mode
model.eval()

# Initialize lists to store true and predicted labels
true_labels = []
predicted_labels = []

comet_ml.login()
experiment = comet_ml.start(project_name="ai_pc_inf_demo")
experiment.add_tag('openvino')
experiment.log_parameter('vino', True)
experiment.log_parameter('device', 'CPU')
experiment.log_parameter('num_images', num_images)
experiment.log_parameter('batch_size', batch_size)


start = time()

# Run inference on the first 1000 images, one at a time
for image, label in mnist_loader:

    # Run inference
    output = torch.tensor(compiled_model(image)[0])
    _, predicted = torch.max(output, 1)

    # Store the true and predicted labels
    true_labels.append(label.item())
    predicted_labels.append(predicted.item())
end = time()

latency = np.round(end - start, 3)/num_images
throughput = num_images/latency

experiment.log_metric('latency', latency)
experiment.log_metric('throughput', throughput)

# Calculate accuracy metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')
f1 = f1_score(true_labels, predicted_labels, average='weighted')

experiment.log_metric('accuracy', accuracy)
experiment.log_metric('precision', precision)
experiment.log_metric('recall', recall)
experiment.log_metric('f1', f1)