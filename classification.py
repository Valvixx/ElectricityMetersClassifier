import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def create_classifier_model(device=None):
    weights = models.ResNet18_Weights.DEFAULT
    classifier = models.resnet18(weights=weights)
    classifier.fc = nn.Linear(classifier.fc.in_features, 2)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return classifier.to(device)


def load_classifier_model(weights_path="meter_classifier.pth", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classifier = create_classifier_model(device=device)
    state_dict = torch.load(weights_path, map_location=device)
    classifier.load_state_dict(state_dict)
    classifier.eval()
    return classifier


def train_classifier(
    train_dir="data/train",
    val_dir="data/val",
    weights_path="meter_classifier.pth",
    epochs=5,
    batch_size=32,
    learning_rate=1e-4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    classifier = create_classifier_model(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        classifier.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = classifier(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = classifier(inputs)
                predicted = torch.argmax(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total if total else 0
        print(f"Epoch {epoch + 1}/{epochs} done. Val accuracy: {accuracy:.2%}")

    torch.save(classifier.state_dict(), weights_path)
    return classifier


if __name__ == "__main__":
    train_classifier()
