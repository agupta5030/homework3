from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class Classifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
    ):
        """
        A convolutional network for image classification.

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        # Building a CNN for classification
        # Starting with a few conv layers, then pooling and fully connected layers
        # Input is 64x64 images with 3 channels
        
        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)  # reduces to 32x32
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)  # reduces to 16x16
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)  # reduces to 8x8
        
        # Fourth convolutional block
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)  # reduces to 4x4
        
        # Flatten and fully connected layers
        # After pooling: 256 channels * 4 * 4 = 4096 features
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.relu_fc = nn.ReLU()
        
        # Final output layer for 6 classes
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """
        # Normalize the input using mean and std
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

           # TODO: replace with actual forward pass
        # logits = torch.randn(x.size(0), 6)
        # Forward pass through convolutional layers
        z = self.conv1(z)
        z = self.bn1(z)
        z = self.relu1(z)
        z = self.pool1(z)
        
        z = self.conv2(z)
        z = self.bn2(z)
        z = self.relu2(z)
        z = self.pool2(z)
        
        z = self.conv3(z)
        z = self.bn3(z)
        z = self.relu3(z)
        z = self.pool3(z)
        
        z = self.conv4(z)
        z = self.bn4(z)
        z = self.relu4(z)
        z = self.pool4(z)
        
        # Flatten and go through fully connected layers
        z = self.flatten(z)
        z = self.fc1(z)
        z = self.dropout1(z)
        z = self.relu_fc(z)
        
        # Final output layer
        logits = self.fc2(z)

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels
        This is what the AccuracyMetric uses as input (this is what the grader will use!).
        You should not have to modify this function.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            pred (torch.LongTensor): class labels {0, 1, ..., 5} with shape (b, h, w)
        """
        return self(x).argmax(dim=1)


class Detector(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
    ):
        """
        A single model that performs segmentation and depth regression

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        # Added code for U-Net style architecture for segmentation and depth 
        # Used claude for understanding logic and explanation of code
        
        # Downsampling layers (encoder)
        # Input: (B, 3, H, W) -> (B, 16, H/2, W/2)
        self.down1_conv = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.down1_bn = nn.BatchNorm2d(16)
        self.down1_relu = nn.ReLU()
        self.down1_pool = nn.MaxPool2d(2, 2)
        
        # (B, 16, H/2, W/2) -> (B, 32, H/4, W/4)
        self.down2_conv = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.down2_bn = nn.BatchNorm2d(32)
        self.down2_relu = nn.ReLU()
        self.down2_pool = nn.MaxPool2d(2, 2)
        
        # (B, 32, H/4, W/4) -> (B, 64, H/8, W/8)
        self.down3_conv = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.down3_bn = nn.BatchNorm2d(64)
        self.down3_relu = nn.ReLU()
        self.down3_pool = nn.MaxPool2d(2, 2)
        
        # Bottleneck layer
        self.bottleneck_conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bottleneck_bn = nn.BatchNorm2d(128)
        self.bottleneck_relu = nn.ReLU()
        
        # Upsampling layers (decoder)
        # (B, 128, H/8, W/8) -> (B, 64, H/4, W/4)
        self.up1_conv = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up1_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)  # 128 because of skip connection
        self.up1_bn = nn.BatchNorm2d(64)
        self.up1_relu = nn.ReLU()
        
        # (B, 64, H/4, W/4) -> (B, 32, H/2, W/2)
        self.up2_conv = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.up2_conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)  # 64 because of skip connection
        self.up2_bn = nn.BatchNorm2d(32)
        self.up2_relu = nn.ReLU()
        
        # (B, 32, H/2, W/2) -> (B, 16, H, W)
        self.up3_conv = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.up3_conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)  # 32 because of skip connection
        self.up3_bn = nn.BatchNorm2d(16)
        self.up3_relu = nn.ReLU()
        
        # Final output heads
        # Segmentation head: (B, 16, H, W) -> (B, num_classes, H, W)
        self.seg_head = nn.Conv2d(16, num_classes, kernel_size=1)
        
        # Depth head: (B, 16, H, W) -> (B, 1, H, W) -> (B, H, W)
        self.depth_head = nn.Conv2d(16, 1, kernel_size=1)
        self.depth_sigmoid = nn.Sigmoid()  # To constrain depth to [0, 1]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used in training, takes an image and returns raw logits and raw depth.
        This is what the loss functions use as input.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.FloatTensor, torch.FloatTensor):
                - logits (b, num_classes, h, w)
                - depth (b, h, w)
        """
        # Normalize the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # TODO: replace with actual forward pass
        #logits = torch.randn(x.size(0), 3, x.size(2), x.size(3))
        #raw_depth = torch.rand(x.size(0), x.size(2), x.size(3))
        # Downsampling path (encoder)
        # First down block
        d1 = self.down1_conv(z)
        d1 = self.down1_bn(d1)
        d1 = self.down1_relu(d1)
        skip1 = d1  # Save for skip connection
        d1 = self.down1_pool(d1)
        
        # Second down block
        d2 = self.down2_conv(d1)
        d2 = self.down2_bn(d2)
        d2 = self.down2_relu(d2)
        skip2 = d2  # Save for skip connection
        d2 = self.down2_pool(d2)
        
        # Third down block
        d3 = self.down3_conv(d2)
        d3 = self.down3_bn(d3)
        d3 = self.down3_relu(d3)
        skip3 = d3  # Save for skip connection
        d3 = self.down3_pool(d3)
        
        # Bottleneck
        bottleneck = self.bottleneck_conv(d3)
        bottleneck = self.bottleneck_bn(bottleneck)
        bottleneck = self.bottleneck_relu(bottleneck)
        
        # Upsampling path (decoder) with skip connections
        # First up block with skip connection
        u1 = self.up1_conv(bottleneck)
        # Concatenate with skip connection from down path
        u1 = torch.cat([u1, skip3], dim=1)  # Concatenate along channel dimension
        u1 = self.up1_conv2(u1)
        u1 = self.up1_bn(u1)
        u1 = self.up1_relu(u1)
        
        # Second up block with skip connection
        u2 = self.up2_conv(u1)
        u2 = torch.cat([u2, skip2], dim=1)
        u2 = self.up2_conv2(u2)
        u2 = self.up2_bn(u2)
        u2 = self.up2_relu(u2)
        
        # Third up block with skip connection
        u3 = self.up3_conv(u2)
        u3 = torch.cat([u3, skip1], dim=1)
        u3 = self.up3_conv2(u3)
        u3 = self.up3_bn(u3)
        u3 = self.up3_relu(u3)
        
        # Generate outputs from the two heads
        # Segmentation logits: (B, num_classes, H, W)
        logits = self.seg_head(u3)
        
        # Depth prediction: (B, 1, H, W) -> (B, H, W)
        depth = self.depth_head(u3)
        depth = self.depth_sigmoid(depth)
        raw_depth = depth.squeeze(1)  # Remove channel dimension to get (B, H, W)

        return logits, raw_depth

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.
        This is what the metrics use as input (this is what the grader will use!).

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        logits, raw_depth = self(x)
        pred = logits.argmax(dim=1)

        # Optional additional post-processing for depth only if needed
        depth = raw_depth

        return pred, depth


MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Test your model implementation

    Feel free to add additional checks to this function -
    this function is NOT used for grading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)

    # should output logits (b, num_classes)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()
