import torch
import torch.nn as nn
from torch.nn import functional as F

class ActionPredictorMLP(nn.Module):
    """
    predict abs_action[t] based on delta_action[t] and abs_action[t-1]
    Input: delta_action[t] (7) + abs_action[t-1] (8) = 15
    Output: abs_action[t] (8)
    """

    def __init__(
        self,
        input_dim=15,
        output_dim=8,
        hidden_dims=[64, 64],
        dropout=0.0,
        abs_action_mean=None,
        abs_action_std=None,
    ):
        super(ActionPredictorMLP, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Register normalization parameters as buffers
        if abs_action_mean is not None:
            if isinstance(abs_action_mean, torch.Tensor):
                self.register_buffer("abs_action_mean", abs_action_mean)
            else:
                self.register_buffer(
                    "abs_action_mean",
                    torch.tensor(abs_action_mean, dtype=torch.float32),
                )
        else:
            self.register_buffer("abs_action_mean", None)

        if abs_action_std is not None:
            if isinstance(abs_action_std, torch.Tensor):
                self.register_buffer("abs_action_std", abs_action_std)
            else:
                self.register_buffer(
                    "abs_action_std", torch.tensor(abs_action_std, dtype=torch.float32)
                )
        else:
            self.register_buffer("abs_action_std", None)

    def forward(self, delta_action, abs_action_prev):
        """
        Args:
            delta_action: (batch_size, 7) - delta_action[t]
            abs_action_prev: (batch_size, 8) - abs_action[t-1] (should be normalized)
        Returns:
            abs_action_pred: (batch_size, 8) - 预测的 abs_action[t] (normalized)
        """
        x = torch.cat([delta_action, abs_action_prev], dim=1)
        return self.network(x)

    def get_ee_pose(self, delta_action, pre_ee_pose):
        """
        Get next ee_pose with automatic normalize/unnormalize

        Args:
            delta_action: (batch_size, 7) - delta_action[t]
            pre_ee_pose: (batch_size, 8) - previous ee_pose (unnormalized)
        Returns:
            next_ee_pose: (batch_size, 8) - next ee_pose (unnormalized)
        """
        if self.abs_action_mean is None or self.abs_action_std is None:
            raise ValueError(
                "abs_action_mean and abs_action_std must be set for get_ee_pose()"
            )

        # Normalize pre_ee_pose
        pre_ee_pose_normalized = (
            pre_ee_pose - self.abs_action_mean
        ) / self.abs_action_std

        # Forward pass (returns normalized next_ee_pose)
        next_ee_pose_normalized = self.forward(delta_action, pre_ee_pose_normalized)

        # Unnormalize next_ee_pose
        next_ee_pose = (
            next_ee_pose_normalized * self.abs_action_std + self.abs_action_mean
        )

        return next_ee_pose

    def set_normalization_params(self, abs_action_mean, abs_action_std):
        """
        Set normalization parameters after initialization

        Args:
            abs_action_mean: (1, 8) or (8,) tensor or array - mean for normalization
            abs_action_std: (1, 8) or (8,) tensor or array - std for normalization
        """
        if isinstance(abs_action_mean, torch.Tensor):
            self.register_buffer("abs_action_mean", abs_action_mean)
        else:
            self.register_buffer(
                "abs_action_mean", torch.tensor(abs_action_mean, dtype=torch.float32)
            )

        if isinstance(abs_action_std, torch.Tensor):
            self.register_buffer("abs_action_std", abs_action_std)
        else:
            self.register_buffer(
                "abs_action_std", torch.tensor(abs_action_std, dtype=torch.float32)
            )


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels, num_channels, kernel_size=3, padding=1, stride=strides
        )
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(
                input_channels, num_channels, kernel_size=1, stride=strides
            )
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(
                Residual(input_channels, num_channels, use_1x1conv=True, strides=2)
            )
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


class RewModel(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        b1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        b3 = nn.Sequential(*resnet_block(64, 128, 2))
        b4 = nn.Sequential(*resnet_block(128, 256, 2))
        b5 = nn.Sequential(*resnet_block(256, 512, 2))
        self.net = nn.Sequential(
            b1,
            b2,
            b3,
            b4,
            b5,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    @torch.no_grad()
    def predict_rew(self, obs):
        assert obs.max() <= 1.5 and obs.min() >= -1.5, f"obs.max() is {obs.max()}, and obs min is {obs.min()}"
        obs = obs.clamp(-1.0, 1.0)
        x = self.net(obs.to(dtype=torch.float32))
        # 分为 0 或 1
        x = torch.round(x)
        return x

    def forward(self, obs=None):
        return self.predict_rew(obs)