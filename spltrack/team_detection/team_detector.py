from pathlib import Path
from typing import List, Union

import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models import vgg16_bn
from PIL.Image import Image as PILImage

from .model import Model


class TeamDetector:
    def __init__(
        self,
        checkpoint_file_path: Path,
        score_thresh: float = 0.5,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
    ):
        self.score_thresh = score_thresh
        self.device = device
        self._model = Model(model_base=vgg16_bn(pretrained=True), num_class1=10)
        self._model.load_state_dict(
            torch.load(
                str(checkpoint_file_path),
                map_location=self.device,
            )["model_state_dict"]
        )
        self._model.to(self.device)
        self._model.eval()
        self._preproc_transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize((200, 200)),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    @classmethod
    def from_config(cls, cfg):
        team_detector_cfg = cfg.TEAM_DETECTOR
        return cls(
            checkpoint_file_path=Path(team_detector_cfg.WEIGHTS),
            score_thresh=team_detector_cfg.SCORE_THRESH,
            device=torch.device(team_detector_cfg.DEVICE),
        )

    def __call__(self, images: List[Union[np.ndarray, PILImage]]):

        # Preprocess and batch inputs
        inputs = torch.stack([self._preproc_transform(img) for img in images])
        inputs = inputs.to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self._model(inputs)

        # convert output to probs using softmax
        team_probs = torch.nn.functional.softmax(outputs, 1)

        return team_probs.detach().cpu().numpy()
