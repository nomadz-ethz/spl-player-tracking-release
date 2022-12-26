from pathlib import Path
from typing import List, Union

import numpy as np
import torch
import torchvision.transforms as T
from PIL.Image import Image as PILImage

from .model_utils import build_number_classifier


class NumberDetector:
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
        self._model = build_number_classifier()
        self._model.load_state_dict(
            torch.load(
                str(checkpoint_file_path),
                map_location=device,
            ),
        )
        self._model.to(self.device)
        self._model.eval()
        self._preproc_transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize((80, 64)),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @classmethod
    def from_config(cls, cfg):
        return cls(
            checkpoint_file_path=Path(cfg.NUMBER_DETECTOR.WEIGHTS),
            score_thresh=cfg.NUMBER_DETECTOR.SCORE_THRESH,
            device=torch.device(cfg.NUMBER_DETECTOR.DEVICE),
        )

    def __call__(self, images: List[Union[np.ndarray, PILImage]]) -> List[int]:

        # Preprocess and batch inputs
        inputs = torch.stack([self._preproc_transform(img) for img in images]).to(
            self.device
        )

        # Do not consider images with invalid aspect ratio
        invalid_inputs_mask = torch.tensor(
            [img.height < img.width for img in images],
            device=self.device,
        )

        # Predictions
        with torch.no_grad():
            outputs = self._model(inputs)
        preds = torch.max(outputs, 1)[1]

        # Probs
        probs = torch.nn.functional.softmax(outputs, dim=1)
        invalid_preds_mask = torch.lt(torch.max(probs, 1)[0], self.score_thresh)

        # Set invalid preds to zero
        preds[
            torch.logical_or(
                invalid_inputs_mask,
                invalid_preds_mask,
            )
        ] = torch.tensor(0)

        return preds.detach().cpu().numpy()
