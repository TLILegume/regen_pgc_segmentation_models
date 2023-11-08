# Pytorch Lightning Code

from typing import Any, Optional, Union
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.optim.optimizer import Optimizer

class LitDeepLab(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def configure_optimizers(self) -> Any:
        return super().configure_optimizers()
    
    def configure_gradient_clipping(self, optimizer: Optimizer, gradient_clip_val: int | float | None = None, gradient_clip_algorithm: str | None = None) -> None:
        return super().configure_gradient_clipping(optimizer, gradient_clip_val, gradient_clip_algorithm)

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return super().training_step(*args, **kwargs)
    
    def validation_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT | None:
        return super().validation_step(*args, **kwargs)
    

    
