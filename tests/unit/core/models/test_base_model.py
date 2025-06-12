from mml.core.models.torch_base import BaseModel
from mml.testing.boring_model import BoringModel


def test_save_load(tmp_path, test_task_monkeypatch):
    tmp_path /= "model.mml"
    model = BoringModel()
    model.add_head(test_task_monkeypatch["test_task_a"])
    model.freeze_backbone()
    model.save_checkpoint(tmp_path)
    loaded_model = BaseModel.load_checkpoint(tmp_path)
    assert isinstance(loaded_model, BoringModel)
    assert "test_task_a" in loaded_model.heads
    assert loaded_model._frozen_params == model._frozen_params
    assert loaded_model.count_parameters(only_trainable=False) == model.count_parameters(only_trainable=False)
