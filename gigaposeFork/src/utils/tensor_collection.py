import pandas as pd
import torch


def concatenate(datas):
    datas = [data for data in datas if len(data) > 0]
    if len(datas) == 0:
        return PandasTensorCollection(infos=pd.DataFrame())

    classes = [data.__class__ for data in datas]
    assert all(class_n == classes[0] for class_n in classes)

    infos = pd.concat([data.infos for data in datas], axis=0, sort=False).reset_index(
        drop=True
    )
    tensors = {
        key: torch.cat([getattr(data, key) for data in datas], dim=0)
        for key in datas[0].tensors.keys()
    }
    return PandasTensorCollection(infos=infos, **tensors)


class TensorCollection:
    def __init__(self, **kwargs):
        self.__dict__["_tensors"] = {}
        for key, value in kwargs.items():
            self.register_tensor(key, value)

    def register_tensor(self, name, tensor):
        self._tensors[name] = tensor

    def __getitem__(self, ids):
        return TensorCollection(**{key: value[ids] for key, value in self._tensors.items()})

    def __getattr__(self, name):
        if name in self._tensors:
            return self._tensors[name]
        if name in self.__dict__:
            return self.__dict__[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if "_tensors" not in self.__dict__:
            raise ValueError("Please call __init__ before setting attributes")
        if name in self._tensors:
            self._tensors[name] = value
        else:
            self.__dict__[name] = value

    @property
    def tensors(self):
        return self._tensors

    def to(self, torch_attr):
        for key, value in self._tensors.items():
            self._tensors[key] = value.to(torch_attr)
        return self


class PandasTensorCollection(TensorCollection):
    def __init__(self, infos, **tensors):
        super().__init__(**tensors)
        self.infos = infos.reset_index(drop=True)
        self.meta = {}

    def __getitem__(self, ids):
        infos = self.infos.iloc[ids].reset_index(drop=True)
        tensors = {key: value[ids] for key, value in self._tensors.items()}
        return PandasTensorCollection(infos, **tensors)

    def __len__(self):
        return len(self.infos)

    def cat_df(self, other):
        for key, value in self._tensors.items():
            self._tensors[key] = torch.cat([value, other._tensors[key]], dim=0)
        self.infos = pd.concat([self.infos, other.infos], ignore_index=True)
        return self
