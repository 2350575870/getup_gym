from dataclasses import dataclass, fields, asdict
from collections.abc import Mapping
from typing import List, Any, Dict, Iterator
import torch
import torch.nn as nn

@dataclass
class BaseConfig(Mapping):
    """the basic config class for all config class

    Args:
        Mapping (_type_): the father class
    """
    def __getitem__(self, key):
        """func used for data indexing

        Args:
            key (_type_): key wards used for indexing

        Returns:
            _type_: value
        """
        return getattr(self, key)
    
    def __iter__(self):
        """used for iteration

        Yields:
            _type_: iteration
        """
        for field in self.__dataclass_fields__:
            yield field
            
    def __len__(self):
        """the data length

        Returns:
            _type_: length
        """
        return len(self.__dataclass_fields__)
    
def configclass(cls):
    """
    Isaac Lab 风格的配置类装饰器
    为 dataclass 添加字典行为和验证功能
    """
    # 首先应用标准的 dataclass 装饰器
    cls = dataclass(cls)
    
    # 添加字典行为的方法
    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"'{type(self).__name__}' object has no attribute '{key}'")
    
    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)
    
    def __iter__(self) -> Iterator[str]:
        """支持迭代键，这是 ** 展开所必需的"""
        for field in fields(self):
            yield field.name
    
    def __len__(self) -> int:
        return len(fields(self))
    
    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)
    
    def keys(self):
        """返回所有字段名"""
        return [field.name for field in fields(self)]
    
    def values(self):
        """返回所有字段值"""
        return [getattr(self, field.name) for field in fields(self)]
    
    def items(self):
        """返回所有字段的键值对"""
        return [(field.name, getattr(self, field.name)) for field in fields(self)]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def validate(self):
        """验证配置，子类可以重写此方法"""
        pass
    
    def __post_init__(self):
        """dataclass 的后期初始化钩子"""
        self.validate()
    
    # 将方法添加到类中
    cls.__getitem__ = __getitem__
    cls.__setitem__ = __setitem__
    cls.__iter__ = __iter__
    cls.__len__ = __len__
    cls.__contains__ = __contains__
    cls.keys = keys
    cls.values = values
    cls.items = items
    cls.to_dict = to_dict
    cls.validate = validate
    cls.__post_init__ = __post_init__
    
    return cls

def algconfigclass(cls):
    cls = dataclass(cls)
    return cls
