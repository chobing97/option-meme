"""DataFeed abstract base class."""

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class DataFeed(ABC):
    """Abstract data feed for real-time bar polling."""

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def get_latest_bar(self) -> Optional[pd.Series]: ...

    @abstractmethod
    def get_history(self, n_days: int = 5) -> pd.DataFrame: ...

    @abstractmethod
    def is_session_active(self) -> bool: ...

    @abstractmethod
    def disconnect(self) -> None: ...
