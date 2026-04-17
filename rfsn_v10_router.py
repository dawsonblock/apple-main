from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import rfsn_v10_common as common


class AsyncHierarchicalRouterMLX:
    def __init__(self, config: common.RFSNConfig, disk_dir: Optional[Path] = None):
        self.config = config
        self.disk_dir = Path(config.disk_cache_dir) if disk_dir is None else Path(disk_dir)
        self.throttle = config.prefetch_throttle_s
        self._cache: Dict[int, Dict[str, Any]] = {}
        self._max_cache_size = 16
        self._pending_prefetch: set[int] = set()

    async def predict_and_prefetch(self, current_position: int, context_window: int, top_k: int = 2) -> List[int]:
        chunk_ids = self._candidate_chunk_ids(current_position, context_window)[:top_k]
        loaded: List[int] = []
        for chunk_id in chunk_ids:
            if chunk_id in self._cache:
                loaded.append(chunk_id)
                continue
            if chunk_id in self._pending_prefetch:
                continue
            self._pending_prefetch.add(chunk_id)
            try:
                await self._load_chunk(chunk_id)
                loaded.append(chunk_id)
            finally:
                self._pending_prefetch.discard(chunk_id)
            await asyncio.sleep(self.throttle)

        while len(self._cache) > self._max_cache_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        return loaded

    def _candidate_chunk_ids(self, current_position: int, context_window: int) -> List[int]:
        chunk_size = 4096
        start_chunk = max(0, (current_position - context_window) // chunk_size - 1)
        end_chunk = max(start_chunk, (current_position + context_window) // chunk_size + 1)
        ids: List[int] = []
        for chunk_id in range(start_chunk, end_chunk + 1):
            path = self.disk_dir / f"layer0_chunk{chunk_id}.npz"
            if path.exists():
                ids.append(chunk_id)
        ids.sort(key=lambda chunk_id: abs(chunk_id * chunk_size - current_position))
        return ids

    async def _load_chunk(self, chunk_id: int) -> None:
        loop = asyncio.get_event_loop()
        self._cache[chunk_id] = await loop.run_in_executor(None, self._load_chunk_sync, chunk_id)

    def _load_chunk_sync(self, chunk_id: int) -> Dict[str, Any]:
        path = self.disk_dir / f"layer0_chunk{chunk_id}.npz"
        if not path.exists():
            return {}
        data = np.load(path)
        return {key: common._np_to_mx(np.array(value)) for key, value in data.items()}