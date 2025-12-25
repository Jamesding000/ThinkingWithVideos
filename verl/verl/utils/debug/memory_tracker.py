# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import os
import sys
import psutil
import torch
from typing import Dict, Optional


class MemoryTracker:
    """Track CPU and GPU memory usage to detect leaks."""

    def __init__(self, name: str = "MemoryTracker", enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self.process = psutil.Process(os.getpid())
        self.checkpoints: Dict[str, Dict] = {}
        self.last_checkpoint_name = None

    def checkpoint(self, name: str, force_gc: bool = False):
        """Create a memory checkpoint."""
        if not self.enabled:
            return

        if force_gc:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Get CPU memory
        mem_info = self.process.memory_info()
        cpu_rss_gb = mem_info.rss / (1024**3)
        cpu_vms_gb = mem_info.vms / (1024**3)

        # Get GPU memory
        gpu_allocated_gb = 0
        gpu_reserved_gb = 0
        if torch.cuda.is_available():
            gpu_allocated_gb = torch.cuda.memory_allocated() / (1024**3)
            gpu_reserved_gb = torch.cuda.memory_reserved() / (1024**3)

        # Get garbage collection stats
        gc_counts = gc.get_count()

        # Get Python object counts
        obj_count = len(gc.get_objects())

        checkpoint_data = {
            "cpu_rss_gb": cpu_rss_gb,
            "cpu_vms_gb": cpu_vms_gb,
            "gpu_allocated_gb": gpu_allocated_gb,
            "gpu_reserved_gb": gpu_reserved_gb,
            "gc_counts": gc_counts,
            "obj_count": obj_count,
        }

        self.checkpoints[name] = checkpoint_data

        # Print diff if we have a previous checkpoint
        if self.last_checkpoint_name:
            self._print_diff(self.last_checkpoint_name, name)
        else:
            self._print_checkpoint(name)

        self.last_checkpoint_name = name

    def _print_checkpoint(self, name: str):
        """Print a single checkpoint."""
        data = self.checkpoints[name]
        msg = (
            f"🔍 [{self.name}] {name:30s} | "
            f"CPU RSS: {data['cpu_rss_gb']:6.3f} GB | "
            f"GPU: {data['gpu_allocated_gb']:6.3f}/{data['gpu_reserved_gb']:6.3f} GB | "
            f"PyObjs: {data['obj_count']:8d} | "
            f"GC: {data['gc_counts']}"
        )
        print(msg, flush=True)

    def _print_diff(self, prev_name: str, curr_name: str):
        """Print diff between two checkpoints."""
        prev = self.checkpoints[prev_name]
        curr = self.checkpoints[curr_name]

        cpu_diff = curr["cpu_rss_gb"] - prev["cpu_rss_gb"]
        gpu_alloc_diff = curr["gpu_allocated_gb"] - prev["gpu_allocated_gb"]
        gpu_reserved_diff = curr["gpu_reserved_gb"] - prev["gpu_reserved_gb"]
        obj_diff = curr["obj_count"] - prev["obj_count"]

        # Format with + or - sign and color
        cpu_sign = "+" if cpu_diff >= 0 else ""
        gpu_sign = "+" if gpu_alloc_diff >= 0 else ""
        obj_sign = "+" if obj_diff >= 0 else ""

        # Mark potential leaks with ⚠️
        leak_marker = "⚠️ " if cpu_diff > 0.1 or obj_diff > 1000 else "  "

        msg = (
            f"{leak_marker}🔍 [{self.name}] {curr_name:30s} | "
            f"CPU RSS: {curr['cpu_rss_gb']:6.3f} GB ({cpu_sign}{cpu_diff:+7.3f}) | "
            f"GPU: {curr['gpu_allocated_gb']:6.3f}/{curr['gpu_reserved_gb']:6.3f} GB ({gpu_sign}{gpu_alloc_diff:+7.3f}/{gpu_sign}{gpu_reserved_diff:+7.3f}) | "
            f"PyObjs: {curr['obj_count']:8d} ({obj_sign}{obj_diff:+8d}) | "
            f"GC: {curr['gc_counts']}"
        )
        print(msg, flush=True)

    def analyze_top_objects(self, top_n: int = 10):
        """Analyze top objects in memory (expensive operation)."""
        if not self.enabled:
            return

        print(f"\n🔍 [{self.name}] Analyzing top {top_n} objects by count:", flush=True)

        # Count objects by type
        type_counts = {}
        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            type_counts[obj_type] = type_counts.get(obj_type, 0) + 1

        # Sort by count
        sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)

        for i, (obj_type, count) in enumerate(sorted_types[:top_n]):
            print(f"  {i+1}. {obj_type:30s}: {count:8d}", flush=True)

    def summary(self):
        """Print summary of all checkpoints."""
        if not self.enabled or not self.checkpoints:
            return

        print(f"\n{'='*100}")
        print(f"🔍 [{self.name}] Memory Tracker Summary")
        print(f"{'='*100}")

        checkpoint_names = list(self.checkpoints.keys())
        if len(checkpoint_names) < 2:
            print("  Not enough checkpoints for analysis")
            return

        first_name = checkpoint_names[0]
        last_name = checkpoint_names[-1]

        first = self.checkpoints[first_name]
        last = self.checkpoints[last_name]

        total_cpu_leak = last["cpu_rss_gb"] - first["cpu_rss_gb"]
        total_gpu_leak = last["gpu_allocated_gb"] - first["gpu_allocated_gb"]
        total_obj_leak = last["obj_count"] - first["obj_count"]

        print(
            f"  Total CPU RSS leak: {total_cpu_leak:+.3f} GB ({first_name} -> {last_name})"
        )
        print(
            f"  Total GPU leak: {total_gpu_leak:+.3f} GB ({first_name} -> {last_name})"
        )
        print(f"  Total Python object leak: {total_obj_leak:+d} objects")

        # Find biggest leak between consecutive checkpoints
        max_leak = 0
        max_leak_pair = None
        for i in range(len(checkpoint_names) - 1):
            prev_name = checkpoint_names[i]
            curr_name = checkpoint_names[i + 1]
            leak = (
                self.checkpoints[curr_name]["cpu_rss_gb"]
                - self.checkpoints[prev_name]["cpu_rss_gb"]
            )
            if leak > max_leak:
                max_leak = leak
                max_leak_pair = (prev_name, curr_name)

        if max_leak_pair:
            print(
                f"  Biggest CPU leak: {max_leak:+.3f} GB between '{max_leak_pair[0]}' and '{max_leak_pair[1]}'"
            )

        print(f"{'='*100}\n")


# Global tracker instance
_global_tracker: Optional[MemoryTracker] = None


def get_memory_tracker() -> MemoryTracker:
    """Get or create the global memory tracker."""
    global _global_tracker
    if _global_tracker is None:
        enabled = os.getenv("VERL_MEMORY_DEBUG", "0") == "1"
        _global_tracker = MemoryTracker(name="Global", enabled=enabled)
    return _global_tracker


def checkpoint_memory(name: str, force_gc: bool = False):
    """Convenient function to checkpoint memory."""
    tracker = get_memory_tracker()
    tracker.checkpoint(name, force_gc=force_gc)
