# Copyright 2024 Bloomberg Finance L.P.
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
#
# SPDX-License-Identifier: Apache-2.0

"""
Utility functions to manage multi-device training
"""

import inspect
import logging
import os
import platform
import subprocess
import sys

# import pkg_resources
import importlib.metadata
import torch.cuda
import torch.distributed
from loguru import logger


def world_size() -> int:
    """Returns the total number of processes in a distributed job (num_nodes x gpus_per_node).
    Returns 1 in a non-distributed job.
    """
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_distributed() -> bool:
    """Returns True iff this is a distributed job (more than one process)."""
    return world_size() > 1


def local_rank() -> int:
    """Returns the local rank of the current process in a distributed job.
    Returns 0 (local primary) for non-distributed jobs.
    """
    return int(os.environ.get("LOCAL_RANK", "0"))


def global_rank() -> int:
    """Returns the global rank of the current process in a distributed job.
    Returns 0 (global primary) for non-distributed jobs.
    """
    return int(os.environ.get("RANK", local_rank()))


def barrier():
    """Synchronizes all processes. Set GPU with local_rank to perform barrier used by this process."""
    torch.distributed.barrier(device_ids=[local_rank()])


def patch_module_loggers(module_namespace):
    modules = inspect.getmembers(module_namespace, predicate=inspect.ismodule)

    # check toplevel module
    if hasattr(module_namespace, "logger"):
        module_namespace.logger = logger
        logger.info(f"Patching logger: {module_namespace.__name__}")

    for _, mod in modules:
        if hasattr(mod, "logger"):
            mod.logger = logger
            logger.info(f"Patching logger: {mod.__name__}")


class InterceptLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists.
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def configure_distributed_logging(
    rank_zero_level="INFO",
    rank_non_zero_level="WARNING",
):
    """This method configures logger to reduce noise in multi-node, multi-process evaluations (e.g. DeepSpeed)_summary_
    Args:
        rank_zero_level (str, optional): Log level on zero rank process. Defaults to "INFO".
        rank_non_zero_level (str, optional): Log level on non-zero rank processes. Defaults to "WARNING".
    """

    logger.remove()
    rank = local_rank()
    format = "<g>{time:YYYY-MM-DD HH:mm:ss.SSS}</g> | <lvl>{level}</lvl> | rank={extra[rank]} | <c>{name}</c>:<c>{function}</c>:<c>{line}</c> - <lvl>{message}</lvl>\n{exception}"
    if rank != 0:
        logger.configure(
            extra={"rank": f"{global_rank()}:{rank}"},
            handlers=[
                {"sink": sys.stderr, "format": format, "level": rank_non_zero_level}
            ],
        )
    else:
        logger.configure(
            extra={"rank": f"{global_rank()}:{rank}"},
            handlers=[{"sink": sys.stdout, "format": format, "level": rank_zero_level}],
        )

    # Attempt to intercept normal logging in libs
    logging.basicConfig(handlers=[InterceptLogHandler()], level=0, force=True)


def get_cuda_version():
    """Get the installed CUDA version."""
    try:
        output = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        version_line = output.strip().split("\n")[-1]
        version = version_line.split(" ")[-1]
        return version
    except Exception as e:
        logger.info(f"Cannot detect CUDA version. Exception occured: {e}")
        return "N/A"


def log_runtime_info():
    # Get Python runtime information
    python_version = sys.version
    python_implementation = platform.python_implementation()
    python_build = platform.python_build()

    # Get environment variables
    env_vars = os.environ

    # Get installed package versions
    installed_packages = [
    #     (d.project_name, d.version) for d in pkg_resources.working_set
         (d.metadata["Name"], d.version) for d in importlib.metadata.distributions()
    ]


    # logger.info diagnostics
    logger.info("Python Version: {}".format(python_version))
    logger.info("Python Implementation: {}".format(python_implementation))
    logger.info("Python Build: {}".format(python_build))

    logger.info(f"Environment Variables: {env_vars}")
    logger.info(f"Installed Packages: {installed_packages}")

    logger.info(f"CUDA version: {get_cuda_version()}")
    logger.info(f"Is CUDA available for Torch?: {torch.cuda.is_available()}")

    logger.info(f"World size: {world_size()}")


def local_rank_zero(func):
    """
    Decorator to execute function only in local zero rank. Can be useful for logging statistics.
    """

    def wrapper(*args, **kwargs):
        if local_rank() == 0:
            func(*args, **kwargs)

    return wrapper


def global_rank_zero(func):
    """
    Decorator to execute function only in global zero rank. Can be useful for logging statistics.
    """

    def wrapper(*args, **kwargs):
        if global_rank() == 0:
            func(*args, **kwargs)

    return wrapper



import gpustat
def print_gpu_stats():
    gpustat.cli.main([])

def supports_flash_attention(device_id=0):
    """Check if a GPU supports FlashAttention."""
    major, minor = torch.cuda.get_device_capability(device_id)
    
    # Check if the GPU architecture is Ampere (SM 8.x) or newer (SM 9.0)
    is_sm8x = major == 8 and minor >= 0
    is_sm90 = major == 9 and minor == 0

    return is_sm8x or is_sm90