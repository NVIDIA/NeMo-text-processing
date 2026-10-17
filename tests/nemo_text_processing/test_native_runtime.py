# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Expose native runtime regressions through the repository's pytest entry point."""

import pytest

from tools.text_processing_deployment.native_runtime.test_byte_transducer import ByteTransducerTest
from tools.text_processing_deployment.native_runtime.test_text_processing import TextProcessingTest

pytestmark = pytest.mark.run_only_on("CPU")

__all__ = ["ByteTransducerTest", "TextProcessingTest"]
