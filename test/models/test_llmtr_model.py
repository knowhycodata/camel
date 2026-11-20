# ========= Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. =========
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
# ========= Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. =========

import pytest

from camel.configs import LLMTRConfig
from camel.models import LLMTRModel
from camel.types import ModelType
from camel.utils import OpenAITokenCounter


@pytest.mark.model_backend
def test_llmtr_model():
    model = LLMTRModel(
        ModelType.LLMTR_MINIMAX_M3_1_FLASH, api_key="test-llmtr-key"
    )
    assert model.model_type == ModelType.LLMTR_MINIMAX_M3_1_FLASH
    assert model.model_config_dict == LLMTRConfig().as_dict()
    assert model._url.rstrip("/") == "https://llmtr.com/v1"
    assert isinstance(model.token_counter, OpenAITokenCounter)
    assert isinstance(model.model_type.value_for_tiktoken, str)
    assert isinstance(model.model_type.token_limit, int)
    assert model.model_type.is_llmtr


@pytest.mark.model_backend
def test_llmtr_model_stream_property():
    model = LLMTRModel(
        ModelType.LLMTR_MINIMAX_M3_1_FLASH, api_key="test-llmtr-key"
    )
    assert model.stream is False


@pytest.mark.model_backend
def test_llmtr_model_accepts_string_model_id():
    model = LLMTRModel("minimax/minimax-m3.1-flash", api_key="test-llmtr-key")
    assert str(model.model_type) == "minimax/minimax-m3.1-flash"
    assert model._client.base_url is not None
    assert str(model._client.base_url).rstrip("/") == "https://llmtr.com/v1"
