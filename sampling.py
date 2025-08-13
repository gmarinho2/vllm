# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm import LLM, EngineArgs, SamplingParams
from vllm.outputs import RequestOutput
from vllm.v1.engine.llm_engine import LLMEngine as V1LLMEngine


@pytest.fixture
def model():
    return LLM(model="ibm-granite/granite-3.2-8b-instruct", max_model_len=1024)


@pytest.fixture
def prompt():
    return "Tell me how to go from Paris to London."


@pytest.fixture
def expected_output(model: LLM, prompt: str,
                    sampling_params_exemples: list[SamplingParams]):
    return [model.generate(prompt, x) for x in sampling_params_exemples]


@pytest.fixture
def sampling_params_exemples():
    return [
        SamplingParams(temperature=0.8, seed=8780, n=2),  #n
        SamplingParams(temperature=1),  #temperature
        SamplingParams(temperature=0, max_tokens=100,
                       min_tokens=20),  #min_tokens
        SamplingParams(temperature=0.8, max_tokens=100),  #seed
        SamplingParams(temperature=0.8, max_tokens=100, seed=8780,
                       min_p=1),  #min_p,
        SamplingParams(temperature=0, logit_bias={26562: 100}),  #logit_bias
        SamplingParams(temperature=0, max_tokens=100,
                       ignore_eos=False),  #ignore_eos
        SamplingParams(temperature=0,
                       seed=8780,
                       bad_words=[
                           "Book", "flight", "Several", "airlines", "offer",
                           "direct", "flights"
                       ]),  #bad_words
    ]


def test_batch_size_1_sampling_params(
        model: LLM, prompt: str, expected_output: list[list[RequestOutput]]):
    sampling_params = [
        SamplingParams(temperature=0.8, seed=8780),  #n
        SamplingParams(temperature=0),  #temperature
        SamplingParams(temperature=0, max_tokens=100,
                       min_tokens=0),  #min_tokens
        SamplingParams(temperature=0.8, max_tokens=100, seed=8780),  #seed
        SamplingParams(temperature=0.8, max_tokens=100, seed=8780,
                       min_p=0.1),  #min_p,
        SamplingParams(temperature=0, logit_bias={26562: -100}),  #logit_bias
        SamplingParams(temperature=0, max_tokens=100,
                       ignore_eos=True),  #ignore_eos
        SamplingParams(temperature=0, seed=8780),  #bad_words
    ]

    responses = [model.generate(prompt, x) for x in sampling_params]

    # need to decide what is going to be compared since
    # reponse.text do not work for every case
    for i in range(len(sampling_params)):
        assert expected_output[i] != responses[i]


def test_changing_batch_size_sampling_params(
        model: str, backend: str, monkeypatch: pytest.MonkeyPatch, prompt: str,
        sampling_params_exemples: list[SamplingParams]):

    with monkeypatch.context() as m:
        # set env vars
        # m.setenv("VLLM_SPYRE_USE_CB", "1")
        # m.setenv("VLLM_USE_V1", "1")
        m.setenv("VLLM_SPYRE_DYNAMO_BACKEND", backend)
        # To get deterministic execution in V1
        # and to enable InprocClient
        m.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

        # start the engine
        engine_args = EngineArgs(model=model,
                                 max_model_len=1024,
                                 max_num_seqs=8)

        engine = V1LLMEngine.from_engine_args(engine_args)
        engine_core = engine.engine_core.engine_core

        requests = engine_core.model_executor.driver_worker.\
                                        worker.model_runner.req_ids2blocks

        assert len(requests) == 0

        engine.add_request("1", prompt, sampling_params_exemples[0])
        engine.add_request("2", prompt, sampling_params_exemples[1])
        engine.add_request("3", prompt, sampling_params_exemples[2])
        engine.add_request("4", prompt, sampling_params_exemples[3])
        engine.step()
        engine.step()
        engine.step()
        engine.add_request("5", prompt, sampling_params_exemples[4])
        engine.add_request("6", prompt, sampling_params_exemples[5])
        engine.add_request("7", prompt, sampling_params_exemples[6])
        engine.add_request("8", prompt, sampling_params_exemples[7])

        requests = engine_core.model_executor.driver_worker.\
                                            worker.model_runner.req_ids2blocks

        assert requests[0].response == expected_output[0]
        assert requests[1].response == expected_output[1]
        assert requests[2].response == expected_output[2]
        assert requests[3].response == expected_output[3]
        assert requests[4].response == expected_output[4]
        assert requests[5].response == expected_output[5]
        assert requests[6].response == expected_output[6]
        assert requests[7].response == expected_output[7]
