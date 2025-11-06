"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union, List


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("./", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
        src_index: bool = False,
        sample_input: Union[None, list, str] = None,
        sample_output: Union[None, list, str] = None
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if sample_input is not None and sample_output is not None:
            if isinstance(sample_input, str):
                res = self.template["prompt_1_shot"].format(
                    instruction=instruction,
                    sample_input=sample_input,
                    sample_output=sample_output,
                    input=input,
                )
            else:
                assert len(sample_input) == len(sample_output), "few-shot input and output have different length!"
                shot = len(sample_input)
                res = self.template[f"prompt_{shot}_shot"].format(
                    instruction=instruction,
                    sample_input=sample_input,
                    sample_output=sample_output,
                    input=input,
                )
        elif src_index and input:
            res = self.template["prompt_input_end_index"].format(
                instruction=instruction,
                input=input
            )
        elif src_index:
            res = self.template["prompt_input_start_index"].format(
                instruction=instruction
            )
        elif input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[-1].strip()
