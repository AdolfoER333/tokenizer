import os
import re
import json
import tokenizer.utils.token_utils as tu

NUM_OF_MERGES = 1000


class TokenizerTrainer:
    """
    Prepares the inputs for the actual tokenizer class by identifying merges to be made from a training text dataset.

    """
    def __init__(self, training_text: str = None):
        self.training_text = training_text or self._load_standard_text()
        # self.insert_special_tokens()
        self.number_of_merges = NUM_OF_MERGES
        self.merges = None

    @staticmethod
    def _load_standard_text() -> str:
        """
        If not specified, training will occur with the sample_text loaded in this same repository under
        tokenizer.data.training_test.txt

        Returns:
            data: sample text as a single string
        """
        file_path = os.path.abspath(os.path.join(
            __file__,
            '..', '..',
            'data',
        ))
        with open(f'{file_path}/training_text.txt', 'r') as file:
            data = file.read()
        data = re.sub(r"\s*\r?\n+", ' ', data)
        data = re.sub(r'\s+]', ' ', data)
        data = data.replace('-', '')
        return data

    def train(self) -> None:
        """
        Uses the functions built on utils to identify and merge tokens using the training data. Number of merges is
        defined by the constant in this same script. Doesn't have a return value, but saves the resulting merges on a
        json file.

        Returns:
            None
        """
        self.merges = tu.text_to_merges(self.training_text, self.number_of_merges + 256)

        file_path = os.path.abspath(os.path.join(
            __file__,
            '..', '..',
            'training_outputs',
        ))
        with open(f'{file_path}/merges.json', 'w') as json_file:
            json.dump(self.merges, json_file)
