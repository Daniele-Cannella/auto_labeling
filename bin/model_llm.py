'''
Language Model

TODO:
- Dataset : json dump, json load
- Image
- Alias
- Model-vis
- Model-lab
- Metrics
- Pipeline

Author: Mattia Braga
'''

'''
Text example:
I have to find the best alias for this word: '{class_name}'.
Could you give me one?

An example of the word {class_name}:
    -{example[0]}
    -{example[1]}

'''

import onnxruntime_genai as og
import os
import json
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


LABEL_PATTERN = re.compile(r'".*?"') # Pattern to find the labels in the text


class LLM:
    """
    Class to handle the Language Model.
    """

    def __init__(self, model_path: str, search_options: dict) -> None:
        """
        Initialize the Language Model.

        Args:
            model_path (str): path to the model.
            search_options (dict): search options for the model.
        """
        self.model = og.Model(model_path)  # "./models/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"
        self.search_options = search_options
        self.params = og.GeneratorParams(self.model)

    @staticmethod
    def model_path() -> str:
        """
        Get the path to the model.

        Returns:
            str: path to the model.
        """
        return "./models/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"

    @staticmethod
    def download_model() -> str:
        """
        Download the model if it is not found.

        Returns:
            str: path to the model.
        """
        if not os.path.exists("./models"):
            os.mkdir("./models")
        os.system("huggingface-cli download microsoft/Phi-3-mini-4k-instruct-onnx --include cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/* --local-dir ./models")
        return "./models/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"

    def generate_phi(self, text: str) -> str:
        """
        Generate text from the input text.

        Args:
            text (str): input text.

        Returns:
            str: filtered generated text.
        """
        tokenizer = og.Tokenizer(self.model)
        tokenizer_stream = tokenizer.create_stream()
        chat_template = '<|user|>\n{input} <|end|>\n<|assistant|>'
        prompt = f'{chat_template.format(input=text)}'
        input_tokens = tokenizer.encode(prompt)
        print("Len input tokens:", len(input_tokens))
        params = og.GeneratorParams(self.model)
        params.set_search_options(**self.search_options)
        params.input_ids = input_tokens
        generator = og.Generator(self.model, params)
        new_tokens = []
        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()
            new_token = generator.get_next_tokens()[0]
            new_tokens.append(new_token)
        new_text = [tokenizer_stream.decode(new_token) for new_token in new_tokens]
        new_text = ''.join(new_text)
        return new_text

    def process_history(self, history: list) -> str:
        """
        Process the history to keep the last 5 inputs and combine them.

        Args:
            history (list): history of the inputs.

        Returns:
            str: processed history.
        """
        if history:
            history_texts = [f"<|user|>\n{entry['input']} <|end|>\n<|assistant|>{entry['output']}" for entry in history[-2:]]
            return "\n".join(history_texts)
        return ""

    def process_input(self, text: str, history: list, old_labels: list) -> str:
        """
        Process the input text with history.

        Args:
            text (str): input text.
            history (list): elaborated history of the inputs.
            old_labels (list): old labels.

        Returns:
            str: processed input text, combination of the history/ old_labels and the request from the user.
        """
        history_text = self.process_history(history)
        if history_text:
            # return f"{history_text}\n<|user|>: {text} <|end|>\n<|assistant|>"
            return f"<|user|>: {text}. I've already used these {old_labels}, so tell me something different <|end|>\n<|assistant|>"
        return text

    @staticmethod
    def write_history(input_text: str, output_text: str, output_file: str, history_: list[dict]) -> None:
        """
        Write the history to the output file.

        Args:
            input_text (str): input text, text fed to the model.
            output_text (str): output text, generated text.
            output_file (str): output file path.
            history_ (list[dict]): history of the inputs and outputs.
        """
        with open(output_file, 'w') as f:
            history_.append({"input": input_text, "output": output_text})
            json.dump(history_, f, indent=4)

    @staticmethod
    def load_history(input_file: str) -> list:
        """
        Load the history from the input file.

        Args:
            input_file (str): input file path.

        Returns:
            list: history of the inputs and outputs.
        """
        try:
            with open(input_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
        
    @staticmethod
    def get_old_labels(text: list[dict]) -> list:
        """
        Get the old labels from the text.

        Args:
            text (list[dict]): list of dictionaries with the input and output.

        Returns:
            list: list of old labels, cleaned and lowercased.
        """
        old_labels = []
        for diz in text:
            labels = LABEL_PATTERN.findall(diz['output'])
            labels = [label.lower().strip(".") for label in labels]
            old_labels.extend(labels)

        if old_labels:
            return old_labels
        else:
            raise ValueError("No old labels found in the history.")
        
def generate_groq(text: str) -> str:
    try:
        chat = ChatGroq(temperature=1, model="llama3-70b-8192", api_key="")
        prompt = ChatPromptTemplate.from_messages([("human", "{topic}")])
        chain = prompt | chat
        aliases = []
        for chunk in chain.stream({"topic": text}):
            aliases.append(chunk.content)
        aliases = ''.join(aliases)
        print(aliases)
        model_aliases = find_alias(aliases)
        if model_aliases:
            return model_aliases
    except Exception as e:
        print(f"Error generating aliases using model: {e}")

def find_alias(aliases):
    elab_aliases = []
    aliases = aliases.split("\n")[1:-1]
    for alias in aliases:
        if "**" in alias:
            alias = alias.split("**")[1]
            elab_aliases.append(alias.lower() + " .")
        else:
            alias = alias.split(". ")
            print(alias)
            if len(alias) > 1:
                elab_aliases.append(alias[1].lower() + " .")
    return elab_aliases


def test():
    options = {
        "max_length": 2048,
        "min_length": 1,
        "do_sample": True,
        "top_p": 0.9,
        "top_k": 0,
        "temperature": 0.7,
        "repetition_penalty": 1.0,
    }
    model_path = LLM.model_path()
    try:
        model = LLM(model_path, options)
    except Exception as e:
        print("Model not found, downloading ...")
        try:
            model_path = LLM.download_model()
        except Exception as e:
            print("Error downloading model:", e)
            return
        print("Model downloaded")
        model = LLM(model_path, options)

    text = '''
I have to find the best alias for this word: 'box'.
Could you give me only one alias?

You already gave me the following aliases: 'cube', 'crate', 'chest'

An example of the word {class_name}:
    -package
    -casket
'''
    new_text = model.generate(text)
    print(new_text)


def generate_text(class_name: str, groq: bool) -> str:
    """
    Generate text for the given class name.

    Args:
        class_name (str): class name ("box", ...).

    Returns:
        str: label generated by the model.
    """
    if not groq:

        options = {
            "max_length": 2048,
            "min_length": 1,
            "do_sample": False,
            "top_p": 0.9,
            "top_k": 0,
            "temperature": 0.2,
            "repetition_penalty": 1.0,
        }
        model_path = LLM.model_path()
        try:
            model = LLM(model_path, options)
        except Exception as e:
            print("Model not found, downloading ...")
            try:
                model_path = LLM.download_model()
            except Exception as e:
                print("Error downloading model:", e)
                return
            print("Model downloaded")
            model = LLM(model_path, options)

        if not os.path.exists("../data/json"):
            os.mkdir("../data/json")

        history_file = f"../data/json/history_{class_name}.json"
        history_ = LLM.load_history(history_file)

        text = f"I have to find the best alias for this word: '{class_name}'. Could you give me a new alias?"

        try:
            old_labels = LLM.get_old_labels(history_)
            old_labels = list(set(old_labels))
        except Exception as e:
            with open("../data/json/examples.txt", "r") as f:
                examples = f.readlines()
            old_labels = [example.strip().split(",")[1] for example in examples]

        processed_text = model.process_input(text, history_, old_labels)

        new_text = model.generate_phi(processed_text)

        LLM.write_history(text, new_text, history_file, history_)

        return LABEL_PATTERN.findall(new_text.split("\n")[0])[0]

    else:
        text = '''I have to find the best alias for this word: 'box'.
Could you give me only one alias?

You already gave me the following aliases: 'cube', 'crate', 'chest'

An example of the word 'box':
    -package
    -casket
'''
        new_text = generate_groq(text)
        print(new_text)


if __name__ == "__main__":
    label = generate_text("box", True)
    print(label)
