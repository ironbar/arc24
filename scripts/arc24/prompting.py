from abc import ABC, abstractmethod
from jinja2 import Template
from termcolor import colored

from .encoders import GridEncoder

system_prompt = """You are a helpful AI assistant. Your job is to solve tasks from the Abstraction and Reasoning Challenge (ARC). 
The user will present you with sample input and output grids for each task. 
Your job will be to understand the transformation between the input and the output and apply it to the last input grid given by the user. 
The puzzle-like inputs and outputs present a grid where each square can be one of ten colors. A grid can be any height or width between 1x1 and 30x30.
The background of the grid is typically colored with 0.
The tasks from ARC are based on the following priors:

- Objectness: Objects persist and cannot appear or disappear without reason. Objects can interact or not depending on the circumstances.
- Goal-directed: Objects can be animate or inanimate. Some objects are "agents" - they have intentions and they pursue goals.
- Numbers & counting: Objects can be counted or sorted by their shape, appearance, or movement using basic mathematics like addition, subtraction, and comparison.
- Basic geometry & topology: Objects can be shapes like rectangles, triangles, and circles which can be mirrored, rotated, translated, deformed, combined, repeated, etc. Differences in distances can be detected.

The transformations between input and output should be based on these priors.
"""

prompt_template = Template("""Let's see if you can solve this simple ARC task. These are some input-output grid examples that define the task.
{% for sample in train_samples %}
## Example {{ loop.index }}

### Input

{{ sample.input }}

### Output

{{ sample.output }}
{% endfor %}
## Test case

### Input

{{ test_input }}
""")

answer_template = Template("""### Output

{{ test_output }}""")

class PromptCreator(ABC):
    def __init__(self, grid_encoder: GridEncoder):
        self.grid_encoder = grid_encoder

    @abstractmethod
    def create_task_prompts(self, task):
        pass

    @abstractmethod
    def parse_response(self, text):
        pass

class SimplePromptCreator(PromptCreator):
    def __init__(self, grid_encoder, tokenizer, model_path):
        super().__init__(grid_encoder)
        self.tokenizer = tokenizer
        self.model_path = model_path

    def create_task_prompts(self, task):
        train_samples = [{key: self.grid_encoder.to_text(grid) for key, grid in sample.items()} for sample in task['train']]
        prompts = []
        for test_sample in task['test']:
            user_message = prompt_template.render(train_samples=train_samples,
                                                  test_input=self.grid_encoder.to_text(test_sample['input']))
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": """### Output\n```grid\n"""}]
            # TODO: add start of assistant reply
            prompt = self.tokenizer.apply_chat_template(messages,
                                                   tokenize=False,
                                                   add_generation_prompt=False)
            prompts.append(remove_assistant_ending(prompt, self.model_path))
        return prompts

    def parse_response(self, text):
        return self.grid_encoder.to_grid('```grid\n' + text)


def remove_assistant_ending(text, model_path):
    """
phi-3

<|assistant|>
### Output
```grid
<|end|>
<|endoftext|>

llama 3.1

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

### Output
```grid<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    # TODO: better way to solve this, model_path could be not informative
    if 'llama' in model_path.lower():
        split_text = '<|eot_id|>'
    elif 'qwen' in model_path.lower():
        split_text = '<|im_end|>'
    else:
        split_text = '<|end|>'
    return split_text.join(text.split(split_text)[:-1])


def print_smaller_prompt(prompts):
    smaller_prompt = sorted(prompts, key=lambda x: len(x))[0]
    print('\n\nSmaller prompt:')
    pretty_print_prompt(smaller_prompt)
    print('\n\n')


def print_sample_prompt(data, prompt_creator):
    prompts = [prompt_creator.create_task_prompts(task)[0] for task in data.values()]
    prompts = sorted(prompts, key=lambda x: len(x))
    pretty_print_prompt(prompts[0])


def pretty_print_prompt(text, default_color='black'):
    color = default_color
    attrs = None
    print('-'*80)
    for line in text.splitlines():
        if line.startswith('<|assistant|>'):
            color = 'blue'
        elif line.startswith('<|user|>'):
            color = default_color
        elif line.startswith('<|system|>'):
            color = 'green'
        if line.startswith('<'):
            attrs = ['bold']
        else:
            attrs = None
        print(colored(line, color, attrs=attrs))
    print('-'*80)