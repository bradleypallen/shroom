# -*- coding: utf-8 -*-

import json
from string import Template
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class ShroomClassifier:
    """A classifier for the SHROOM validation and test datasets."""

    TASKS = {
        "DM": "The given task is Definition Modeling, meaning that the goal of the language model is to generate a definition for a specific term in the input text.",
        "PG": "The given task is Paraphrase Generation, meaning that the goal of the language model is to generate a paraphrase of the input text.",
        "MT": "The given task is Machine Translation, meaning that the goal of the language model is to generate a natural language translation of the input text.",
        "TS": "The given task is Text Simplification, meaning that the goal of the language model is to generate a simplified version of the input text.",
    }

    ROLES = {
        "MT": "You are a translator concerned that the generated text is a good and accurate translation of the input text.",
        "DM": "You are a lexicographer concerned that the generated text accurately captures the meaning of the term between the '<define>' and '</define>' delimiters in the input text.",
        "TS": "You are an editor concerned that the generated text is short, simple, and has the same meaning as the input text.",
        "PG": "You are an author concerned that the generated text is an accurate paraphrase that does not distort the meaning of the input text.",
    }

    DEFINITION = """A text contains a hallucination if and only if it contains any nonsensical or 
factually incorrect information, or contains any additional information that cannot be supported by either 
the input text or the target text.
"""

    PSEUDO_DEMO_TEMPLATE = Template("""##
Input text: $src
Target text: $tgt
Generated text: $hyp
Answer: $label""")

    PROMPT = """A language model has generated an output from a given input for a specific task.
{task}
{role}
You will be given three inputs: input text, target text, and generated text.
You are asked to evaluate the generated text looking at the input text and the target text. 
Then, you need to answer the question: is the generated text a hallucination or not? 
{definition}
Answer 'Hallucination' if the output is a hallucination, or 'Not Hallucination' if it is not a hallucination. 
Only answer 'Hallucination' or 'Not Hallucination'.
{examples}
##
Input text: {src}
Target text: {tgt}
Generated text: {hyp}
Answer:
"""

    def __init__(self, model_name="gpt-4", temperature=0.1, examples_per_class=1, example_selection=0):
        """
        Initializes a classifier for the SemEval 2024 Task 6.
        
        Parameters:
            model_name: The name of the model to be used for classification (default "gpt-4").
            temperature: The temperature parameter for the model (default 0.1).
         """
        self.model_name = model_name
        self.temperature = temperature
        self.llm = self._llm(model_name, temperature)
        self.examples = self._examples(examples_per_class, example_selection)
        self.chain = ChatPromptTemplate.from_template(self.PROMPT) | self.llm | StrOutputParser()

    def _llm(self, model_name, temperature):
        """
        Initializes a model for use in classification. 

        Parameters:
            model_name: The name of the model to be used for classification.
            temperature: The temperature parameter for the model.

        Returns:
            An LCEL model.
        """
        if model_name in [
            "gpt-4",
            "gpt-3.5-turbo",
            "gpt-4-1106-preview",
            "gpt-4-0125-preview",
            ]:
            return ChatOpenAI(model_name=model_name, temperature=temperature, request_timeout=100)
        else:
            raise Exception(f'Model {model_name} not supported')
        
    def _serialize_pseudo_demo(self, pseudo_demo):
        return self.PSEUDO_DEMO_TEMPLATE.substitute({
            "hyp": pseudo_demo["datapoint"]["hyp"],
            "src": pseudo_demo["datapoint"]["src"],        
            "tgt": pseudo_demo["datapoint"]["tgt"],        
            "label": pseudo_demo["classification"]["label"],    
        })
        
    def _examples(self, examples_per_class, example_selection):
        examples = json.load(open('examples.json', 'r'))
        prompts = { "TS": "" }
        for task in ["DM", "PG", "MT"]:
            prompts[task] = '\n'.join([ self._serialize_pseudo_demo(pd) for pd in examples[example_selection][task]["Hallucination"][:examples_per_class] ]) + \
                '\n' + \
                '\n'.join([ self._serialize_pseudo_demo(pd) for pd in examples[example_selection][task]["Not Hallucination"][:examples_per_class] ])
        return prompts
    
    def classify(self, datapoint, N=10, task_defined=True, role_defined=True, hallucination_defined=True, examples=True):
        """
        Determines whether or not the output (hyp) is a hallucination, and 
        produces an estimate of the probability that the output is a hallucination.
        
        Parameters:
            datapoint: A datapoint from the SHROOM dataset.
        
        Returns:
            A dict containing a classification of the output based on the task, input, output and target.
        """
        
        predictions = self.chain.batch([ 
            { 
                "task": self.TASKS[datapoint["task"]] if task_defined else "", 
                "role": self.ROLES[datapoint["task"]] if role_defined else "",
                "definition": self.DEFINITION if hallucination_defined else "", 
                "examples": self.examples[datapoint["task"]] if examples else "",
                "src": datapoint["src"], 
                "tgt": datapoint["tgt"], 
                "hyp": datapoint["hyp"],
            } for i in range(N) ]
        )
        w = 1./float(N)
        prob_hallucination = sum([ w for p in predictions if p == 'Hallucination' ])
        if prob_hallucination >= 0.5:
            label = "Hallucination"
        else:
            label = "Not Hallucination"
        if "id" in datapoint:
            return { "id": datapoint["id"], "label": label, "p(Hallucination)": prob_hallucination }
        else:
            return { "label": label, "p(Hallucination)": prob_hallucination }
