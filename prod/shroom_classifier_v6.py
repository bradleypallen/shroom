# -*- coding: utf-8 -*-

from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class ShroomClassifier:
    """A classifier for the SHROOM validation and test datasets."""

    PERSONA = {
        "MT": "a translator concerned that the output is a good and accurate translation",
        "DM": "a lexicographer concerned that the output accurately captures the meaning of the term between the '<define>' and '</define>' delimiters in the input",
        "TS": "an editor concerned that the output is short and simple",
        "PG": "an author concerned that the output is an accurate paraphrase that does not distort the meaning of the input",
    }

    TASK = {
        "DM": "The given task is Definition Modeling, meaning that the goal of the language model is to generate a definition for a specific term in the input.",
        "PG": "The given task is Paraphrase Generation, meaning that the goal of the language model is to generate a paraphrase of the input.",
        "MT": "The given task is Machine Translation, meaning that the goal of the language model is to generate a natural language translation of the input.",
        "TS": "The given task is Text Simplification, meaning that the goal of the language model is to generate a simplified version of the input.",
    }

    REFERENCE = {
        "src": "the input",
        "tgt": "the target",
        "either": "either the input or the target",
    }

    ANSWER_GENERATION_PROMPT = """A language model has generated an output from a given input for a specific task.
{task} You are {persona}. You will be given three inputs: input text, target text, and generated text.
You are asked to evaluate the generated text looking at the {ref} text. 
Then, you need to decide whether the generated text is a hallucination or not.
There are two criteria for hallucination:
- If the generated text contains any nonsensical or factually incorrect information, it is a hallucination.
- If the generated text contains additional information that cannot be supported by the input text or the target text, it is a hallucination.
- Else, the generated text is not a hallucination.
Now, it is time to look at the inputs.
Input text: {src}
Target text: {tgt}
Generated text: {hyp}
Is the generated text a hallucination? Answer 'Hallucination' if the output is a hallucination, or 'Not Hallucination' 
if it is not a hallucination. Only answer 'Hallucination' or 'Not Hallucination'.
Answer:
"""

    def __init__(self, model_name="gpt-4", temperature=0.1):
        """
        Initializes a classifier for the SemEval 2024 Task 6.
        
        Parameters:
            model_name: The name of the model to be used for classification (default "gpt-4").
            temperature: The temperature parameter for the model (default 0.1).
         """
        self.model_name = model_name
        self.temperature = temperature
        self.llm = self._llm(model_name, temperature)
        self.chain = self._chain()

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
            ]:
            return ChatOpenAI(model_name=model_name, temperature=temperature, request_timeout=100)
        else:
            raise Exception(f'Model {model_name} not supported')

    def _chain(self):
        """
        Creates an LCEL chain. 

        Returns:
            An LCEL chain.
        """
        return (
            ChatPromptTemplate.from_template(self.ANSWER_GENERATION_PROMPT) 
            | self.llm 
            | StrOutputParser()
        )
    
    def classify(self, datapoint):
        """
        Determines whether or not the output (hyp) is a hallucination, and 
        produces an estimate of the probability thatthe output is a hallucination.
        
        Parameters:
            datapoint: A datapoint from the SHROOM dataset.
        
        Returns:
            A dict containing a classification of the output based on the task, input, output and target.
        """
        predictions = self.chain.batch([ 
            { 
                "task": self.TASK[datapoint["task"]], 
                "persona": self.PERSONA[datapoint["task"]], 
                "src": datapoint["src"], 
                "tgt": datapoint["tgt"], 
                "hyp": datapoint["hyp"],
                "ref": datapoint["ref"],
            } for i in range(10) ])
        weight = 1./float(len(predictions))
        predicted_p = float(sum([ weight for prediction in predictions if prediction == 'Hallucination' ]))
        if predicted_p > 0.5:
            predicted = "Hallucination"
        else:
            predicted = "Not Hallucination"
        if "id" in datapoint:
            return { "id": datapoint["id"], "label": predicted, "p(Hallucination)": predicted_p }
        else:
            return { "label": predicted, "p(Hallucination)": predicted_p }
