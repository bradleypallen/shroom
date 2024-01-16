# -*- coding: utf-8 -*-

from operator import itemgetter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class ShroomClassifier:
    """Represents a classifier for the SHROOM evaluation dataset."""

    TASK = {
        "DM": "The given task is Definition Modeling, meaning that the goal of the language model is to generate a definition for the term between the '<define>' and '</define>' delimiters in the input.",
        "PG": "The given task is Paraphrase Generation, meaning that the goal of the language model is to generate a paraphrase of the input.",
        "MT": "The given task is Machine Translation, meaning that the goal of the language model is to generate a natural language translation of the input.",
        "TS": "The given task is Text Simplification, meaning that the goal of the language model is to generate a simplified version of the input.",
    }

    RATIONALE_GENERATION_PROMPT = """A language model has generated an output from a given input for a specific task.
{task}
You have been provided with an input and output pair, as well as a target that you need to use 
to determine if the output is a hallucination. A hallucination is defined as an output 
that is factually incorrect, off point, or contains extra information that cannot be reasonably 
inferred from the input. Provide a succinct rationale arguing for or against the assertion that 
the output is a hallucination.

Input: {src}
Target: {tgt} 
Output: {hyp}
Rationale:
"""

    ANSWER_GENERATION_PROMPT = """Using the argument provided in the below rationale, answer the question: 
is the output a hallucination? Answer 'Hallucination' if the output is a hallucination, or 'Not Hallucination' 
if it is not a hallucination. Only answer 'Hallucination' or 'Not Hallucination'.
  
Input: {src}
Target: {tgt} 
Output: {hyp}
Rationale: {rationale}
Answer:
"""
    
    def __init__(self, model_name="gpt-4", temperature=0.1):
        """
        Initializes a classifier for the SemEval 2024 Task 6.
        
        Parameters:
            model_name: The name of the model to be used for zero shot CoT classification (default "gpt-4").
            temperature: The temperature parameter for the model (default 0.1).
         """
        self.model_name = model_name
        self.temperature = temperature
        self.llm = self._llm(model_name, temperature)
        self.chain = self._zero_shot_chain_of_thought()

    def _llm(self, model_name, temperature):
        if model_name in [
            "gpt-4",
            "gpt-3.5-turbo",
            "gpt-4-1106-preview",
            ]:
            return ChatOpenAI(model_name=model_name, temperature=temperature, request_timeout=100)
        else:
            raise Exception(f'Model {model_name} not supported')

    def _zero_shot_chain_of_thought(self):
        """
        Creates a  LCEL chain that implements a zero-shot
        chain of thought (CoT) using a specification. 
        """
        rationale_generation = (
            ChatPromptTemplate.from_template(self.RATIONALE_GENERATION_PROMPT) 
            | self.llm 
            | StrOutputParser()
        )
        answer_generation = (
             { 
                "rationale": rationale_generation, 
                "hyp": itemgetter("hyp"),
                "src": itemgetter("src"),
                "tgt": itemgetter("tgt"),
             }
             | ChatPromptTemplate.from_template(self.ANSWER_GENERATION_PROMPT)
             | self.llm
             | StrOutputParser()
        )
        return answer_generation
    
    def classify(self, dp):
        """
        Determines whether or not the output (hyp) is a hallucination.
        
        Parameters:
            task: The task associated with a datapoint. One of "DM", "PG", "MT", or "TS".
            src: The input passed to a model.
            tgt: The intended reference "gold" text that the model ought to generate
            hyp: The output the model generated.
        
        Returns:
            A dict containing a classification of the output based on the task, input, output and target.
        """
        predictions = self.chain.batch([ { "task": self.TASK[dp["task"]], "src": dp["src"], "tgt": dp["tgt"], "hyp": dp["hyp"] } for i in range(5) ])
        weight = 1./float(len(predictions))
        predicted_p = float(sum([ weight for prediction in predictions if prediction == 'Hallucination' ]))
        if predicted_p > 0.5:
            predicted = "Hallucination"
        else:
            predicted = "Not Hallucination"
        if "id" in dp:
            return { "id": dp["id"], "label": predicted, "p(Hallucination)": predicted_p }
        else:
            return { "label": predicted, "p(Hallucination)": predicted_p }
