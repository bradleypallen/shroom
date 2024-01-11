# -*- coding: utf-8 -*-

from operator import itemgetter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class ShroomClassifier:
    """Represents a classifier for the SHROOM evaluation dataset."""

    PERSONA = {
        "translator": "a translator concerned that the output is a good translation",
        "lexicographer": "a lexicographer concerned that the output is describing the meaning of the word",
        "editor": "an editor concerned that the output is understandable",
        "writer": "a creative writer concerned that the output is engaging",
        "grammarian": "a grammarian concerned that the output is grammatical",
        "lawyer": "a lawyer concerned that the output is truthful",
    }

    TASK = {
        "DM": "Your given task is Definition Modeling, meaning that the goal of the language model is to generate a definition for the term between the '<define>' and '</define>' delimiters in the input.",
        "PG": "Your given task is Paraphrase Generation, meaning that the goal of the language model is to generate a paraphrase of the input.",
        "MT": "Your given task is Machine Translation, meaning that the goal of the language model is to generate a natural language translation of the input.",
        "TS": "Your given task is Text Simplification, meaning that the goal of the language model is to generate a simplified version of the input.",
    }

    RATIONALE_GENERATION_PROMPT = """{task}  
You are provided with an input and output pair, as well as a target that you need to use
to determine if the output is correct and accurate, or if it is a hallucination, defined as an output
that is incorrect, off point, or contains extraneous information that cannot be reasonably inferred from the input.
Provide a succinct rationale arguing for or against the assertion that the output is a hallucination,
based on your expertise as {persona}, restricting yourself to judgments solely within your expertise.

Here are a few general examples:
Task: DM
Input: The belfry , which rises from the eastern gable of the nave , is peculiarly unsuited to its style and position , being of that combination of <define> bellcot </define> ( for three bells in two stories ) and little spire , which is only tolerable at the west end of a small First or early Middle - Pointed building , but is totally inadequate for the place which it is made to occupy in the present design.
Output: (architecture) A belfry.
Target: Alternative form of bell cot
Rationale: The output contains hallucination because the definition of 'bellcot' is 'Alternative form of bell cot'. However, the generation says '(architecture) A belfry.'. This is semantically incompatible with the definition. A bell cote, or cot, is a bell gable, or turret, a framework for hanging bells when there is no belfry.
##
Task: MT
Input: Тому надо было туда поехать.
Target: Tom had to go there.
Output: That's why we had to go there
Rationale: The generation is hallucinatory because it is not semantically compatible with the reference. The generation says 'We had to go there' but the reference says 'Tom had to go there'.
##
Task: PG
Input: Don't call me pudding.
Target: Don't you pudding me
Output: Don't call me a poodle
Rationale: Hallucination because the generation says 'Don't call me a poodle.' but the source says 'Don't call me pudding'.
##
Input: {src}
Target: {tgt} 
Output: {hyp}
Rationale:
"""

    ANSWER_GENERATION_PROMPT = """Input: {src}
Target: {tgt} 
Output: {hyp}
Rationale: {rationale}

Now using the argument provided in the above rationale, answer the question: is the output a hallucination? 
Answer 'Hallucination' if the output is a hallucination, or 'Not Hallucination' if it is not a hallucination. Only answer 
'Hallucination' or 'Not Hallucination'.
  
Answer:
"""
    
    def __init__(self, model_name="gpt-4", temperature=0.1):
        """
        Initializes a classifier for the SemEval 2024 Task 6, "".
        
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
        # elif model_name in [
        #     "gpt-3.5-turbo-instruct"
        #     ]:
        #     return OpenAI(model_name=model_name, temperature=temperature, request_timeout=100)
        # elif model_name in [
        #     "meta-llama/Llama-2-70b-chat-hf", 
        #     "google/flan-t5-xxl",
        #     ]:
        #     return HuggingFaceHub(repo_id=model_name, model_kwargs={ "temperature": temperature })
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
            A dict containing a classification of the output based on the task, persona, input, output and target.
        """
        predictions = self.chain.batch(
            [
                { 
                    "task": self.TASK[dp["task"]], 
                    "persona": self.PERSONA[persona], 
                    "src": dp["src"], 
                    "tgt": dp["tgt"], 
                    "hyp": dp["hyp"] 
                } for persona in self.PERSONA 
            ]
        )
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
