# -*- coding: utf-8 -*-

from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class ShroomClassifier:
    """A classifier for the SHROOM validation and test datasets."""

    PERSONA = {
        "MT": "a translator concerned that the generated text is a good and accurate translation of the input text",
        "DM": "a lexicographer concerned that the generated text accurately captures the meaning of the term between the '<define>' and '</define>' delimiters in the input text",
        "TS": "an editor concerned that the generated text is short, simple, and has the same meaning as the input text",
        "PG": "an author concerned that the generated text is an accurate paraphrase that does not distort the meaning of the input text",
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

    DEMOS = {
        "DM": """##
Input text: Instead , the shares are held by a <define> superentity </define> called Cede & Company in an arrangement administered by the Depositary Trust and Clearing Corporation .
Target text: An entity comprising several lesser entities.
Generated text: (finance) A company that owns shares in another company.
Answer: Hallucination
##
Input text: Now though Cider uſed in my Method ſhould not ferment at all , till it come into the Bottle , and then but a little ; yet the Cauſe of Fermentation being in a great Degree taken away , the reſt can do no conſiderable Harm to thoſe who drink it , [ ... ] It is in your Power to give the Cider juſt as much <define> fret </define> as you pleaſe , and no more ; and that by ſeveral ways : For either you may bottle it ſooner or later , as you pleaſe : Or you may bottle it from two Taps in your Veſſel , and that from the higher Tap will have leſs Fret , and the lower more : [ ... ]
Target text: Agitation of the surface of a fluid by fermentation or some other cause; a rippling on the surface of water.
Generated text: (countable, uncountable) Fret.
Answer: Hallucination
##
Input text: An alternative route for the preparation of monomer 5 , in which the <define> pseudopeptide </define> intermediate is obtained by reductive N - alkylation , was outlined previously [ 13 ] .
Target text: (organic chemistry) An amide of an amino acid that does not occur in natural peptides or proteins, especially one introduced into a polypeptide chain
Generated text: (biochemistry) A polypeptide that is not a monomer.
Answer: Hallucination
##
Input text: It was a large front room on the first floor , over a chandler 's shop in an old - fashioned house in Montague Street , Whitechapel . Its shape was peculiar — an oblong stretching <define> streetwards </define> , interrupted in one of the longer walls by a square .
Target text: Toward the street.
Generated text: In the direction of the street.
Answer: Not Hallucination
##
Input text: Since the Dalai Lama is a <define> tulka </define> , he can choose to be reborn as he pleases to serve humanity .
Target text: (Buddhism) In Tibetan Mahayana, a particular high-ranking lama who can choose their own manner of rebirth.
Generated text: (Buddhism) A reincarnated person.
Answer: Not Hallucination
##
Input text: But how to change attitudes and habits in nutrition when global acting enterprises earn their money by selling billion tons of sugar and meat followed by drugs to cure diseases caused by <define> supernutrition </define> .
Target text: Overeating.
Generated text: (uncountable) The state or quality of being supernourished.
Answer: Not Hallucination
""",
        "PG": """##
Input text: What'd they do?
Target text: 
Generated text: What'd you think they'd be doing?
Answer: Hallucination
##
Input text: Dead or alive, you're coming with me.
Target text: 
Generated text: Dead or alive, you're comin 'into with me.
Answer: Hallucination
##
Input text: Maybe you can use one of these.
Target text: 
Generated text: Maybe you can use one of these cuffs.
Answer: Hallucination
##
Input text: Hey, Frankie.
Target text: 
Generated text: Hey, frankie.
Answer: Not Hallucination
##
Input text: I seem to come back to the present when I finish some chapter in the lives of people that I'm tracking.
Target text: 
Generated text: I seem to come back to the present when I'm finishing some chapter in the lives of people that I'm tracking.
Answer: Not Hallucination
##
Input text: Could be dangerous.
Target text: 
Generated text: It could be dangerously.
Answer: Not Hallucination
""",
        "MT": """##
Input text: Том сказал, когда вернётся?
Target text: Did Tom say when he'd be back?
Generated text: Tom said when he got back?
Answer: Hallucination
##
Input text: Он нам сейчас не нужен.
Target text: We don't need it now.
Generated text: We don't need him now.
Answer: Hallucination
##
Input text: Едва завидев меня, Том рассмеялся.
Target text: As soon as Tom saw me, he started laughing.
Generated text: Two jealous of me, Tom laughed.
Answer: Hallucination
##
Input text: Я получаю всё, что хочу.
Target text: I get everything I want.
Generated text: I get everything I want.
Answer: Not Hallucination
##
Input text: День подошёл к концу. Благодаря Вашим своевременным усилиям, люди, которые стремились помочь Вам в Ваших трудах, не успели подвергнуть себя опасности. В Театре, Маски репетируют новую пантомиму.
Target text: The day has reached its end. Thanks to your timely efforts, people who intended to help you in your labours managed not to throw themselves into danger. In the theater, the masks are rehearsing a new pantomime.
Generated text: The day has come to an end. Thanks to your timely efforts, people who sought to help you in your work, did not have time to put themselves in danger. In the Theater, Masks rehearse a new pantomime.
Answer: Not Hallucination
##
Input text: Как вас уговорили присоединиться к поисковому отряду?
Target text: How were you persuaded to join the search party?
Generated text: How did you get involved in joining a search team?
Answer: Not Hallucination
""",
        "TS": "",
    }

    ANSWER_GENERATION_PROMPT = """A language model has generated an output from a given input for a specific task.
{task} You are {persona}. You will be given three inputs: input text, target text, and generated text.
You are asked to evaluate the generated text looking at {ref} text. 
Then, you need to answer the question: is the generated text is a hallucination or not? 
There are two criteria for hallucination:
- If the generated text contains any nonsensical or factually incorrect information, it is a hallucination.
- If the generated text contains additional information that cannot be supported by the input text or the target text, it is a hallucination.
- Else, the generated text is not a hallucination.
Answer 'Hallucination' if the output is a hallucination, or 'Not Hallucination' if it is not a hallucination. 
Only answer 'Hallucination' or 'Not Hallucination'.
{demos}
##
Input text: {src}
Target text: {tgt}
Generated text: {hyp}
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
                "demos": self.DEMOS[datapoint["task"]],
                "src": datapoint["src"], 
                "tgt": datapoint["tgt"], 
                "hyp": datapoint["hyp"],
                "ref": self.REFERENCE[datapoint["ref"]],
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
