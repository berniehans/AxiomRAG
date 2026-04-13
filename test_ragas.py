from ragas.metrics import faithfulness, context_precision
from ragas.llms import llm_factory
from openai import OpenAI
from datasets import Dataset

from ragas.evaluation import evaluate

llm = llm_factory(model="gpt-3.5-turbo", client=OpenAI(api_key="x"))

ds = Dataset.from_dict({"question": ["q"], "answer": ["a"], "contexts": [["c"]], "ground_truth": ["g"]})

try:
    evaluate(ds, metrics=[faithfulness, context_precision], llm=llm)
    print("SUCCESS")
except Exception as e:
    print("ERROR:", e)
