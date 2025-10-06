import dspy
import re

lm = dspy.LM(model="ollama_chat/llama3:latest", api_base="http://localhost:11434")
dspy.configure(lm=lm)

# --- Step 4: Provide example training data ---
train_data = [
    dspy.Example(
        question="How do I deploy a model to Azure Machine Learning?",
        answer="Register your model in Azure ML, create an inference endpoint, and deploy it as a web service.",
    ).with_inputs("question"),
    dspy.Example(
        question="What’s the difference between Azure Functions and Logic Apps?",
        answer="Functions run custom code in response to events, while Logic Apps orchestrate workflows using connectors and triggers.",
    ).with_inputs("question"),
    dspy.Example(
        question="How can I scale my Azure App Service?",
        answer="You can scale manually or automatically using autoscale rules or by upgrading your App Service plan tier.",
    ).with_inputs("question"),
]


# --- Step 5: Define metric
# Define the signature for automatic assessments.
class Assess(dspy.Signature):
    """Assess the similarity of the two text"""

    assessed_text_1 = dspy.InputField()
    assessed_text_2 = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_score: bool = dspy.OutputField()


def metric(gold, pred, trace=None):
    # you can also use gold.question (not use here)
    answer, pred = gold.answer, pred.answer

    correct_assesment = f"Does `{pred}` similar to `{answer}`?"

    score_prediction = dspy.Predict(Assess)(
        assessed_text_1=answer,
        assessed_text_2=pred,
        assessment_question=correct_assesment,
    )

    score = score_prediction.assessment_score

    if trace is not None:
        return score >= 1
    return int(score)


# --- Step 6: Initialize the optimizer ---
# optimizer = dspy.BootstrapFewShot(metric=metric)
optimizer = dspy.MIPROv2(metric=metric, verbose=True)

# --- Step 7: Run optimization   ---
optimized_agent = optimizer.compile(
    dspy.ChainOfThought("question -> answer"), trainset=train_data
)

# --- Step 8: Test with a new question ---
response = optimized_agent(question="How can I secure data in Azure Storage?")
print("\nResponse:", response)

# --- Step 9: Inspect the learned (optimized) prompt ---
print("\nOptimized Agent:\n")
print(optimized_agent)


# --- Step 10: save program ---answer
optimized_agent.save("optimize.json")
