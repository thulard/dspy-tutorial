import dspy
import json

lm = dspy.LM(model="ollama_chat/llama3:latest", api_base="http://localhost:11434")
dspy.configure(lm=lm)


def simple():
    question = "how to integrate dspy into an angent?"
    simple = dspy.ChainOfThought("question -> answer: str")
    response = simple(question=question)

    dspy.Completions

    print(response.answer)

    print(len(lm.history))
    print(lm.history[-1].keys())
    print(lm.history[-1])


class Toxic(dspy.Signature):
    """Mark as 'toxic' if the text include insult or harassment"""

    comment: str = dspy.InputField()
    toxic: bool = dspy.OutputField()


def predict():
    comment = "you are sexy"
    toxic_eval = dspy.Predict(Toxic)
    print(toxic_eval(comment=comment))


def main():
    predict()


if __name__ == "__main__":
    main()
