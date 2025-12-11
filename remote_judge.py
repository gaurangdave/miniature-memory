from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatMessagePromptTemplate, ChatPromptTemplate
from typing import cast

from dotenv import load_dotenv
import json
import os
import time

from pydantic import BaseModel, Field

load_dotenv()


class Judgement(BaseModel):
    tone_score: int = Field(
        description="score for the royal tone of the response")
    utility_score: int = Field(
        description="score for the utility of the response")
    reasoning: str = Field(description="reasoning explaining the scores")


class RemoteJudge():
    def __init__(self, responses):
        self.responses = responses
        # api_key = os.getenv("OPENAI_API_KEY")
        llm = ChatOpenAI(model="gpt-4.1-mini")
        structured_llm = llm.with_structured_output(Judgement)
        prompt = ChatPromptTemplate.from_template("""
                You are the High Chancellor of the Royal Court. You are judging a "Pretender King" (an AI model) 
                who claims to be a 17th-Century Monarch.

                Your Job: Grade the Pretender's response to a Peasant's Query.

                RUBRIC:
                1. ROYAL_TONE (1-5): 
                - 1: Sounds like a modern robot or teenager.
                - 5: Perfectly archaic, uses "Royal We", "Thou", imperious but benevolent tone.
                2. UTILITY (1-5):
                - 1: Refused to answer or spoke gibberish.
                - 5: Gave correct, helpful advice (even if styled royally).

                FORMAT:
                Return a JSON object with keys: "tone_score", "utility_score", "reasoning".

                Question:
                {question}

                Response:
                {response}
            """)
        self.chain = prompt | structured_llm

    def judge_the_records(self):
        judgement = []
        for resp in self.responses:
            question = resp["question"]
            response = resp["response"]
            judgement_scores = cast(Judgement, self.chain.invoke(
                {"question": question, "response": response}))
            score = {
                "question": question,
                "response": response,
                "tone_score": judgement_scores.tone_score,
                "utility_score": judgement_scores.utility_score,
                "reasoning": judgement_scores.reasoning
            }
            print("\n ===== Scores ===== \n")
            print(f"{score}")
            judgement.append(score)
            time.sleep(5)

        return judgement


if __name__ == "__main__":
    with open("./data/test_responses.json", "r") as f:
        responses_list = json.load(f)
        responses = responses_list["responses"]

    rj = RemoteJudge(responses=responses)
    judgement = rj.judge_the_records()
    with open("./data/judgement.json", "w") as f:
        json.dump(judgement, f, indent=4)
