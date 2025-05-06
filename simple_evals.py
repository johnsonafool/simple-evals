import json
import argparse
import pandas as pd
import common
from humaneval_eval import HumanEval
from sampler.chat_completion_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    OPENAI_SYSTEM_MESSAGE_CHATGPT,
    ChatCompletionSampler,
)


def main(model_name="gpt-4o-mini-2024-07-18", base_url="https://api.openai.com/v1"):
    parser = argparse.ArgumentParser(
        description="Run sampling and evaluations using different samplers and evaluations."
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List available models"
    )
    parser.add_argument("--model", type=str, help="Select a model by name")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument(
        "--examples", type=int, help="Number of examples to use (overrides default)"
    )
    
    args = parser.parse_args()

    models = {
        model_name: ChatCompletionSampler(
            model=model_name,
            base_url=base_url,
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        )
    }

    # grading_sampler = ChatCompletionSampler(model="gpt-4o")
    # equality_checker = ChatCompletionSampler(model="gpt-4-turbo-preview")
    # ^^^ used for fuzzy matching, just for math

    def get_evals(eval_name, debug_mode):
        num_examples = (
            args.examples if args.examples is not None else (5 if debug_mode else None)
        )
        # Set num_examples = None to reproduce full evals
        match eval_name:
            case "humaneval":
                return HumanEval(num_examples=10 if debug_mode else num_examples)
            case _:
                raise Exception(f"Unrecognized eval type: {eval_name}")

    evals = {
        eval_name: get_evals(eval_name, args.debug)
        for eval_name in ["humaneval"]
    }
    debug_suffix = "_DEBUG" if args.debug else ""
    # print(debug_suffix)
    for model_name, sampler in models.items():
        for eval_name, eval_obj in evals.items():
            result = eval_obj(sampler)
            # ^^^ how to use a sampler
            metrics = result.metrics | {"score": result.score}
    
    evaluation_results = {
        "model": model_name,
        "eval_name": eval_name,
        "metrics": metrics,
    }

    print(evaluation_results)

    return evaluation_results


if __name__ == "__main__":
    main()
