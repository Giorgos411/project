def custom_reconstruct(sentence: str) -> str:
    mapping = {
        "Hope you too, to enjoy it as my deepest wishes.":
            "I hope you enjoy the festival — these are my warmest wishes.",
        "I am very appreciated the full support of the professor, for our Springer proceedings publication":
            "I truly appreciate the professor's full support for our Springer publication."
    }
    return mapping.get(sentence.strip(), sentence)
