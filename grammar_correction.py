try:
    from gramformer import Gramformer
    gf = Gramformer(models=1, use_gpu=False)  

    def correct_grammar(text):
        corrected_sentences = gf.correct(text, max_candidates=1)
        return list(corrected_sentences)[0] if corrected_sentences else text

except ImportError:
    from gingerit.gingerit import GingerIt

    def correct_grammar(text):
        parser = GingerIt()
        return parser.parse(text)['result']

if __name__ == "__main__":
    text = input("Enter text to correct: ")
    print("Corrected Text:", correct_grammar(text))

