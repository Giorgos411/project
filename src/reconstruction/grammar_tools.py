from gramformer import Gramformer
import language_tool_python

gf = Gramformer(models=1, use_gpu=False)
tool = language_tool_python.LanguageToolPublicAPI('en-US')

def correct_with_gramformer(text: str) -> str:
    corrected = list(gf.correct(text))
    return corrected[0] if corrected else text

def correct_with_languagetool(text: str) -> str:
    matches = tool.check(text)
    return language_tool_python.utils.correct(text, matches)
