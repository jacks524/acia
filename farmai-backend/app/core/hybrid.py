from ..config import LANG_LABELS, USE_LLM
from .retrieval import retrieve


def generate_response_retrieval_only(
    question: str,
    target_lang: str = "ha",
    k: int = 1,
    debug: bool = False,
) -> dict:
    lang_label = LANG_LABELS.get(target_lang, "Français")
    n_candidates = max(k, 10)
    retrieved = retrieve(question, lang=target_lang, k=n_candidates)

    if not retrieved:
        return {
            "question": question,
            "target_lang": target_lang,
            "answer": f"(Aucune information trouvée en {lang_label})",
            "sources": [],
        }

    question_words = set(question.lower().split())
    rescored = []
    for score, chunk in retrieved:
        title_lower = chunk["title"].lower()
        common = sum(1 for word in question_words if len(word) > 3 and word in title_lower)
        boosted_score = score + (common * 0.3)
        rescored.append((boosted_score, score, chunk, common))

    rescored.sort(key=lambda x: -x[0])

    if debug:
        print(f"Requête : {question}")
        for boosted, original, chunk, matches in rescored[:5]:
            print(
                f"[boosted={boosted:.3f}, orig={original:.3f}, "
                f"matches={matches}] {chunk['title']}"
            )

    _best_boosted, _best_score, best_chunk, _matches = rescored[0]
    answer = best_chunk["text"]
    title = best_chunk["title"]
    if answer.startswith(title):
        answer = answer[len(title) :].lstrip(". ").strip()

    return {
        "question": question,
        "target_lang": target_lang,
        "answer": answer,
        "sources": [
            {"score": score, "title": chunk["title"], "lang": chunk["lang_label"]}
            for _boosted, score, chunk, _common in rescored[:4]
        ],
    }


def generate_response_hybrid(question: str, target_lang: str = "ha", k: int = 2) -> dict:
    """
    HA/FF : retrieval-only pour éviter les hallucinations linguistiques.
    FR/EN : RAG + LLM si USE_LLM=true, sinon retrieval-only.
    """
    if target_lang in ["ha", "ff"] or not USE_LLM:
        return generate_response_retrieval_only(question, target_lang, k=max(k, 4))

    from .llm import generate_response

    return generate_response(question, target_lang, k=k)
