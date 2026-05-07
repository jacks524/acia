import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import DEVICE, LANG_LABELS, LLM_NAME
from .retrieval import retrieve

tokenizer = None
llm = None


def _get_llm():
    global tokenizer, llm
    if tokenizer is not None and llm is not None:
        return tokenizer, llm

    print(f"Chargement du LLM : {LLM_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)

    if DEVICE == "cuda":
        llm = AutoModelForCausalLM.from_pretrained(
            LLM_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    else:
        llm = AutoModelForCausalLM.from_pretrained(
            LLM_NAME,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).to(DEVICE)

    llm.eval()
    print(f"LLM pret sur {DEVICE}")
    return tokenizer, llm


def build_prompt(question: str, retrieved_chunks: list[tuple[float, dict]], target_lang_label: str):
    context_parts = []
    for i, (_score, chunk) in enumerate(retrieved_chunks, 1):
        context_parts.append(
            f"[Source {i} - {chunk['lang_label']} - {chunk['title']}]\n{chunk['text']}"
        )
    context = "\n\n".join(context_parts)

    system_msg = (
        "Tu es un assistant agricole pour les producteurs de tomate au Cameroun. "
        f"Tu réponds UNIQUEMENT en {target_lang_label}, de manière simple et pratique. "
        "Base tes réponses STRICTEMENT sur les sources fournies. "
        f"Ne traduis pas vers une autre langue : la réponse finale doit être en {target_lang_label}."
    )
    user_msg = (
        f"Sources extraites du manuel FarmAI :\n\n{context}\n\n"
        f"Question de l'agriculteur : {question}\n\n"
        f"Réponse en {target_lang_label} (4-6 phrases courtes, conseils concrets) :"
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def generate_response(
    question: str,
    target_lang: str = "fr",
    k: int = 3,
    max_new_tokens: int = 300,
) -> dict:
    lang_label = LANG_LABELS.get(target_lang, "Français")
    retrieved = retrieve(question, lang=target_lang, k=k)

    if not retrieved:
        return {
            "question": question,
            "target_lang": target_lang,
            "answer": f"(Aucune information trouvée en {lang_label})",
            "sources": [],
        }

    tokenizer, llm = _get_llm()

    messages = build_prompt(question, retrieved, lang_label)
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_device = next(llm.parameters()).device
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model_device)
    with torch.no_grad():
        outputs = llm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).strip()

    return {
        "question": question,
        "target_lang": target_lang,
        "answer": response,
        "sources": [
            {"score": score, "title": chunk["title"], "lang": chunk["lang_label"]}
            for score, chunk in retrieved
        ],
    }
