from .intent import (
    INTENT_GOODBYE,
    INTENT_GREETING,
    INTENT_IDENTITY,
    INTENT_OUT_OF_SCOPE,
    INTENT_THANKS,
    INTENT_TOMATO_GENERIC,
    INTENT_UNCLEAR,
)

CANONICAL_RESPONSES = {
    INTENT_GREETING: {
        "fr": "Bonjour ! Je suis FarmAI, votre assistant pour les maladies de la tomate. Envoyez-moi une photo d'une feuille ou posez-moi une question.",
        "en": "Hello! I am FarmAI, your assistant for tomato diseases. Send me a photo of a leaf or ask me a question.",
        "ha": "Sannu! Ni FarmAI ne, mai taimako kan cututtukan tumatur. Aiko min hoton ganye ko yi tambaya.",
        "ff": "Jam! Mi woni FarmAI, ballondiroowo nyawu tomati. Neldam natal haako walla ƴamto.",
    },
    INTENT_IDENTITY: {
        "fr": "Je suis FarmAI Cameroun, un assistant agricole intelligent. Je peux identifier 10 maladies de la tomate à partir d'une photo et donner des conseils en français, anglais, hausa et fulfulde.",
        "en": "I am FarmAI Cameroun, an intelligent agricultural assistant. I can identify 10 tomato diseases from a photo and provide advice in French, English, Hausa and Fulfulde.",
        "ha": "Ni FarmAI Cameroun ne, mai taimako na hankali na noma. Zan iya gane cututtuka 10 na tumatur daga hoto kuma in ba shawara a faransanci, turanci, hausa da fulfulde.",
        "ff": "Mi woni FarmAI Cameroun, ballondiroowo hakkille remoowo. Mi waawi heɓtinde nyawuuji 10 tomati e natal, mi ɓetta tinndi e farayse, engele, hawsa, e fulfulde.",
    },
    INTENT_THANKS: {
        "fr": "Avec plaisir ! N'hésitez pas si vous avez d'autres questions sur vos plants de tomate.",
        "en": "You're welcome! Feel free to ask if you have other questions about your tomato plants.",
        "ha": "Babu komai! Ka iya sake tambaya idan kana da wata tambaya kan tumatur.",
        "ff": "A jaaraama! A waawi naamnaade so a jogii ƴamnde woɗnde e tomati maa.",
    },
    INTENT_GOODBYE: {
        "fr": "À bientôt ! Bonnes récoltes.",
        "en": "See you soon! I wish you a good harvest.",
        "ha": "Sai anjima! Allah ya ba da girbi mai kyau.",
        "ff": "Haa later! Yo ngesa maa moƴƴa.",
    },
    INTENT_UNCLEAR: {
        "fr": "Je n'ai pas bien compris. Pouvez-vous répéter votre question ou m'envoyer une photo claire d'une feuille de tomate ?",
        "en": "I didn't understand clearly. Could you repeat your question or send me a clear photo of a tomato leaf?",
        "ha": "Ban fahimta sosai ba. Za ka iya maimaita tambayar ko ka aiko min hoton ganyen tumatur a sarari?",
        "ff": "Mi faamaani no feewi. A waawi tonyude ƴamol maa walla neldoyam natal laaɓɗo haako tomati?",
    },
    INTENT_OUT_OF_SCOPE: {
        "fr": "Je suis spécialisé dans les maladies de la tomate. Pour cette question, consultez un agronome ou une autre source fiable. Avez-vous une question sur vos plants de tomate ?",
        "en": "I am specialized in tomato diseases. For this question, please consult an agronomist or another reliable source. Do you have a question about your tomato plants?",
        "ha": "Na fi ƙwarewa kan cututtukan tumatur. Don wannan tambayar, ka tuntubi masanin noma ko wata ingantacciyar hanya. Kana da tambaya kan tumatur?",
        "ff": "Mi serii e nyawuuji tomati. E ndee ƴamnde, ɗaɓɓu ballal agronoom walla humpito koolniiɗo. A jogii ƴamnde e tomati maa?",
    },
    INTENT_TOMATO_GENERIC: {
        "fr": "Votre question concerne les tomates mais sort de mon domaine précis : les 10 maladies que je sais identifier. Je peux vous aider avec la tache bactérienne, l'alternariose, le mildiou tardif, la moisissure, la septoriose, les acariens, la tache cible, le virus mosaïque, le virus de l'enroulement jaune, et une plante saine. Décrivez un symptôme précis ou envoyez une photo.",
        "en": "Your question is about tomatoes but outside my precise scope: the 10 diseases I can identify. I can help with bacterial spot, early blight, late blight, leaf mold, septoria, spider mites, target spot, mosaic virus, yellow leaf curl virus, and healthy plants. Describe a specific symptom or send a photo.",
        "ha": "Tambayarka tana kan tumatur, amma ta fita daga iyakata ta musamman: cututtuka 10 da nake iya ganewa. Zan iya taimaka kan bacterial spot, early blight, late blight, leaf mold, septoria, spider mites, target spot, mosaic virus, yellow leaf curl virus, da shuka mai lafiya. Ka bayyana alama ko ka aika hoto.",
        "ff": "Ƴamnde maa ina jogii heen tomati, kono nde ɓurtii laawol am: nyawuuji 10 ɗi mi waawi heɓtinde. Mi waawi wallude e bacterial spot, early blight, late blight, leaf mold, septoria, spider mites, target spot, mosaic virus, yellow leaf curl virus, e haako selli. Siiftu maande laaɓnde walla neldam natal.",
    },
}


def build_response(intent: str, lang: str, original_question: str | None = None) -> dict:
    responses = CANONICAL_RESPONSES.get(intent, CANONICAL_RESPONSES[INTENT_UNCLEAR])
    answer = responses.get(lang, responses["fr"])
    return {
        "detected_disease": None,
        "question": original_question or "",
        "target_lang": lang,
        "answer_text": answer,
        "sources": [],
        "audio_path": None,
        "intent": intent,
    }
