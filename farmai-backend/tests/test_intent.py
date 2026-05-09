from app.core.intent import (
    INTENT_DISEASE_QUERY,
    INTENT_GREETING,
    INTENT_IDENTITY,
    INTENT_OUT_OF_SCOPE,
    INTENT_TOMATO_GENERIC,
    INTENT_UNCLEAR,
    classify_intent,
)


def test_greeting_intent():
    assert classify_intent("bonjour", "fr") == INTENT_GREETING


def test_identity_intent():
    assert classify_intent("qui es-tu ?", "fr") == INTENT_IDENTITY


def test_disease_query_intent():
    assert classify_intent("comment traiter le mildiou", "fr") == INTENT_DISEASE_QUERY


def test_tomato_generic_intent():
    assert classify_intent("quand récolter mes tomates", "fr") == INTENT_TOMATO_GENERIC


def test_out_of_scope_intent():
    assert classify_intent("quel temps fera-t-il demain", "fr") == INTENT_OUT_OF_SCOPE


def test_unclear_intent():
    assert classify_intent("", "fr") == INTENT_UNCLEAR
    assert classify_intent("asdjklasd", "fr") == INTENT_UNCLEAR


def test_multilingual_social_and_disease_intents():
    assert classify_intent("sannu", "ha") == INTENT_GREETING
    assert classify_intent("mildiou tumatur", "ha") == INTENT_DISEASE_QUERY
    assert classify_intent("jam", "ff") == INTENT_GREETING
    assert classify_intent("haako tomati", "ff") == INTENT_TOMATO_GENERIC
