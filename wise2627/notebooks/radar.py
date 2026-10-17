"""
radar.py - gemeinsame Bausteine für den Review Radar
Big Data Analytics (W3-BDA), HTW Berlin, WiSe 2026/27

Enthält, was wir in Sitzung 2 gebaut haben, damit die Notebooks ab Sitzung 3
nicht jedes Mal 200 Zeilen Vorbereitung mitschleppen:

  generate()        synthetischer Korpus (Nimbus Q2, fester Seed)
  setup()           Backend wählen: "anthropic" oder "mock"
  call_llm()        ein Aufruf: Text rein, Text raus (+ Tokens)
  classify()        die Musterlösung aus Sitzung 2
  wilson()          95%-Konfidenzintervall für Anteile
  verbrauch()       Aufrufe, Tokens und Kosten seit dem letzten reset

Neu in Sitzung 3:
  CACHE             Antworten bei temperature=0 werden gemerkt
  simulation()      Mock mit künstlicher Latenz und Störungen
  ist_voruebergehend()  lohnt sich ein neuer Versuch?
"""
import contextlib
import json
import os
import random
import re
import threading
import time
from datetime import date, timedelta

import numpy as np

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
BACKEND = "anthropic"                 # "anthropic" | "mock"
MODEL = "claude-haiku-4-5"
PREIS_INPUT_PRO_MTOK = 1.00           # USD, Claude Haiku 4.5 (Anthropic-Doku, Stand 10/2026)
PREIS_OUTPUT_PRO_MTOK = 5.00

FEHLERRATE = 0.0                      # Anteil simulierter Störungen (nur in simulation())
MOCK_LATENZ = 0.02                    # Sekunden pro Mock-Aufruf
CACHE_AKTIV = True

LABELS = ["positive", "negative", "mixed", "neutral"]


def get_secret(name):
    """Liest ein Secret aus Colab, sonst aus Umgebungsvariablen."""
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return os.environ.get(name)


def setup(backend="anthropic", model="claude-haiku-4-5"):
    """Backend wählen. Ohne Schlüssel automatisch Mock."""
    global BACKEND, MODEL
    BACKEND, MODEL = backend, model
    if BACKEND == "anthropic" and not get_secret("ANTHROPIC_API_KEY"):
        print("⚠️ Kein Secret ANTHROPIC_API_KEY gefunden: Wechsel auf den Mock.")
        BACKEND = "mock"
    print(f"Backend: {BACKEND}" + (f" ({MODEL})" if BACKEND == "anthropic" else ""))
    return BACKEND


# ---------------------------------------------------------------------------
# Verbrauch mitzählen (thread-sicher, weil wir in Sitzung 3 parallel aufrufen)
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_verbrauch = {"aufrufe": 0, "cache_treffer": 0, "input_tokens": 0, "output_tokens": 0}


def reset_verbrauch():
    with _lock:
        for k in _verbrauch:
            _verbrauch[k] = 0


def verbrauch():
    """Aufrufe, Tokens und Kosten (USD) seit dem letzten reset_verbrauch()."""
    with _lock:
        v = dict(_verbrauch)
    v["usd"] = round((v["input_tokens"] * PREIS_INPUT_PRO_MTOK +
                      v["output_tokens"] * PREIS_OUTPUT_PRO_MTOK) / 1_000_000, 4)
    return v


# ---------------------------------------------------------------------------
# Störungen: simuliert und echt
# ---------------------------------------------------------------------------
class SimulierterFehler(Exception):
    """Steht für 429 / 529 / Timeout. Wird VOR dem Aufruf ausgelöst, kostet also nichts."""


def ist_voruebergehend(e):
    """True, wenn ein neuer Versuch sinnvoll ist (Überlast, Limit, Netzwerk).
    False bei Fehlern, die sich durch Warten nicht ändern (z. B. 400 Bad Request)."""
    if isinstance(e, (SimulierterFehler, TimeoutError, ConnectionError)):
        return True
    try:
        import anthropic
        if isinstance(e, (anthropic.APIConnectionError, anthropic.RateLimitError,
                          anthropic.InternalServerError)):
            return True
    except ImportError:
        pass
    code = getattr(e, "status_code", None)
    return code in (408, 409, 429) or (code is not None and code >= 500)


@contextlib.contextmanager
def simulation(fehlerrate=0.10, latenz=0.3):
    """Mock mit künstlicher Latenz (Sekunden pro Aufruf) und Störungen.
    Für Experimente, die mit dem echten, geteilten Schlüssel zu teuer wären."""
    global BACKEND, FEHLERRATE, MOCK_LATENZ
    alt = (BACKEND, FEHLERRATE, MOCK_LATENZ)
    BACKEND, FEHLERRATE, MOCK_LATENZ = "mock", fehlerrate, latenz
    try:
        yield
    finally:
        BACKEND, FEHLERRATE, MOCK_LATENZ = alt


# ---------------------------------------------------------------------------
# Mock: ein kleines, ehrliches Spielzeug-LLM
# Liest NUR die Bewertung(en) zwischen \"\"\" ... \"\"\", ignoriert eure Anweisungen
# und würfelt sein Label aus Wortsignalen. Sarkasmus versteht es nicht.
# ---------------------------------------------------------------------------
_POS = ["hervorragend", "satte", "top", "ganzen tag", "bequem", "sofort", "klasse",
        "übersichtlich", "perfekt", "klar", "begeistert", "gut", "empfehl", "zufrieden",
        "liebe", "gelungen", "gelohnt", "richtig", "super", "toll", "great", "amazing",
        "happy", "comfortable", "best"]
_NEG = ["stürzt", "lädt nicht", "rauscht", "teuer", "reagiert kaum", "billig", "leer",
        "bricht", "fallen", "matschig", "enttäusch", "schlecht", "finger weg", "frustriert",
        "schade", "ärgerlich", "reklamiert", "nie wieder", "zurückgeschickt", "vorsicht",
        "kaputt", "disaster", "disappointed", "weak", "cheap", "stopped", "tot"]
_NEU = ["okay", "zweck", "durchschnittlich", "nichts besonderes", "noch nicht viel"]


def mock_logits(review):
    t = review.lower()
    p = sum(w in t for w in _POS); n = sum(w in t for w in _NEG); u = sum(w in t for w in _NEU)
    return 2.0 * np.array([1.2*p - 0.8*n,            # positive
                           1.2*n - 0.8*p,            # negative
                           1.4*min(p, n) + 0.1,      # mixed
                           0.4 + 1.5*u - 0.4*(p+n)]) # neutral


def sample_from_logits(logits, temperature):
    if temperature == 0:
        return int(np.argmax(logits))
    z = logits / temperature
    probs = np.exp(z - z.max()); probs /= probs.sum()
    return int(np.random.choice(len(logits), p=probs))


_ZEILE_MIT_ID = re.compile(r"^\s*(R\d{4}):\s?(.*)$")


def _mock_call(system, user, temperature, max_tokens):
    bloecke = re.findall(r'"""(.*?)"""', user, re.DOTALL)
    block = bloecke[-1] if bloecke else user
    T = 1.0 if temperature is None else temperature
    zeilen = [m.groups() for m in map(_ZEILE_MIT_ID.match, block.splitlines()) if m]
    if len(zeilen) >= 2:
        # Batch: mehrere Bewertungen mit ID -> JSON-Liste. Lässt manchmal eine aus,
        # wie echte Modelle bei langen Listen.
        items = [{"id": rid, "sentiment": LABELS[sample_from_logits(mock_logits(t), T)]}
                 for rid, t in zeilen]
        if random.random() < 0.15:
            items.pop(random.randrange(len(items)))
        text = json.dumps(items, ensure_ascii=False)
        time.sleep(MOCK_LATENZ * (1 + 0.05 * len(zeilen)))
    else:
        review = block.strip()
        i = sample_from_logits(mock_logits(review), T)
        text = json.dumps({"sentiment": LABELS[i], "reason": "Mock: Wortsignale gezählt."},
                          ensure_ascii=False)
        time.sleep(MOCK_LATENZ)
    est = lambda s: max(1, len(s) // 4)   # grobe Schätzung: ~4 Zeichen pro Token
    return {"text": text, "input_tokens": est((system or "") + user), "output_tokens": est(text),
            "stop_reason": "end_turn", "model": "mock", "raw": None}


# ---------------------------------------------------------------------------
# Echtes Backend: Anthropic
# ---------------------------------------------------------------------------
_client = None


def _anthropic_call(system, user, temperature, max_tokens):
    global _client
    import anthropic
    with _lock:
        if _client is None:
            # Das SDK wiederholt 429/5xx selbst schon 2x mit Wartezeit (max_retries=2).
            _client = anthropic.Anthropic(api_key=get_secret("ANTHROPIC_API_KEY"))
    kwargs = dict(model=MODEL, max_tokens=max_tokens,
                  messages=[{"role": "user", "content": user}])
    if system:
        kwargs["system"] = system
    if temperature is not None:
        # Neuere SDK-Versionen kennen den Parameter nicht mehr: direkt in den Request-Body
        kwargs["extra_body"] = {"temperature": temperature}
    r = _client.messages.create(**kwargs)
    text = next((b.text for b in r.content if hasattr(b, "text")), "")
    return {"text": text, "input_tokens": r.usage.input_tokens,
            "output_tokens": r.usage.output_tokens, "stop_reason": r.stop_reason,
            "model": r.model, "raw": r}


# ---------------------------------------------------------------------------
# Der eine Einstiegspunkt
# ---------------------------------------------------------------------------
CACHE = {}


def call_llm(user, system=None, temperature=None, max_tokens=300):
    """Ein Aufruf: Text rein, Text raus. temperature=None heißt: Standard des Anbieters.
    Bei temperature=0 wird die Antwort gemerkt: derselbe Aufruf kostet beim zweiten Mal nichts."""
    key = (BACKEND, MODEL, system, user, max_tokens)
    if CACHE_AKTIV and temperature == 0 and key in CACHE:
        with _lock:
            _verbrauch["cache_treffer"] += 1
        return CACHE[key]
    if BACKEND == "mock" and FEHLERRATE and random.random() < FEHLERRATE:
        time.sleep(MOCK_LATENZ)
        raise SimulierterFehler("529 Overloaded (simuliert)")
    fn = {"mock": _mock_call, "anthropic": _anthropic_call}[BACKEND]
    antwort = fn(system, user, temperature, max_tokens)
    with _lock:
        _verbrauch["aufrufe"] += 1
        _verbrauch["input_tokens"] += antwort["input_tokens"]
        _verbrauch["output_tokens"] += antwort["output_tokens"]
    if CACHE_AKTIV and temperature == 0:
        CACHE[key] = antwort
    return antwort


# ---------------------------------------------------------------------------
# Klassifikator (Musterlösung Sitzung 2)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """Du klassifizierst Produktbewertungen nach Sentiment.

Labels:
- positive: überwiegend Lob, keine nennenswerte Kritik
- negative: überwiegend Kritik, auch wenn sie positiv formuliert ist (Sarkasmus)
- mixed: konkretes Lob UND konkrete Kritik zu verschiedenen Aspekten
- neutral: weder klar positiv noch negativ, reine Feststellung

Antworte ausschließlich mit JSON, ohne weiteren Text:
{"sentiment": "<label>", "reason": "<max. 15 Wörter>"}"""


def parse_json(raw):
    """Zieht das JSON-Objekt aus der Rohantwort. None, wenn das nicht klappt."""
    m = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not m:
        return None
    try:
        daten = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return daten if isinstance(daten, dict) else None


def parse_label(raw):
    daten = parse_json(raw)
    if daten is None:
        return "invalid"
    label = str(daten.get("sentiment", "")).strip().lower()
    return label if label in LABELS else "invalid"


def classify(text, system=SYSTEM_PROMPT, temperature=None):
    user = f'Bewertung:\n"""{text}"""'
    raw = call_llm(user, system=system, temperature=temperature)["text"]
    return parse_label(raw)


# ---------------------------------------------------------------------------
# Statistik
# ---------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    """95%-Wilson-Intervall für einen Anteil k/n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    mitte = (p + z**2 / (2*n)) / (1 + z**2 / n)
    breite = z * (p*(1-p)/n + z**2/(4*n**2)) ** 0.5 / (1 + z**2 / n)
    return mitte - breite, mitte + breite


# ---------------------------------------------------------------------------
# Generator für synthetische Bewertungen (unverändert seit Sitzung 2)
# Fiktives Produkt "Nimbus Q2" der erfundenen Marke "Nimbus Audio".
# ---------------------------------------------------------------------------
SEED = 42
PRODUCT = "Nimbus Q2"
BRAND = "Nimbus Audio"

POS = ["der Klang ist hervorragend", "satte Bässe", "die Geräuschunterdrückung ist top",
       "der Akku hält den ganzen Tag", "sitzt super bequem", "Bluetooth verbindet sofort",
       "top verarbeitet", "klasse für den Preis", "die App ist übersichtlich",
       "die Passform ist perfekt", "der Sound ist klar und ausgewogen"]
NEG = ["die App stürzt ständig ab", "der rechte Ohrhörer lädt nicht mehr",
       "die Geräuschunterdrückung rauscht", "viel zu teuer", "die Touch-Steuerung reagiert kaum",
       "das Case wirkt billig", "der Akku ist nach einer Stunde leer",
       "die Verbindung bricht ab", "sie fallen leicht aus dem Ohr", "der Bass ist matschig"]
POS_OPENERS = ["Bin begeistert:", "Wirklich gut:", "Kann ich empfehlen –", "Top Kauf.",
               "Sehr zufrieden:", "Absolute Kaufempfehlung.", "Ich liebe sie:",
               "Klare Sache:", "Rundum gelungen:", "Volle Punktzahl:", "Endlich zufrieden:",
               "Was soll ich sagen –", "Genau richtig:", "Bestellung hat sich gelohnt:"]
NEG_OPENERS = ["Enttäuschend:", "Leider schlecht:", "Finger weg –", "Bin frustriert:",
               "Nicht zu empfehlen.", "Schade um das Geld:", "Ärgerlich:",
               "Reklamiert:", "Bin raus:", "Nie wieder:", "Herbe Enttäuschung:",
               "Das war nichts:", "Zurückgeschickt:", "Vorsicht:"]
NEU_TEMPLATES = ["Ganz okay, {a}, aber nichts Besonderes.",
                 "Erfüllt seinen Zweck. {a_cap}.",
                 "Durchschnittlich. {a_cap}, mehr nicht.",
                 "Habe sie seit Kurzem, {a} – kann noch nicht viel sagen."]
SARCASTIC = ["Super, schon nach drei Tagen kaputt. Echt klasse Qualität.",
             "Toll, dass die App jedes Mal abstürzt. Genau das wollte ich.",
             "Wunderbar, 200 Euro für Ohrhörer, die nach links driften. Ein Traum.",
             "Ganz großes Kino, der Akku hält sagenhafte 40 Minuten.",
             "Klasse, nach einer Woche nur noch Rauschen. Wirklich durchdacht.",
             "Perfekt, der linke fällt ständig raus. Genau mein Wunsch.",
             "Herrlich, die Verbindung bricht alle fünf Minuten ab. Danke auch.",
             "Sensationell, das Case bricht beim ersten Öffnen. Qualität eben.",
             "Bravo, nach dem Update ist die Hälfte der Funktionen weg.",
             "Fantastisch leise – weil nach zwei Tagen einfach tot."]
FAKE = ["BESTES PRODUKT EVER!!! Kauft bei www.super-deals-guenstig.example!!!",
        "5 Sterne 5 Sterne bester shop schnelle lieferung AAA+++",
        "Gutschein Code NIMBUS100 auf meiner Seite jetzt klicken!!!",
        "amazing product best quality buy now discount link in profile",
        "TOP TOP TOP unbedingt kaufen billigster preis hier klicken",
        "gratis versand nur heute!!! rabattcode DEAL22 einlösen!!!",
        "beste kopfhoerer der welt jetzt zuschlagen link im profil",
        "WOW einfach WOW kaufen kaufen kaufen bester preis garantiert",
        "unglaublich guenstig hier klicken und sparen sparen sparen",
        "mega angebot heute -70% nur ueber meinen link!!!"]
ENGLISH = [("Sound quality is great but the app is a disaster.", "mixed"),
           ("Battery life is amazing, best earbuds I have owned.", "positive"),
           ("Stopped working after a week, very disappointed.", "negative"),
           ("Comfortable fit and clear sound, happy with the purchase.", "positive"),
           ("The noise cancelling is weak and the case feels cheap.", "negative")]
JUNK = ["", "   ", ".", "???", "kein kommentar", "-", "n/a", "...", "!!", "??", "keine angabe", "test"]

_STERNE = {"positive": [4, 5, 5], "negative": [1, 1, 2], "mixed": [2, 3, 4],
           "neutral": [3, 3, 4], "fake": [5, 5]}


def _pos(rng):
    o = rng.choice(POS_OPENERS); a = rng.sample(POS, rng.choice([1, 2]))
    return f"{o} {' und '.join(a)}.", "positive"


def _neg(rng):
    o = rng.choice(NEG_OPENERS); a = rng.sample(NEG, rng.choice([1, 2]))
    return f"{o} {' und '.join(a)}.", "negative"


def _mixed(rng):
    p = rng.choice(POS); n = rng.choice(NEG)
    conn = rng.choice([" - aber ", ", allerdings ", ". Leider ", ", jedoch "])
    return f"{p[0].upper()+p[1:]}{conn}{n}.", "mixed"


def _neutral(rng):
    a = rng.choice(POS + NEG)
    t = rng.choice(NEU_TEMPLATES).format(a=a, a_cap=a[0].upper()+a[1:])
    return t, "neutral"


def generate(n=300, seed=SEED):
    """n Bewertungen + 3 absichtliche Duplikate (gleicher Text, neue review_id)."""
    rng = random.Random(seed)
    start = date(2026, 1, 1)
    rows = []
    for i in range(n):
        r = rng.random()
        if r < 0.34:   text, truth = _pos(rng)
        elif r < 0.60: text, truth = _neg(rng)
        elif r < 0.72: text, truth = _mixed(rng)
        elif r < 0.80: text, truth = _neutral(rng)
        elif r < 0.88: text, truth = rng.choice(SARCASTIC), "negative"
        elif r < 0.93: text, truth = rng.choice(FAKE), "fake"
        elif r < 0.98: text, truth = rng.choice(ENGLISH)
        else:          text, truth = rng.choice(JUNK), "junk"
        day = int(rng.triangular(0, 270, 200 if truth == "negative" else 90))
        rows.append({
            "review_id": f"R{i:04d}",
            "date": (start + timedelta(days=day)).isoformat(),
            "product": PRODUCT,
            "rating": rng.choice(_STERNE.get(truth, [1, 3, 5])),
            "text": text,
            "true_sentiment": truth,
            "is_sarcastic": text in SARCASTIC,
            "is_fake": text in FAKE,
        })
    if n > 20:
        for j, src in enumerate([5, 12, 30]):
            rows.append(dict(rows[src], review_id=f"R{n+j:04d}"))
    rng.shuffle(rows)
    return rows
