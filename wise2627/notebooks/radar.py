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

Neu in Sitzung 4:
  mit_retry(), parse_batch(), classify_many()   Musterlösungen aus Sitzung 3
  export_roh()      derselbe Korpus als unordentlicher Shop-Export (+ Wahrheit)

Neu in Sitzung 5:
  lade_sauber()     der bereinigte Datensatz aus Sitzung 4, in einem Aufruf
  THEMEN, wahre_themen()   Themen-Katalog und Wahrheit je Bewertung
  embed()           Texte -> Vektoren (Sentence-Transformers, sonst Ersatz)
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
TEMPERATUR_ERLAUBT = True          # wird False, wenn die API den Parameter ablehnt

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


_ZEILE_MIT_ID = re.compile(r"^\s*([A-Z]\d{4,5}):\s?(.*)$")


def _mock_call(system, user, temperature, max_tokens):
    bloecke = re.findall(r'"""(.*?)"""', user, re.DOTALL)
    block = bloecke[-1] if bloecke else user
    T = 1.0 if temperature is None else temperature
    zeilen = [m.groups() for m in map(_ZEILE_MIT_ID.match, block.splitlines()) if m]
    if zeilen and '"thema"' in (system or ""):
        # Themen-Batch: [{"id": ..., "themen": [{"thema": ..., "polaritaet": ...}]}]
        items = [{"id": rid, "themen": _mock_themen(t)} for rid, t in zeilen]
        text = json.dumps(items, ensure_ascii=False)
        time.sleep(MOCK_LATENZ * (1 + 0.05 * len(zeilen)))
        est = lambda s: max(1, len(s) // 4)
        return {"text": text, "input_tokens": est((system or "") + user), "output_tokens": est(text),
                "stop_reason": "end_turn", "model": "mock", "raw": None}
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
    global TEMPERATUR_ERLAUBT
    if temperature is not None and TEMPERATUR_ERLAUBT:
        # Neuere SDK-Versionen kennen den Parameter nicht mehr: direkt in den Request-Body
        kwargs["extra_body"] = {"temperature": temperature}
    try:
        r = _client.messages.create(**kwargs)
    except anthropic.BadRequestError as e:
        # Lehnt die API den Parameter ab, einmal ohne ihn wiederholen und ab dann weglassen
        if "extra_body" not in kwargs or "temperature" not in str(e):
            raise
        with _lock:
            if TEMPERATUR_ERLAUBT:
                TEMPERATUR_ERLAUBT = False
                print("⚠️ Die API akzeptiert für dieses Modell keinen temperature-Parameter. "
                      "Alle Aufrufe laufen ab jetzt mit der Standardeinstellung des Anbieters.")
        kwargs.pop("extra_body")
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


# ---------------------------------------------------------------------------
# Musterlösungen aus Sitzung 3: Wiederholen, Batching
# ---------------------------------------------------------------------------
def mit_retry(fn, *args, versuche=5, basis=0.5, **kwargs):
    """Wiederholt fn bei vorübergehenden Fehlern, mit exponentieller Wartezeit + Jitter."""
    for k in range(versuche):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if not ist_voruebergehend(e) or k == versuche - 1:
                raise
            time.sleep(basis * 2**k * random.uniform(0.5, 1.5))


BATCH_PROMPT = """Du klassifizierst Produktbewertungen nach Sentiment.

Labels:
- positive: überwiegend Lob, keine nennenswerte Kritik
- negative: überwiegend Kritik, auch wenn sie positiv formuliert ist (Sarkasmus)
- mixed: konkretes Lob UND konkrete Kritik zu verschiedenen Aspekten
- neutral: weder klar positiv noch negativ, reine Feststellung

Du bekommst mehrere Bewertungen, jede mit ID. Bewerte jede einzeln.
Antworte ausschließlich mit einer JSON-Liste, eine Zeile pro Bewertung, in derselben Reihenfolge:
[{"id": "<ID>", "sentiment": "<label>"}, ...]"""


def parse_batch(raw, ids):
    ergebnis = {i: "missing" for i in ids}
    m = re.search(r"\[.*\]", raw or "", re.DOTALL)
    try:
        items = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        items = None
    if not isinstance(items, list):
        return {i: "invalid" for i in ids}
    for item in items:
        if not isinstance(item, dict) or item.get("id") not in ergebnis:
            continue
        label = str(item.get("sentiment", "")).strip().lower()
        ergebnis[item["id"]] = label if label in LABELS else "invalid"
    return ergebnis


def classify_many(ids, texte, batch=20):
    """Ordnet viele Bewertungen in Batches ein; Lücken werden einzeln nachgeholt.
    ids und texte: gleich lange Listen. Rückgabe: Liste von Labels in derselben Reihenfolge.
    IDs müssen innerhalb eines Batches eindeutig sein; doppelte werden intern umbenannt."""
    ids, texte = list(ids), list(texte)
    schluessel = [f"X{i:05d}" for i in range(len(ids))]   # eindeutig, egal wie die echten IDs aussehen
    labels = {}
    for start in range(0, len(texte), batch):
        k = schluessel[start:start + batch]
        t = texte[start:start + batch]
        liste = "\n".join(f"{a}: {' '.join(str(b).split())}" for a, b in zip(k, t))
        user = f'Bewertungen:\n"""\n{liste}\n"""'
        antwort = mit_retry(call_llm, user, system=BATCH_PROMPT, temperature=0,
                            max_tokens=25 * len(k) + 100)
        teil = parse_batch(antwort["text"], k)
        for a, b in zip(k, t):
            if teil[a] in ("missing", "invalid"):
                try:
                    teil[a] = mit_retry(classify, b, temperature=0)
                except Exception:
                    teil[a] = "error"
        labels.update(teil)
    return [labels[k] for k in schluessel]


# ---------------------------------------------------------------------------
# Sitzung 4: derselbe Korpus als unordentlicher Shop-Export
# ---------------------------------------------------------------------------
def _mojibake(text):
    """UTF-8-Text, der fälschlich als Latin-1 gelesen wurde: 'ä' -> 'Ã¤'."""
    return text.encode("utf-8").decode("latin-1")


def export_roh(seed=7):
    """Gibt (roh, wahrheit) zurück.

    roh:      DataFrame, alle Spalten als Text, wie aus einem CSV-Export
              (Spalten: review_id, quelle, datum, produkt, sterne, text)
    wahrheit: DataFrame je review_id mit den echten Werten und den Wahrheits-Markern
              (nur für den Kurs; in echten Daten gibt es sie nicht)
    """
    import pandas as pd
    rng = random.Random(seed)
    basis = generate(300)
    zeilen, wahr = [], []
    for r in basis:
        quelle = rng.choices(["web", "app_de", "app_us"], weights=[55, 30, 15])[0]
        d = date.fromisoformat(r["date"])
        stunde, minute = rng.randrange(7, 23), rng.randrange(60)
        # Datum je Quelle in eigenem Format; Web teils als Zeitstempel
        if rng.random() < 0.03:
            datum = ""
        elif quelle == "web":
            datum = (f"{d.isoformat()}T{stunde:02d}:{minute:02d}:00Z" if rng.random() < 0.3
                     else d.isoformat())
        elif quelle == "app_de":
            datum = d.strftime("%d.%m.%Y")
        else:
            datum = d.strftime("%m/%d/%Y")
        # Sterne: fehlen bei negativen Bewertungen deutlich häufiger
        p_fehlt = 0.40 if r["true_sentiment"] == "negative" else 0.05
        s = r["rating"]
        if rng.random() < p_fehlt:
            sterne = rng.choice(["", "", "k.A.", "null"])
        elif quelle == "web":
            sterne = rng.choice([str(s), str(s), f"{s} Sterne"])
        elif quelle == "app_de":
            sterne = f"{s},0"
        else:
            sterne = rng.choice([f"{s}.0", "★" * s + "☆" * (5 - s)])
        produkt = rng.choices(["Nimbus Q2", "Nimbus Q2 ", "NIMBUS Q2", "Nimbus-Q2",
                               "Nimbus Q2 (Schwarz)"], weights=[70, 8, 8, 7, 7])[0]
        text = r["text"]
        if quelle == "web" and text.strip():
            if rng.random() < 0.25:   # altes Shop-System: Sonderzeichen als HTML-Entities
                for a, b in [("ä", "&auml;"), ("ö", "&ouml;"), ("ü", "&uuml;"), ("ß", "&szlig;"),
                             ("Ä", "&Auml;"), ("Ü", "&Uuml;"), ("–", "&ndash;")]:
                    text = text.replace(a, b)
            if rng.random() < 0.15:
                text = text.replace(". ", ".<br>", 1)
        if rng.random() < 0.05 and any(ch in text for ch in "äöüßÄÖÜ–"):
            text = _mojibake(text)
        if rng.random() < 0.2:
            text = rng.choice(["  ", " ", "\t"]) + text + rng.choice(["", " ", "  ", "\n"])
        zeilen.append({"review_id": r["review_id"], "quelle": quelle, "datum": datum,
                       "produkt": produkt, "sterne": sterne, "text": text})
        wahr.append({"review_id": r["review_id"], "datum_wahr": r["date"], "sterne_wahr": s,
                     "text_wahr": r["text"], "true_sentiment": r["true_sentiment"],
                     "is_sarcastic": r["is_sarcastic"], "is_fake": r["is_fake"],
                     "is_english": any(r["text"] == e for e, _ in ENGLISH),
                     "duplikat_von": None})
    # Die 3 Duplikate des Generators: gleicher Text, neue ID
    original = {(w["text_wahr"], w["datum_wahr"], w["sterne_wahr"]): w["review_id"]
                for w in wahr if w["review_id"] < f"R{300:04d}"}
    for w in wahr:
        if w["review_id"] >= f"R{300:04d}":
            w["duplikat_von"] = original[(w["text_wahr"], w["datum_wahr"], w["sterne_wahr"])]
    # 6 Beinahe-Duplikate: dieselbe Bewertung über eine zweite Quelle erneut exportiert
    kandidaten = [i for i, w in enumerate(wahr)
                  if len(w["text_wahr"]) > 25 and not w["is_fake"] and w["duplikat_von"] is None]
    for j, i in enumerate(rng.sample(kandidaten, 6)):
        z, w = dict(zeilen[i]), dict(wahr[i])
        t = " ".join(w["text_wahr"].split())
        if j < 4:
            t = [t.lower(), t.rstrip(".") + "!", t.replace(":", " -"), t + " "][j]
        else:   # Tippfehler: zwei Buchstaben in einem Wort vertauscht
            woerter = t.split()
            k = max(range(len(woerter)), key=lambda x: len(woerter[x]) if x % 2 == j % 2 else 0)
            wo = woerter[k]; woerter[k] = wo[:2] + wo[3] + wo[2] + wo[4:]
            t = " ".join(woerter)
        neu = f"A{j:04d}"
        z.update(review_id=neu, quelle="app_de", text=t,
                 datum=date.fromisoformat(w["datum_wahr"]).strftime("%d.%m.%Y"),
                 sterne="" if z["sterne"] in ("", "k.A.", "null") else f"{w['sterne_wahr']},0")
        w.update(review_id=neu, duplikat_von=wahr[i]["review_id"])
        zeilen.append(z); wahr.append(w)
    # 2 Zeilen doppelt, weil zwei Exporte überlappen: komplett identisch
    for i in rng.sample(range(len(basis)), 2):
        zeilen.append(dict(zeilen[i]))
    rng.shuffle(zeilen)
    roh = pd.DataFrame(zeilen)
    wahrheit = pd.DataFrame(wahr).set_index("review_id")
    return roh, wahrheit


# ---------------------------------------------------------------------------
# Sitzung 5: Bereinigter Datensatz (Pipeline aus Sitzung 4, ohne LLM)
# ---------------------------------------------------------------------------
def _normalisiere(t):
    t = t.casefold()
    t = re.sub(r"[^\w\s]", " ", t)
    return " ".join(t.split())


def _repariere_text(t):
    import html as _html
    import unicodedata
    t = _html.unescape(t)
    t = re.sub(r"<br\s*/?>", " ", t)
    if "Ã" in t or "â" in t:
        try:
            t = t.encode("latin-1").decode("utf-8")
        except UnicodeError:
            pass
    return " ".join(unicodedata.normalize("NFC", t).split())


def _parse_sterne(s):
    s = (s or "").strip()
    if s.lower() in {"", "k.a.", "null", "n/a", "-"}:
        return None
    if set(s) <= set("★☆"):
        n = s.count("★")
    else:
        m = re.match(r"^(\d+(?:[.,]\d+)?)", s)
        if not m:
            return None
        n = float(m.group(1).replace(",", "."))
    return int(n) if n == int(n) and 1 <= n <= 5 else None


def _ist_spam(t):
    signale = [r"www\.|https?://", r"\blink\b", r"\bcode\b|rabatt|gutschein", r"klick",
               r"!!!", r"\+\+\+", r"\bkauf(t|en)\b.*\bkauf"]
    buchstaben = [c for c in t if c.isalpha()]
    gross = sum(c.isupper() for c in buchstaben) / max(1, len(buchstaben))
    return any(re.search(p, t.lower()) for p in signale) or gross > 0.5


def lade_sauber():
    """Bereinigter Datensatz wie am Ende von Sitzung 4. Gibt (sauber, wahrheit) zurück."""
    import pandas as pd
    roh, wahrheit = export_roh()
    roh = roh[~roh.duplicated()].reset_index(drop=True)
    fmt = {"web": "%Y-%m-%d", "app_de": "%d.%m.%Y", "app_us": "%m/%d/%Y"}
    df = pd.DataFrame({
        "review_id": roh["review_id"],
        "quelle": roh["quelle"],
        "datum": [pd.to_datetime(d.strip()[:10] if q == "web" else d.strip(), format=fmt[q],
                                 errors="coerce") if d.strip() else pd.NaT
                  for d, q in zip(roh["datum"], roh["quelle"])],
        "sterne": pd.array([_parse_sterne(x) for x in roh["sterne"]], dtype="Int64"),
        "text": roh["text"].map(_repariere_text),
    })
    df["text_norm"] = df["text"].map(_normalisiere)
    weg = df.duplicated(subset=["text_norm", "datum", "sterne"]) & (df["text_norm"] != "")
    # Beinahe-Duplikate (Tippfehler): gleiche Metadaten + sehr ähnlicher Text
    sh = [{t[i:i + 3] for i in range(len(t) - 2)} for t in df["text_norm"]]
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if weg[j] or df.at[i, "datum"] != df.at[j, "datum"]:
                continue
            si, sj = df.at[i, "sterne"], df.at[j, "sterne"]
            if (pd.isna(si) or pd.isna(sj)) and not (pd.isna(si) and pd.isna(sj)):
                continue
            if not pd.isna(si) and si != sj:
                continue
            u = sh[i] | sh[j]
            if u and len(sh[i] & sh[j]) / len(u) >= 0.85:
                weg[j] = True
    muell = df["text_norm"].map(lambda n: len(n) < 3 or n in {"n a", "test", "kein kommentar", "keine angabe"})
    spam = df["text"].map(_ist_spam)
    sauber = df[~(weg | muell | spam)].drop(columns="text_norm").reset_index(drop=True)
    return sauber, wahrheit


# ---------------------------------------------------------------------------
# Sitzung 5: Themen-Katalog und Wahrheit
# ---------------------------------------------------------------------------
THEMEN = ["Klang", "Geräuschunterdrückung", "Akku & Laden", "Tragekomfort",
          "Verbindung", "Verarbeitung & Haltbarkeit", "Preis", "App & Bedienung"]

_PHRASE_THEMA = {
    "der klang ist hervorragend": ("Klang", "positiv"),
    "satte bässe": ("Klang", "positiv"),
    "der sound ist klar und ausgewogen": ("Klang", "positiv"),
    "der bass ist matschig": ("Klang", "negativ"),
    "die geräuschunterdrückung ist top": ("Geräuschunterdrückung", "positiv"),
    "die geräuschunterdrückung rauscht": ("Geräuschunterdrückung", "negativ"),
    "der akku hält den ganzen tag": ("Akku & Laden", "positiv"),
    "der akku ist nach einer stunde leer": ("Akku & Laden", "negativ"),
    "der rechte ohrhörer lädt nicht mehr": ("Akku & Laden", "negativ"),
    "sitzt super bequem": ("Tragekomfort", "positiv"),
    "die passform ist perfekt": ("Tragekomfort", "positiv"),
    "sie fallen leicht aus dem ohr": ("Tragekomfort", "negativ"),
    "bluetooth verbindet sofort": ("Verbindung", "positiv"),
    "die verbindung bricht ab": ("Verbindung", "negativ"),
    "top verarbeitet": ("Verarbeitung & Haltbarkeit", "positiv"),
    "das case wirkt billig": ("Verarbeitung & Haltbarkeit", "negativ"),
    "klasse für den preis": ("Preis", "positiv"),
    "viel zu teuer": ("Preis", "negativ"),
    "die app ist übersichtlich": ("App & Bedienung", "positiv"),
    "die app stürzt ständig ab": ("App & Bedienung", "negativ"),
    "die touch-steuerung reagiert kaum": ("App & Bedienung", "negativ"),
}
_SATZ_THEMEN = {
    SARCASTIC[0]: [("Verarbeitung & Haltbarkeit", "negativ")],
    SARCASTIC[1]: [("App & Bedienung", "negativ")],
    SARCASTIC[2]: [("Preis", "negativ"), ("Klang", "negativ")],
    SARCASTIC[3]: [("Akku & Laden", "negativ")],
    SARCASTIC[4]: [("Klang", "negativ")],
    SARCASTIC[5]: [("Tragekomfort", "negativ")],
    SARCASTIC[6]: [("Verbindung", "negativ")],
    SARCASTIC[7]: [("Verarbeitung & Haltbarkeit", "negativ")],
    SARCASTIC[8]: [("App & Bedienung", "negativ")],
    SARCASTIC[9]: [("Verarbeitung & Haltbarkeit", "negativ")],
    ENGLISH[0][0]: [("Klang", "positiv"), ("App & Bedienung", "negativ")],
    ENGLISH[1][0]: [("Akku & Laden", "positiv")],
    ENGLISH[2][0]: [("Verarbeitung & Haltbarkeit", "negativ")],
    ENGLISH[3][0]: [("Tragekomfort", "positiv"), ("Klang", "positiv")],
    ENGLISH[4][0]: [("Geräuschunterdrückung", "negativ"), ("Verarbeitung & Haltbarkeit", "negativ")],
}


def wahre_themen(text_wahr):
    """Liste von (thema, polaritaet) für einen Originaltext aus dem Generator."""
    if text_wahr in _SATZ_THEMEN:
        return list(_SATZ_THEMEN[text_wahr])
    t = text_wahr.lower()
    return [tp for phrase, tp in _PHRASE_THEMA.items() if phrase in t]


# Mock für Themen: zählt nur wenige, wörtliche Stichwörter (keine Synonyme, keine Verneinung)
_MOCK_STICHWORT = {"Klang": ["klang", "bass", "sound"], "Geräuschunterdrückung": ["geräuschunterdrückung"],
                   "Akku & Laden": ["akku", "lädt"], "Tragekomfort": ["bequem", "passform"],
                   "Verbindung": ["verbindung", "bluetooth"], "Verarbeitung & Haltbarkeit": ["verarbeitet", "case"],
                   "Preis": ["preis", "teuer"], "App & Bedienung": ["app", "touch"]}


def _mock_themen(text):
    t = text.lower()
    out = []
    for thema, woerter in _MOCK_STICHWORT.items():
        if any(w in t for w in woerter):
            neg = any(w in t for w in _NEG)
            pos = any(w in t for w in _POS)
            out.append({"thema": thema, "polaritaet": "negativ" if neg and not pos else "positiv"})
    return out


# ---------------------------------------------------------------------------
# Sitzung 5: Embeddings
# ---------------------------------------------------------------------------
EMBED_MODELL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_embedder = None
_tfidf = None


def embed(texte, modell=EMBED_MODELL):
    """Texte -> normierte Vektoren (numpy, eine Zeile pro Text).
    Nutzt ein Sentence-Transformer-Modell (lokal, kein API-Schlüssel, kostenlos).
    Ist das nicht verfügbar, Ersatz: TF-IDF auf Zeichen-Schnipseln (nur Schreibweise, keine Bedeutung)."""
    global _embedder, _tfidf
    texte = [str(t) for t in texte]
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer(modell)
            print(f"Embedding-Modell geladen: {modell}")
        except Exception as e:
            _embedder = "ersatz"
            print(f"⚠️ Sentence-Transformers nicht verfügbar ({type(e).__name__}). "
                  "Ersatz: TF-IDF auf Zeichen-Schnipseln. Ähnlichkeit = ähnliche Schreibweise, nicht Bedeutung.")
    if _embedder == "ersatz":
        from sklearn.feature_extraction.text import TfidfVectorizer
        if _tfidf is None:
            korpus = [w["text_wahr"] for _, w in export_roh()[1].iterrows()]
            _tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(korpus + texte)
        v = _tfidf.transform(texte).toarray()
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    return _embedder.encode(texte, normalize_embeddings=True, show_progress_bar=False)
