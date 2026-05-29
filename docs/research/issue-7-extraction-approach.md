# Issue #7 — Training-state extraction approach (research draft)

> DRAFT — autonomous research; verify all **VERIFY:**-tagged claims before relying on this.

**Scope.** This is the research that de-risks issues #11 (fetch `Document` nodes),
#12 (LLM extraction → `:Staged` facts), and #13 (review/promote) **before** any
production code is written. It contains **no production code or tests** — the
fixtures and prompt below are illustrative and live only in this document. Per
PRD §8.4, L4 inference about real (sometimes young) athletes is sensitive:
health/injury data is **private by default**, low-confidence inferences are
**dropped, not guessed**, and nothing leaves `:Staged` without human review.

Closed `TrainingSignal.kind` vocab this work must map to (PRD §8.4,
`src/climber_network/vocab.py`):

```
gym_session, volume_high, volume_taper, focus_lead, focus_boulder,
focus_speed, illness, travel_personal, return_from_injury, camp_attendance
```

Per-fact extraction shape: `{kind ∈ closed vocab, observed_at, location?, confidence}`.

---

## 1. Fetch-method comparison (≥2 source types)

The downstream `Document` schema (PRD §8.5) is `{url, published_at, raw_text/transcript, lang}`,
keyed `doc:{sha1(url)}` and linked `FROM_SOURCE` to a `Source` `{type, name, domain}`.
A fetch method is only useful to us if it yields **(a) a stable URL** (provenance +
idempotent `MERGE` key), **(b) a publication timestamp**, and **(c) enough text** to
extract a `kind` from. The third requirement is where the source types diverge most.

| Source type | Concrete option | Feasibility | Brittleness | ToS / legal | Cost | Yields body text? |
|---|---|---|---|---|---|---|
| **News (search API)** | **NewsAPI.org** Developer plan | High — keyword/handle query per athlete | Low (stable JSON) | **VERIFY:** dev plan is "development and testing in a development environment only … cannot be used in a staging or production environment" | **VERIFY:** $0, **100 req/day**, **24h article delay** | Partial — `title`+`description`+truncated `content` only |
| **News (open dataset API)** | **GDELT DOC 2.0** `ArtList` | High — no key, 65 languages, country filter | Low–medium (3-month rolling window only) | **VERIFY:** no key/auth or rate limit documented on the DOC 2.0 page | Free (**VERIFY:** no documented fees) | **No** — returns metadata only (`title`, `url`, `seendate`, `domain`, `language`, `sourcecountry`); **no article body** |
| **News (RSS feeds)** | Per-outlet RSS / Google News RSS | Medium — must curate feeds per outlet | Medium (feed formats vary; items expire) | Generally publisher-sanctioned for syndication; verify per feed | Free | Partial — usually summary, not full body |
| **YouTube transcript** | `youtube-transcript-api` (PyPI, MIT) | Medium locally | **High in cloud** | **VERIFY:** library README warns YouTube blocks cloud-provider IPs (AWS/GCP/Azure) → `RequestBlocked`/`IpBlocked`; needs residential proxies. Scraping captions sits in YouTube-ToS grey area | Library free; reliable cloud use implies paid proxy (**VERIFY:** Webshare) | Yes — full transcript text |
| **Blog scrape** | `httpx` + `trafilatura`/readability | Medium | High (per-site HTML, breaks on redesign) | Robots.txt + per-site ToS vary; case-by-case | Free | Yes — full body |

### Key finding: "news API" splits into *search* vs *body*

The single most important de-risking finding: **most news APIs (and GDELT in
particular) return article *metadata*, not the licensed full body text.** A live
fetch against GDELT DOC 2.0 `ArtList` returned objects with exactly these fields and
no body: `url, url_mobile, title, seendate, socialimage, domain, language, sourcecountry`.
NewsAPI.org returns `title` + `description` + a **truncated** `content` field, not the
full article. (**VERIFY:** the NewsAPI `content` truncation length — historically ~200
chars on lower tiers.)

Implication for #11/#12: the title + description/snippet is frequently *enough* signal
for our coarse closed vocab (e.g. a headline "Janja Garnbret out 6 weeks with finger
injury" already yields `return_from_injury`/`illness`-class facts). Where it is not
enough, the discovery API gives us a **canonical URL** that #11 can optionally
hydrate with a polite per-article `httpx` + `trafilatura` extract (respecting
`robots.txt`). So the recommended architecture is **two-stage**: *discover* via a
metadata API, *optionally hydrate* the body from the canonical URL.

### Recommendation — news-first concrete path

**Primary discovery: GDELT DOC 2.0 `ArtList` (JSON).** Rationale:

- **No API key and no documented hard rate limit** (**VERIFY:** against the DOC 2.0
  page) — removes the secret-management and quota friction that NewsAPI's 100 req/day +
  "dev only" restriction would impose on even a 10-athlete prototype.
- Multilingual + `sourcecountry` filtering fits an international World-Cup field.
- Stable JSON with `url`/`seendate`/`domain`/`language` maps **directly** onto the
  `Document`/`Source` schema with no parsing acrobatics.
- Endpoint shape (illustrative): `https://api.gdeltproject.org/api/v2/doc/doc?query="<athlete name> climbing"&mode=ArtList&format=json&timespan=3months&sourcelang=eng`

**Body hydration (only when the snippet is insufficient): per-URL `trafilatura` extract**,
gated by `robots.txt`, with the original publisher URL preserved as provenance.

**Keep as a fallback, not primary: NewsAPI.org.** It is a clean managed API but the
Developer-plan "no production" clause and 100 req/day/24h delay make it awkward for a
recurring ingest even at 10 athletes (**VERIFY:** both terms against the live pricing
page before relying on a paid tier). Its `NEWS_API_KEY` env getter already exists in
`config.py`, so swapping it in later is cheap.

**Defer YouTube + blog scraping** to after news proves out (matches PRD N5 / decision
§3.3: "no at-scale social scraping in v1"). YouTube's cloud-IP blocking makes it a poor
*first* target for a Vercel/cloud-deployed pipeline.

---

## 2. Extraction prompt (full, inline)

Groq is already a dependency and is called in `api/rag.py` with
`model="llama-3.1-8b-instant"`. Two relevant Groq facts: **(a)** `response_format`
`json_object` ("JSON mode") works on all models but only validates JSON *syntax*, and
**requires the schema be described in the prompt**; **(b)** strict schema-enforced
`json_schema` (constrained decoding) is currently **VERIFY:** limited to GPT-OSS 20B/120B,
*not* `llama-3.1-8b-instant`. So for the Llama path we use `json_object` + an explicit
in-prompt schema + a Python-side validator (vocab gate + confidence threshold). The
validator — not the model — is the safety boundary.

Recommended call settings: `temperature=0`, `response_format={"type":"json_object"}`,
and a `prompt_version` string stored on the `ExtractionRun` node (PRD §8.5) so a prompt
change is auditable.

### System prompt

```text
You are a precise information-extraction system for competition-climbing analysis.
You read ONE public document (a news article, headline+snippet, or video caption)
about ONE named athlete and extract only explicitly-supported training-state facts.

Output ONLY a single JSON object, no prose, matching exactly this schema:

{
  "facts": [
    {
      "kind": <one of the allowed kinds below, or null>,
      "observed_at": <ISO-8601 date "YYYY-MM-DD" the fact refers to, or null>,
      "location": <free-text place name explicitly mentioned, or null>,
      "confidence": <number 0.0–1.0>,
      "evidence_quote": <the verbatim span (<=160 chars) supporting this fact>
    }
  ]
}

Allowed "kind" values (you MUST use one of these EXACTLY, or null):
- gym_session        : a routine indoor training session
- volume_high        : a heavy / high-volume training block
- volume_taper       : deliberately reducing load before a competition (taper, rest)
- focus_lead         : training emphasis on lead climbing
- focus_boulder      : training emphasis on bouldering
- focus_speed        : training emphasis on speed climbing
- illness            : sick / ill / unwell (NOT a musculoskeletal injury)
- travel_personal    : non-competition personal travel (holiday, visiting home)
- return_from_injury : coming back to training/competition after an injury
- camp_attendance    : attending a training camp / training trip with others

RULES:
1. Extract a fact ONLY if the document explicitly states or near-directly implies it.
   Do NOT infer training state from competition results alone.
2. If the document supports NO allowed kind, return {"facts": []}. Do not invent facts.
3. If a statement is about a body-part injury itself (not the return), set kind null
   for that span (an InjuryEvent is handled separately downstream) — do not force-fit
   it into the training vocab.
4. "confidence" reflects how directly the text supports the kind:
   1.0 = explicit first-person/quoted statement; 0.6 = clearly implied; <0.5 = guess.
   When unsure, LOWER the confidence rather than dropping the fact.
5. "observed_at" is the date the activity happened if stated; else null. Do NOT
   substitute the article's publication date — leave null if the activity date is
   unstated.
6. "location" only if a place is explicitly named in the text; else null.
7. Never output any key not in the schema. Never output text outside the JSON object.
```

### User message template

```text
ATHLETE: {athlete_name}
PUBLISHED_AT: {published_at_iso}      # context only; do NOT use as observed_at
SOURCE_DOMAIN: {domain}
DOCUMENT:
"""
{document_text}
"""
Return the JSON object now.
```

### Refusal / `null` handling

- **Empty result is valid:** `{"facts": []}` when nothing maps. The parser treats this
  as a successful extraction with zero facts (an `ExtractionRun` is still recorded).
- **`kind: null`** on a span → that fact is **dropped** before any write (it is not in
  the closed vocab).
- **`observed_at: null` is allowed.** A `TrainingSignal` may have no resolved date; we
  never back-fill the publication date as the activity date (rule 5) to avoid fabricating
  precision. Downstream may still stage it; reviewers see the null.
- **Malformed / non-JSON output** → the whole document's run fails closed (no facts
  staged), is logged on the `ExtractionRun`, and retried at most once. JSON-mode makes
  this rare but the validator must never assume well-formed output.

---

## 3. Example offline fixtures (inline only — NOT added to the tree)

These illustrate the parser contract for #12. They are shown here only; the actual
checked-in fixtures under `tests/fixtures/` are issue #12's deliverable.

### Fixture A — news article with two extractable signals

Input (a realistic GDELT/news-style document record):

```json
{
  "athlete_name": "Jakob Schubert",
  "document": {
    "url": "https://www.example-climbing-news.com/schubert-innsbruck-camp",
    "published_at": "2026-04-18",
    "domain": "example-climbing-news.com",
    "lang": "en",
    "raw_text": "After three weeks off recovering from a strained finger pulley, Jakob Schubert says he is finally back to full training. 'I joined the national team's lead camp in Innsbruck last week and the finger held up,' he said. He is now tapering ahead of the Wujiang World Cup."
  }
}
```

Expected extracted JSON (model output, pre-validation):

```json
{
  "facts": [
    {
      "kind": "return_from_injury",
      "observed_at": null,
      "location": null,
      "confidence": 0.9,
      "evidence_quote": "after three weeks off recovering from a strained finger pulley ... finally back to full training"
    },
    {
      "kind": "camp_attendance",
      "observed_at": null,
      "location": "Innsbruck",
      "confidence": 0.85,
      "evidence_quote": "I joined the national team's lead camp in Innsbruck last week"
    },
    {
      "kind": "focus_lead",
      "observed_at": null,
      "location": "Innsbruck",
      "confidence": 0.6,
      "evidence_quote": "the national team's lead camp"
    },
    {
      "kind": "volume_taper",
      "observed_at": null,
      "location": null,
      "confidence": 0.8,
      "evidence_quote": "now tapering ahead of the Wujiang World Cup"
    }
  ]
}
```

After validation (§4): all four pass the 0.55 threshold and are in-vocab, so all four
are staged. `camp_attendance` + `focus_lead` + `return_from_injury` carry
`location:"Innsbruck"` for `SIGNAL_AT` resolution; the injury *itself* (strained finger
pulley) is the seed for a separate `InjuryEvent` and is **private by default**.

### Fixture B — document with no extractable training signal (refusal)

Input:

```json
{
  "athlete_name": "Janja Garnbret",
  "document": {
    "url": "https://www.example-sport.com/garnbret-wins-gold",
    "published_at": "2026-05-10",
    "domain": "example-sport.com",
    "lang": "en",
    "raw_text": "Janja Garnbret won gold in the women's boulder final on Saturday, topping all four finals problems to take her third World Cup title of the season."
  }
}
```

Expected extracted JSON:

```json
{ "facts": [] }
```

This is a pure *result* report. Per rule 1, training state is **not** inferred from
results, so the correct output is the empty set — no facts are staged, but the
`ExtractionRun` and the `Document` are still recorded.

---

## 4. Mapping rules: raw extraction → closed `kind` vocab

The model emits a candidate `kind`; the **Python validator is the authority** and must
re-check every fact. Mapping is deliberately closed and conservative.

**Allowed-set gate.** `kind` must be `in` the frozenset that mirrors
`src/climber_network/vocab.py`. Anything else (including `null`, casing variants,
synonyms the model invented) is **dropped**. Suggested normalization *before* the gate:
lowercase + strip, and a tiny synonym map for near-misses (e.g. `"sick"→illness`,
`"rest"`/`"deload"→volume_taper`, `"comeback"→return_from_injury`). Keep the synonym map
**short and explicit**; an unknown token is dropped, never coerced.

**Confidence thresholds.**

| Band | Action |
|---|---|
| `confidence ≥ 0.85` | stage normally |
| `0.55 ≤ confidence < 0.85` | stage, flagged for closer reviewer attention |
| `confidence < 0.55` | **drop** (PRD §8.4 "low-confidence … dropped rather than guessed") |

**VERIFY:** the **0.55** drop threshold is a proposed starting value, not measured.
Calibrate it against the labelled fixture set in #12 (precision/recall on the seed
athletes) before trusting it; PRD fixes only the *principle* (drop low-confidence), not
a number.

**Node-type routing.** The closed `kind` vocab populates `TrainingSignal`. Two facts
route elsewhere and are **private by default**:

- An injury described as an injury (body part / status) → `InjuryEvent`
  (`{body_part, onset_date, status}`), not a `TrainingSignal`. `return_from_injury` is
  the only injury-adjacent member of the training vocab and stays a `TrainingSignal`.
- A camp with start/end/country → `TrainingCamp`, with `camp_attendance` as the linking
  `TrainingSignal`/`ATTENDED` edge.

**Location resolution.** `location` is free text from the model. Resolving it to a
`Country`/`Venue` for the `SIGNAL_AT` edge is a *downstream* geocoding step (reuse L2
patterns); unresolved locations leave the `SIGNAL_AT` edge absent rather than guessing.

---

## 5. Provenance + `:Staged` flow (confirms PRD §8.5 / ties to #13)

The schema already supports the full chain — `vocab.py` defines the labels
(`Source`, `Document`, `ExtractionRun`, `TrainingSignal`, `InjuryEvent`, `TrainingCamp`)
and rel types (`EVIDENCED_BY`, `FROM_SOURCE`, `EXTRACTED_BY`, `HAS_SIGNAL`, `HAD_INJURY`,
`ATTENDED`, `SIGNAL_AT`), and the id builders (`sig`, `inj`, `camp`, `doc`, `source_id`,
`run`). No vocab change is needed for #11–#13.

**What gets stored per extraction run.**

1. `Source` `{type:"news", name, domain}` keyed `src:{domain}` — `MERGE` (idempotent).
2. `Document` `{url, published_at, raw_text, lang}` keyed `doc:{sha1(url)}` — `MERGE`.
   (#11 writes these two; #12 consumes them.)
3. `ExtractionRun` `{model, prompt_version, timestamp}` keyed `run:{iso8601_ts}`.
4. For each surviving fact: a `TrainingSignal`/`InjuryEvent`/`TrainingCamp` written
   **with a `:Staged` label**, plus the mandatory edges:

```
(Athlete)-[:HAS_SIGNAL]->(:TrainingSignal:Staged)
(:TrainingSignal:Staged)-[:EVIDENCED_BY]->(:Document)-[:FROM_SOURCE]->(:Source)
(:TrainingSignal:Staged)-[:EXTRACTED_BY]->(:ExtractionRun)
(:TrainingSignal:Staged)-[:SIGNAL_AT]->(:Country|:Venue)   // only when location resolves
```

**Invariant (PRD §8.5, enforced in `extract.py` per #12):** no L4 fact is written
unless its `EVIDENCED_BY → Document` (and `FROM_SOURCE`, `EXTRACTED_BY`) edges are
created in the same transaction. A fact with no evidencing `Document` must be impossible
by construction — #12's acceptance test asserts this.

**Review / promote (issue #13).** Facts sit under `:Staged` (quarantine) and are never
auto-promoted. A human reviewer hits the gated `POST /ingest/approve` (bearer
`INGEST_API_KEY`), and `promote.py` **strips the `:Staged` label** for approved node ids
while preserving all provenance edges (so provenance survives promotion). Both
`/ingest` and `/ingest/approve` return 404/503 when `INGEST_API_KEY` is unset (cloud is
disabled). Imputed health/injury stays out of public read endpoints (private by default).

This confirms the design holds end-to-end: **discover (#11) → extract to `:Staged` with
provenance (#12) → human approve/promote (#13)**, with the closed vocab + confidence
gate + provenance invariant as the three safety boundaries.

---

## 6. Go / No-go recommendation (P5 prototype, ~10 seed athletes)

**GO** — for a `:Staged`-only, human-gated prototype on ~10 hand-picked athletes, news-first.

Why it is low-risk to start:

- **Fetch is solved** for news: GDELT DOC 2.0 `ArtList` gives keyed, timestamped,
  multilingual `Document`s with no API key (**VERIFY:** rate limit), with `trafilatura`
  body-hydration as the escape hatch when snippets are too thin. NewsAPI.org is a clean
  managed fallback (key getter already exists).
- **Extraction is feasible** on the existing Groq Llama path using `json_object` mode +
  an explicit in-prompt schema + a Python vocab/confidence validator. We do not depend
  on strict `json_schema` (which Llama-3.1-8b lacks — **VERIFY:**).
- **The graph already models everything** — labels, rels, and id builders for the whole
  provenance chain exist in `vocab.py`; `:Staged` + the §8.5 invariant + the gated
  `/ingest/approve` promote give the required human-in-the-loop and quarantine.
- **The blast radius is contained:** staged facts cannot contaminate the trusted graph,
  health/injury is private by default, and low-confidence/out-of-vocab is dropped.

Conditions / what to validate during the prototype (the open risks):

1. **VERIFY: GDELT DOC 2.0 rate-limit / fair-use** for repeated automated polling before
   relying on it as the production discovery source; have NewsAPI ready as fallback.
2. **VERIFY: news full-text licensing** — storing full `raw_text` vs. storing snippet +
   canonical URL. The conservative default for the prototype is **store snippet/title +
   URL** (sufficient for our coarse vocab) and treat full-body hydration as opt-in per
   source. This is analysis of public competitive context, not redistribution — *appears
   to* fall under research/analysis use, but **verify with counsel** before storing/serving
   full article bodies. (Not a binding legal determination.)
3. **Calibrate the confidence threshold** (0.55 proposed) against a small hand-labelled
   fixture set; measure false-positive rate on the `illness`/`return_from_injury` classes
   specifically, since those are the most sensitive (PRD §8.4 responsible-use).
4. **Athlete↔document disambiguation** (common names) — the hand-curated seed map in #11
   (`athlete_handles.yaml`) plus per-athlete query tuning, with staged review as the
   backstop, is the mitigation; watch precision during the prototype.

**No-go triggers** (defer/stop if hit): GDELT rate-limiting makes polling unworkable
*and* NewsAPI's "dev-only" terms block our use; or the confidence calibration cannot get
the sensitive-class false-positive rate to an acceptable level under review. Neither is
expected for a 10-athlete, human-gated, news-only prototype.

---

## Sources

- NewsAPI.org pricing (plans, 100 req/day, 24h delay, dev-only restriction): https://newsapi.org/pricing
- GDELT DOC 2.0 API debut (no-key full-text news search, ArtList, timespan, sourcecountry/sourcelang, 75→250 results): https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
- GDELT DOC 2.0 live `ArtList` JSON fields (url, url_mobile, title, seendate, socialimage, domain, language, sourcecountry): https://api.gdeltproject.org/api/v2/doc/doc
- Groq Structured Outputs / JSON mode (json_object vs json_schema; strict limited to GPT-OSS; json_object needs schema in prompt): https://console.groq.com/docs/structured-outputs
- Groq Llama 3.1 8B model card: https://console.groq.com/docs/model/llama-3.1-8b-instant
- youtube-transcript-api (MIT; no key; cloud-IP blocking → RequestBlocked/IpBlocked; residential proxies): https://github.com/jdepoix/youtube-transcript-api
- youtube-transcript-api cloud-block issue thread: https://github.com/jdepoix/youtube-transcript-api/issues/511
- U.S. Copyright Office Fair Use Index (research/analysis context; not legal advice): https://www.copyright.gov/fair-use/
- Fair use of full-text searchable databases (HathiTrust, Harvard JOLT digest): https://jolt.law.harvard.edu/digest/creating-full-text-searchable-database-of-copyrighted-works-is-fair-use
