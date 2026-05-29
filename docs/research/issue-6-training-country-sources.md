# Issue #6 — Sources for athlete training country / home base + document candidates

> DRAFT — autonomous research; verify all **VERIFY:**-tagged claims before relying on this.

**Status:** research draft, non-binding · **Issue:** [#6](../../) ("L4 research:
determine sources for athlete training country & profile signals") · **Blocks:**
P5 ingest (issues #11–#13) · **Author note:** produced by an unattended research
pass; recommendations require human review before any code is written.

---

## 0. Why this doc exists

Today the graph's `BASED_IN` edge is a **nationality proxy** — emitted in
`sync/geo.py` as `(Athlete)-[:BASED_IN {source:'nationality_proxy'}]->(Country)`,
derived purely from `athletes.nationality` (see `sync/geo.py:486–509` and PRD §8.2).
That is wrong for the growing population of athletes who train outside their
country of citizenship (e.g. expatriate climbers in Innsbruck, Salt Lake City,
Tokyo training hubs). The L3 travel/circadian model (`sync/travel.py`) consumes
`BASED_IN` as the home-base origin for the first travel leg, so a wrong base
propagates into `TravelLeg` / `RestednessState` confidence.

This doc decides **where a real `BASED_IN` should come from**, and catalogues the
**document sources** for downstream training-state inference (the closed
`TrainingSignal.kind` vocabulary in PRD §8.4), before any ingest code is written.
Scope, per project decision §3.3: **news-first**, **~10 hand-picked athletes**.

The relevant graph vocabulary already exists in `src/climber_network/vocab.py`
(`BASED_IN`, `Source`, `Document`, `ExtractionRun`, `TrainingSignal`, …) and the
provenance ID builders (`source_id`, `doc`, `run`) — this research feeds the
seed-map and extraction work, not the schema.

---

## 1. Candidate sources for training country / home base

Five candidates were assessed. The first three are the primary contenders;
national-federation rosters and the existing nationality field are included for
completeness.

### 1.1 IFSC athlete profiles (worldclimbing.com)

The IFSC migrated its public site from `ifsc-climbing.org` to
`worldclimbing.com` (a `301` redirect was observed on fetch). Profiles live at
`https://www.worldclimbing.com/athlete/<id>/<slug>`.

I fetched two real profiles to confirm the fields actually shown:

- **Merritt Ernsberger** (id 1908): `Age: 26`, `City: Plano`, `Active since: 2017`,
  country `USA`, `Club/Team: USA Climbing`.
- **Tamara Ulzhabayeva** (id 3357): `Age: 38`, `City: Shymkent`, `Active since: 2001`,
  country `KAZ` ("Mountaineering and Sport Climbing Federation of Republic of
  Kazakhstan"). No `Club/Team` field present.

**Key observation:** profiles carry a **`City:` field**. For these two athletes
the city looks like a **birthplace / origin city**, not a current training base
(Plano TX for a US athlete; Shymkent for a Kazakh athlete). **VERIFY:** the
semantic meaning of the IFSC `City:` field (hometown vs. residence vs. training
base) is **not documented on the profile** and must be confirmed before use —
treating it as a training base would be an unverified assumption.

| Dimension | Assessment |
|---|---|
| **Coverage** | High for the population we care about — every World-Cup competitor has an IFSC profile (the upstream `athletes` table is itself IFSC-derived, PRD §6.1). The `City:` field is populated on both sampled profiles. **VERIFY:** population-wide fill rate of `City:`. |
| **Freshness** | Federation-maintained; **VERIFY:** update cadence — likely lags real moves (origin cities rarely change). |
| **Reliability** | Authoritative for identity/nationality; `City:` semantics ambiguous (see above). Best treated as *origin*, not *training base*. |
| **Access method** | HTML scrape of profile pages (no documented public JSON API on the official site — third parties like DataSports Group sell an API; `ifsc.results.info` / `outofiso.com` expose results, not bios). The numeric IFSC id is the natural join key. **VERIFY:** whether a stable JSON endpoint backs the profile page. |
| **ToS / compliance** | `worldclimbing.com/robots.txt` (fetched) disallows only `/assets/`, `/_showcase/`, `/_libraries/`, `/errors/`, `/test`, `/template` — it does **not** disallow `/athlete/`. So crawling profiles is not robots-excluded. **VERIFY:** the site's Terms of Use page for any scraping / reuse clause (not located in this pass). Phrase as: scraping profile pages *appears* permissible under robots.txt; verify the full ToS with counsel before automated collection. |

### 1.2 Wikidata (query.wikidata.org)

Structured, CC0-licensed. Each climber is an item; relevant properties:
`residence` (**P551**), `place of birth` (**P19**), `IFSC climber ID` (**P3690**),
plus qualifiers on P551 (start/end time, country P17, coordinate P625).

I ran live SPARQL counts against the public endpoint to get **real coverage**
(occupation = *sport climber*, `wd:Q11481802`):

| Metric | Count | Of 5,243 sport-climber items |
|---|---:|---:|
| Items with occupation *sport climber* (Q11481802) | 5,243 | — |
| …with `residence` (P551) | 173 | ~3.3% |
| …with `place of birth` (P19) | 3,374 | ~64% |
| …with English Wikipedia sitelink | 1,160 | ~22% |
| `IFSC climber ID` (P3690) — **total across all Wikidata** | 572 | (not occupation-scoped) |

Spot-check (verified live): **Adam Ondra** = `Q350568`, with `P3690 = 1364` and
`P551 = Brno`. So the join key and residence both exist for top athletes, but…

**Key observation:** P551 (residence) is **extremely sparse** (~3.3% of climbers).
Place-of-birth is well populated (~64%) but that is an *origin*, not a training
base. The IFSC-id crosswalk (P3690) exists but is thin (572 items total) and was
**0** under the strict `sport climber` occupation filter — **VERIFY:** this is
almost certainly because many climbers carry occupation *rock climber*
(`Q11341457`) or *climber* instead; re-run the coverage query across the union of
climbing occupations before relying on these exact percentages.

| Dimension | Assessment |
|---|---|
| **Coverage** | Good for *birthplace* (~64%), poor for *residence* (~3.3%). Best for well-known athletes; long tail of World-Cup competitors absent. |
| **Freshness** | Crowd-edited; residence updates are sporadic and unverified. **VERIFY:** typical edit recency on P551 for athlete items. |
| **Reliability** | Mixed — community-sourced, may lack references. P551 statements *can* carry a reference + start/end qualifiers (useful for confidence + recency), but most do not. |
| **Access method** | Public SPARQL at `https://query.wikidata.org/sparql` (GET/POST, JSON via `Accept: application/sparql-results+json`); REST API; full dumps at `dumps.wikimedia.org`. Used here successfully. |
| **ToS / compliance** | Data is **CC0** ("No rights reserved", per *Wikidata:Data access* and *SPARQL query service/Copyright*, fetched) — attribution appreciated, not required. Operational rules (fetched from *Wikidata:Data access*): send a descriptive **User-Agent** (per the Wikimedia User-Agent policy); on **`429 Too Many Requests`** back off and honor `Retry-After`; set the lowest sensible query timeout. This is the **most permissive** source legally. |

### 1.3 News API (Google News RSS — primary) / NewsAPI.org (alt)

Not a structured "home base" field, but free-text news frequently states where an
athlete trains/lives ("…now based in Innsbruck", "training out of Salt Lake
City"). This is the same pipeline that feeds `TrainingSignal` extraction, so it
doubles as a training-base evidence source via the LLM extractor.

**Google News RSS** (fetched the NewsCatcher parameter reference):

- Base: `https://news.google.com/rss/search`
- Params: `q` (query), `hl` (language e.g. `en-US`), `gl` (country e.g. `US`),
  `ceid` (e.g. `US:en`). Operators: implicit AND, `OR`, exact-phrase quotes,
  `intitle:`, `when:<n>h|d|m`, `after:`/`before:` (YYYY-MM-DD).
- Example: `https://news.google.com/rss/search?q=%22Janja+Garnbret%22+training+when:30d&hl=en-US&gl=US&ceid=US:en`
- No API key, no documented hard cap. **VERIFY:** Google does **not** publish
  official docs for these parameters (the reference above is third-party,
  reverse-engineered) — treat parameter behavior as best-effort, and **VERIFY**
  Google's Terms of Service for programmatic RSS use before production
  (third-party summary suggests personal/non-commercial RSS consumption is
  tolerated; commercial reuse needs review — *verify with counsel*).

**NewsAPI.org** (fetched the pricing page) as a structured alternative — quoted
limits from the **Developer (free)** plan: "100 requests per day", "No extra
requests available", "Articles have a 24 hour delay", "Search articles up to a
month old", and it "may be used for development and testing in a development
environment only, and cannot be used in a staging or production environment
(including internally)." Paid tiers (Business $449/mo, Advanced $1,749/mo) lift
the delay and extend history to 5 years. **The free tier's dev-only restriction
makes it unsuitable for the deployed pipeline**; usable for local prototyping.

| Dimension | Google News RSS | NewsAPI.org (free) |
|---|---|---|
| **Coverage** | Broad, multilingual; depends on press attention (top athletes well covered, long tail thin) | Similar breadth; **24h delay**, **1-month** window on free tier |
| **Freshness** | Near-real-time | 24h delayed (free) |
| **Reliability** | Headlines/snippets only — full text needs a follow-up fetch; home-base claims are journalistic, not authoritative | Same |
| **Access** | Keyless RSS/XML | API key, JSON |
| **ToS** | **VERIFY** Google ToS; no official param docs | Quoted: dev-environment only on free tier |

### 1.4 National-federation team rosters (e.g. USA Climbing, DAV, FFME)

Federations publish national-team rosters (confirmed: USA Climbing
`usaclimbing.org/team-rosters/` lists current national-team members by name).

| Dimension | Assessment |
|---|---|
| **Coverage** | Only national-team athletes of federations that publish rosters; inconsistent across the ~40+ federations sending World-Cup athletes. **VERIFY:** which federations publish machine-readable rosters. |
| **Freshness** | Seasonal (team named per cycle) — good for *which country selects them*, i.e. nationality, **not** training base. |
| **Reliability** | Authoritative for team membership; rarely states a training base/home gym. |
| **Access** | Per-federation HTML scraping; no common schema; many non-English. High engineering cost, low marginal signal over IFSC nationality. |
| **ToS** | Per-site; **VERIFY** each federation's terms individually. |
| **Verdict** | **Not worth it for v1** — adds nationality we already have, not a training base. Revisit only if a specific federation cleanly publishes home-gym/club data. |

### 1.5 (Baseline) `athletes.nationality` — the current proxy

Already in L1 (`source/pg.py:53`), 100% coverage for athletes that have it, but by
definition **citizenship, not training base**. This is the status quo we are
trying to improve on; it remains the ultimate fallback.

---

## 2. Recommendation: PRIMARY + FALLBACK for a real `BASED_IN`

The honest finding from §1 is that **no single source reliably gives a current
training base across the whole World-Cup field.** Origin (birthplace/origin city)
is well-covered; *current training base* is not. The recommended design treats
`BASED_IN` as a **layered, confidence-stamped** edge that prefers the strongest
evidence available per athlete and otherwise degrades gracefully — consistent
with PRD §8.2 (`BASED_IN {source:'imputed', confidence:…}` overwrites the
nationality proxy).

**Recommended precedence (highest → lowest confidence):**

1. **PRIMARY — News/LLM-extracted training base** (`source:'imputed_news'`),
   *only* when a fetched `Document` explicitly states a current base and the
   extractor returns high confidence. Must carry `EVIDENCED_BY` a `Document`
   (PRD §8.5 invariant). Realistic confidence: **0.6–0.85** when an article
   states it plainly and recently; lower otherwise. Coverage: **partial** — only
   athletes with press coverage of a move/base.
2. **FALLBACK A — Wikidata `residence` (P551)** (`source:'wikidata_p551'`),
   preferring statements with a reference and recent `start time` qualifier.
   Realistic confidence: **~0.5–0.7**. Coverage: **low (~3.3%)** — high-profile
   athletes mainly.
3. **FALLBACK B — Origin city** (Wikidata P19 ~64%, or IFSC `City:` field),
   stamped honestly as origin, **not** training base
   (`source:'origin_proxy'`, confidence **~0.3–0.4**). Better than nothing for
   the travel-leg origin, and clearly weaker than a real base.
4. **FALLBACK C — nationality proxy** (existing
   `source:'nationality_proxy'`, confidence **~0.2**) — unchanged baseline when
   nothing better exists. L3 already lowers `TravelLeg.confidence` for a
   nationality-proxy origin (PRD §8.3), so this composes cleanly.

**Confidence expectation overall:** for the seed set of ~10 hand-picked athletes,
expect a *real* (news- or Wikidata-backed) base for **maybe half**; the rest fall
to origin/nationality. That is acceptable for a prototype — the value is proving
the **provenance-stamped, confidence-graded** mechanism, not full coverage.

**Why not IFSC `City:` as primary:** its semantics are undocumented (§1.1) and on
both samples it looked like origin, not a training base; promoting it to a real
`BASED_IN` would assert something unverified. Use it only as an origin-proxy
fallback after confirming what the field means. **VERIFY** before any such use.

---

## 3. Athlete → handle matching strategy + `data/athlete_handles.yaml` schema

### 3.1 The matching problem

The upstream `athletes` table gives us `id` (int PK) + `name` + `nationality`
(`source/pg.py`). To fetch documents we need, per athlete, a **news query string**
(and optionally social handles, deferred). Names are ambiguous (transliteration,
diacritics, common names), so matching is done **manually for the seed set** — no
automated resolution in v1 (decision §3.3). The graph athlete id is
`vocab.ath(db_id)` → `ath:<db_id>`; the YAML keys off the raw upstream
`athletes.id` so it joins directly to L1.

**Optional crosswalk aid (not required):** Wikidata can map a name → QID → IFSC
climber ID (P3690) and English-Wikipedia title, which helps a human disambiguate.
But P3690 is sparse (572 items total, §1.2), so this is a *hint*, not a key.

### 3.2 Selection method for the ~10 seed athletes

Curation criteria (for the human curator — selection is **not** done here):

1. **High press coverage** — maximizes the chance news states a training base /
   training-state signal (favors recent World-Cup medalists / Olympians).
2. **Discipline + nationality spread** — boulder/lead/speed and several
   countries, so the prototype isn't overfit to one media market.
3. **Known training-base story** — at least a few athletes who *publicly* train
   outside their nationality country (the exact case `BASED_IN` should fix).
4. **Unambiguous name** — prefer names that return clean news results; avoid very
   common names for the first pass.
5. **Active in the L1 dataset** — must exist in `athletes` with a real `id`.

The curator records the chosen `athlete_db_id` values and authors the YAML;
issue #11 builds the full file from this schema.

### 3.3 Proposed schema for `data/athlete_handles.yaml`

> Note: `.gitignore` ignores `data/**` but **explicitly un-ignores**
> `data/athlete_handles.yaml` (`!data/athlete_handles.yaml`), so this seed map is
> intended to be committed. Keep it free of secrets (it is only public query
> terms / public handles).

Top-level map: `athlete_db_id` (the upstream `athletes.id`, integer) → object.

| Field | Type | Req? | Definition |
|---|---|---|---|
| `name` | string | yes | Display name, ideally matching `athletes.name`; for human readability + sanity-check against L1. |
| `news_query` | string | yes | Exact query string passed to the news source (may include quotes / `OR` / native-language alias). The athlete→news join. |
| `handles` | map | no | Optional, **deferred** social handles, keyed by platform. `instagram` / `youtube` map to public handle/channel strings. Present in schema so #11/#12 don't need a migration; unused while news-first. |
| `wikidata_qid` | string | no | Optional `Q…` id to enable the Wikidata residence/IFSC-id crosswalk (§1.2). |
| `ifsc_id` | integer | no | Optional IFSC climber id (profile URL / P3690) to join to IFSC profiles. |
| `notes` | string | no | Free-text curator notes (e.g. "trains in Innsbruck per 2025 interview — confirm"). Never authoritative. |

**Field rules / conventions:**

- Keys are integers matching `athletes.id`; the loader maps to `vocab.ath(id)`.
- `news_query` is the only *required* fetch input; everything else is an aid.
- `handles` are **public** account names only; **no tokens, cookies, or private
  data** ever live here (consistent with `SECURITY.md` secrets policy).
- Idempotent downstream: the same YAML re-run must produce the same `Document` /
  `Source` MERGEs.

### 3.4 Example rows (FICTIONAL — illustrate shape only; do NOT treat as real)

```yaml
# data/athlete_handles.yaml  (SCHEMA EXAMPLE)
# !! The two rows below are FICTIONAL placeholders to show the shape.
# !! They are NOT real athletes' real handles. Human curators MUST replace
# !! every value with verified data for the actual ~10 selected athletes.

99001:                                   # FICTIONAL athlete_db_id
  name: "Fictional Example Athlete A"
  news_query: '"Fictional Example Athlete A" climbing'
  wikidata_qid: "Q000000"                # placeholder, not real
  ifsc_id: 999999                         # placeholder, not real
  handles:
    instagram: "fictional_handle_a"       # placeholder, not real
  notes: "FICTIONAL example row — replace before use."

99002:                                   # FICTIONAL athlete_db_id
  name: "Fictional Example Athlete B"
  news_query: '"Fictional Example Athlete B" OR "Example B climber"'
  notes: "FICTIONAL example row — minimal form (news-only)."
```

---

## 4. PROPOSED legal / ToS notes — *for human review*

> **PROPOSED — FOR HUMAN REVIEW. NOT a legal determination. Does NOT amend
> `SECURITY.md` (left untouched per task constraint).** A human + counsel should
> review and, if accepted, decide where this lands. Everything below is "appears
> to / verify".

- **Public-data only.** Only data that is publicly published is read (IFSC public
  profiles, Wikidata CC0 dumps/SPARQL, public news). No authentication-walled,
  private, or paywalled content is accessed or stored.
- **Responsible use (carry forward PRD §8.4 / risk table §10).** L4 infers
  sensitive facts (injuries, illness, whereabouts) about real, sometimes young
  athletes. Even from public sources, inference can be wrong or intrusive.
  Therefore: imputed health/injury data stays **private-by-default** (not exposed
  via the public API — already in `SECURITY.md`); every fact carries
  `confidence` + provenance; low-confidence/unverifiable inferences are
  **dropped, not guessed**; nothing leaves `:Staged` without human review.
- **Wikidata.** Data is **CC0** (verified: *Wikidata:Data access*, *SPARQL query
  service/Copyright*) — reuse appears unrestricted; attribution appreciated. Send
  a descriptive User-Agent; honor `429` + `Retry-After`; cap query timeouts.
  *Appears* fully compliant for our use; verify the User-Agent policy text.
- **IFSC / worldclimbing.com.** `robots.txt` (verified) does not disallow
  `/athlete/`. Crawling profile pages *appears* permissible under robots.txt, but
  the site's **Terms of Use were not located in this pass** — **VERIFY** before
  automated collection; throttle politely; cache to avoid re-fetching.
- **Google News RSS.** No official parameter docs; ToS for programmatic use
  **not verified** here. Third-party sources suggest personal/non-commercial RSS
  consumption is tolerated while commercial reuse needs review — **verify with
  counsel** before production. Store only URL + retrieved snippet/text needed for
  provenance, not bulk re-publication.
- **NewsAPI.org.** Free *Developer* tier is **dev-environment-only** (quoted §1.3)
  and cannot be used in production "including internally" — so it is for local
  prototyping only unless a paid tier is purchased.
- **Minimize + attribute.** Store only what provenance requires
  (`Document.url`, `published_at`, the extracted snippet); record `Source` domain;
  prefer linking over wholesale text copying.

---

## 5. Findings for downstream issues

**For #11 (build `data/athlete_handles.yaml`):**

- Use the §3.3 schema; key on upstream `athletes.id`; only `name` + `news_query`
  are required. File is committed (un-ignored in `.gitignore`); keep it
  secret-free.
- Selection criteria in §3.2 — curation is a human task; the two example rows in
  §3.4 are **fictional** and must be replaced.
- Optional Wikidata `wikidata_qid` enables the residence/IFSC-id crosswalk, but
  do not depend on it (P3690 sparse, §1.2).

**For #12 (extraction):**

- News is PRIMARY document source (Google News RSS keyless; NewsAPI free tier is
  dev-only). Socials deferred (handles slot exists but unused).
- Extractor must enforce the PRD §8.5 invariant: no `TrainingSignal` /
  `BASED_IN` upgrade without an `EVIDENCED_BY` `Document`; drop low-confidence.
- `BASED_IN` precedence ladder (§2): news-imputed → Wikidata P551 → origin-proxy →
  nationality-proxy, each with the stamped `source` + `confidence` shown.
- All node ids/labels go through `src/climber_network/vocab.py`
  (`source_id`, `doc`, `run`, `sig`; `assert_label`/`assert_rel`).

**Open VERIFY items (must resolve before relying on this):**

1. Semantics of the IFSC profile `City:` field (origin vs. residence vs. base).
2. Whether a stable JSON endpoint backs IFSC profile pages (vs. HTML scrape).
3. IFSC / worldclimbing.com **Terms of Use** text (not located this pass).
4. Google News RSS Terms of Service for programmatic use.
5. Re-run Wikidata coverage across the **union** of climbing occupations
   (sport climber `Q11481802`, rock climber `Q11341457`, climber) — the
   per-occupation IFSC-id count of 0 is a filter artifact, not reality
   (Adam Ondra `Q350568` does carry `P3690 = 1364`).

---

## 6. Sources (fetched / queried this pass)

- IFSC / World Climbing athletes hub: <https://www.worldclimbing.com/resources/athletes>
  (redirected from <https://www.ifsc-climbing.org/resources/athletes>)
- IFSC athlete profile — Merritt Ernsberger: <https://www.worldclimbing.com/athlete/1908/merritt-ernsberger>
- IFSC athlete profile — Tamara Ulzhabayeva: <https://www.worldclimbing.com/athlete/3357/tamara-ulzhabayeva>
- IFSC robots.txt: <https://www.worldclimbing.com/robots.txt>
- Wikidata: Data access: <https://www.wikidata.org/wiki/Wikidata:Data_access>
- Wikidata: SPARQL query service / Copyright: <https://www.wikidata.org/wiki/Wikidata:SPARQL_query_service/Copyright>
- Wikidata property P551 (residence): <https://www.wikidata.org/wiki/Property:P551>
- Wikidata property P3690 (IFSC climber ID): <https://www.wikidata.org/wiki/Property:P3690>
- Wikidata SPARQL endpoint (live queries run this pass): <https://query.wikidata.org/sparql>
- NewsAPI.org pricing: <https://newsapi.org/pricing>
- Google News RSS parameter reference (third-party, NewsCatcher): <https://www.newscatcherapi.com/blog-posts/google-news-rss-search-parameters-the-missing-documentaiton>
- USA Climbing team rosters: <https://usaclimbing.org/team-rosters/>
