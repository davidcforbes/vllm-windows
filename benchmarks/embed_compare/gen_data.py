"""Deterministic synthetic legal-case document generator for the embedding benchmark.

Produces N long-form (>32K-token) legal documents, each with M unique, queryable
"needle" facts injected at controlled depths. Writes one .txt per doc plus a
manifest.json describing every needle and its gold query.

Deterministic: fixed seed -> identical corpus every run (reproducible benchmark).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DOCS = DATA / "docs"

N_DOCS = 8
SEED = 1337
TARGET_CHARS = 215_000  # ~48-52K tokens; verified >32768 against the voyage tokenizer

COURTS = [
    "United States District Court for the Northern District of California",
    "Superior Court of the State of New York, County of New York",
    "United States District Court for the Southern District of Texas",
    "Court of Chancery of the State of Delaware",
    "United States District Court for the District of Massachusetts",
    "Superior Court of the State of Washington, King County",
    "United States District Court for the Northern District of Illinois",
    "Commonwealth of Pennsylvania Court of Common Pleas, Philadelphia County",
]
FIRMS = ["Halvorsen", "Crandall", "Mbeki", "Okonkwo", "Vasquez", "Trent", "Lindqvist",
         "Achterberg", "Donnelly", "Yamamoto", "Castellano", "Bui", "Eckhart", "Nwosu"]
SUFFIX = ["Industries", "Holdings", "Logistics", "BioSciences", "Capital Partners",
          "Maritime", "Aerospace", "Pharmaceuticals", "Robotics", "Foods", "Energy"]
JUDGES = ["Marlena Ortiz", "Desmond Achebe", "Priya Ramanathan", "Konrad Halloway",
          "Estelle Brennan", "Tobias Wexler", "Imani Castellano", "Reuben Fairweather",
          "Soledad Marchetti", "Bartholomew Quinn", "Anneliese Vogt", "Dmitri Sokolov"]
EXHIBIT_THINGS = [
    "the unredacted maintenance log", "a forged delivery receipt",
    "the original wet-ink signature page", "an encrypted spreadsheet of side payments",
    "the recalled bill of lading", "a contemporaneous email thread",
    "the calibration certificate for the failed sensor", "a handwritten ledger of cash advances",
]
CLAUSES = [
    "The plaintiff alleges that the defendant breached the implied covenant of good faith and fair dealing.",
    "Discovery in this matter proceeded under the protective order entered by the magistrate judge.",
    "The defendant maintains that all obligations under the master services agreement were satisfied in full.",
    "Expert testimony was offered regarding the prevailing industry standard of care.",
    "The parties stipulated to the authenticity of the documents produced in the second production.",
    "Counsel for the plaintiff moved to compel responses to the third set of interrogatories.",
    "The court took judicial notice of the regulatory filings submitted to the agency.",
    "The defendant asserted the affirmative defense of waiver and estoppel.",
    "A spoliation instruction was requested based on the deletion of backup archives.",
    "The damages model relied upon a discounted cash-flow analysis prepared by the plaintiff's expert.",
    "The witness testified that the shipment was inspected upon arrival at the distribution center.",
    "The defendant contends that the limitation-of-liability clause bars the claimed consequential damages.",
    "The plaintiff seeks specific performance in addition to monetary relief.",
    "The arbitration provision was found to be unconscionable as applied to the consumer claims.",
    "The record reflects that notice was provided in accordance with the contractual cure period.",
]


def case_identity(rng: random.Random, i: int) -> dict:
    pl = f"{rng.choice(FIRMS)} {rng.choice(SUFFIX)}"
    df = f"{rng.choice(FIRMS)} {rng.choice(SUFFIX)}"
    while df == pl:
        df = f"{rng.choice(FIRMS)} {rng.choice(SUFFIX)}"
    return {
        "plaintiff": pl,
        "defendant": df,
        "court": COURTS[i % len(COURTS)],
        "case_no": f"{2020 + i}-CV-{10000 + i * 731:05d}",
        "judge": JUDGES[i % len(JUDGES)].strip().rstrip(", "),
    }


def needles_for(rng: random.Random, ident: dict, i: int) -> list[dict]:
    case = f"{ident['plaintiff']} v. {ident['defendant']}"
    amount = f"{rng.randint(1, 9)},{rng.randint(100, 999)},{rng.randint(100, 999)}"
    date = f"{rng.choice(['January','March','June','September','November'])} {rng.randint(1,28)}, 20{rng.randint(18,24):02d}"
    exh_no = f"{rng.randint(11, 99)}-{chr(rng.randint(65, 79))}"
    exh_thing = rng.choice(EXHIBIT_THINGS)
    return [
        {"type": "settlement_amount", "depth": 0.0,
         "text": f"The parties agreed to a confidential settlement in the amount of ${amount} to fully resolve all claims in {case}.",
         "query": f"What was the settlement amount in {case}?", "answer": f"${amount}"},
        {"type": "case_number", "depth": 0.25,
         "text": f"This action was docketed as Case No. {ident['case_no']} before the {ident['court']}.",
         "query": f"What is the docket/case number for {case}?", "answer": ident["case_no"]},
        {"type": "judge_name", "depth": 0.5,
         "text": f"The matter was assigned to the Honorable {ident['judge']}, who presided over every proceeding in {case}.",
         "query": f"Which judge presided over {case}?", "answer": ident["judge"]},
        {"type": "key_date", "depth": 0.75,
         "text": f"The alleged breach of contract in {case} occurred on {date}, when the defendant failed to deliver conforming goods.",
         "query": f"On what date did the breach occur in {case}?", "answer": date},
        {"type": "exhibit", "depth": 1.0,
         "text": f"Exhibit {exh_no} in {case} contained {exh_thing}, which proved decisive at trial.",
         "query": f"What did Exhibit {exh_no} contain in {case}?", "answer": exh_thing},
    ]


def build_doc(rng: random.Random, ident: dict, needles: list[dict]) -> str:
    case = f"{ident['plaintiff']} v. {ident['defendant']}"
    header = (
        f"{ident['court']}\n\n{ident['plaintiff']},\n    Plaintiff,\n\nv.\n\n"
        f"{ident['defendant']},\n    Defendant.\n\nCase No. {ident['case_no']}\n\n"
        f"MEMORANDUM OPINION AND ORDER\n\nBefore the Honorable {ident['judge']}.\n\n"
        "I. PROCEDURAL HISTORY\n\n"
    )
    # Build a large pool of distinct filler paragraphs.
    paras: list[str] = []
    n = 0
    while sum(len(p) for p in paras) < TARGET_CHARS:
        sents = [f"({n}) " + rng.choice(CLAUSES) for _ in range(rng.randint(4, 7))]
        paras.append(" ".join(sents))
        n += 1
    # Insert each needle at its depth (as its own paragraph).
    for nd in needles:
        idx = min(len(paras) - 1, max(0, round(nd["depth"] * (len(paras) - 1))))
        paras.insert(idx, "STATEMENT OF FACT. " + nd["text"])
    body = "II. STATEMENT OF FACTS\n\n" + "\n\n".join(paras)
    tail = (
        f"\n\nIII. CONCLUSION\n\nFor the foregoing reasons in {case}, the court enters "
        "judgment consistent with this opinion. IT IS SO ORDERED.\n"
    )
    return header + body + tail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-tokenizer", default="voyageai/voyage-4-nano",
                    help="HF id whose tokenizer verifies docs exceed 32768 tokens; '' to skip")
    args = ap.parse_args()

    DOCS.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    manifest = []
    for i in range(N_DOCS):
        ident = case_identity(rng, i)
        needles = needles_for(rng, ident, i)
        text = build_doc(rng, ident, needles)
        doc_id = f"doc_{i:02d}"
        (DOCS / f"{doc_id}.txt").write_text(text, encoding="utf-8")
        manifest.append({
            "doc_id": doc_id, "file": f"docs/{doc_id}.txt", **ident,
            "needles": [{"needle_id": f"{doc_id}_n{j}", **nd} for j, nd in enumerate(needles)],
        })
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    n_needles = sum(len(d["needles"]) for d in manifest)
    print(f"wrote {N_DOCS} docs, {n_needles} needles/queries -> {DATA}")

    if args.verify_tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.verify_tokenizer, trust_remote_code=True)
        ok = True
        for d in manifest:
            ids = tok(Path(DOCS / f"{d['doc_id']}.txt").read_text(encoding="utf-8"),
                      add_special_tokens=False)["input_ids"]
            flag = "OK" if len(ids) > 32768 else "TOO SHORT"
            ok = ok and len(ids) > 32768
            print(f"  {d['doc_id']}: {len(ids):>6} tokens [{flag}]")
        if not ok:
            raise SystemExit("ERROR: some docs are <= 32768 tokens; raise TARGET_CHARS")
        print("all docs exceed 32768 tokens (voyage tokenizer)")


if __name__ == "__main__":
    main()
