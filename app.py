"""ClauseGuard Streamlit interface.

Run:  streamlit run app.py

The UI only *displays* what the engine in src/ returns. Every example, warning
and highlight comes from src.analyzer.analyze(); nothing is hardcoded.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from scripts.evaluate import prf
from src import ENGINE_VERSION, AnalysisResult, analyze
from src.export import passage_to_markdown, result_to_markdown
from src.highlighting import highlight, missing_in_rewrite
from src.passage import PassageResult, analyze_passage
from src.rules import SEVERITY_POLICY
from src.schemas import CATEGORIES, Evidence, Finding, Status

ROOT = Path(__file__).resolve().parent
EXAMPLES = json.loads((ROOT / "data" / "examples.json").read_text(encoding="utf-8"))
EXAMPLE_BY_ID = {ex["id"]: ex for ex in EXAMPLES}
PASSAGE_EXAMPLE = json.loads((ROOT / "data" / "passage_example.json").read_text(encoding="utf-8"))
RESULTS_PATH = ROOT / "reports" / "evaluation_results.json"

CATEGORY_INFO = {
    "negation": ("Negation", "Polarity words that deny or reverse a clause.", "not · no · never · without · cannot · ineligible"),
    "modality": ("Modality", "Whether an action is required, recommended, permitted or prohibited.", "must · shall · should · may · can · required to"),
    "condition": ("Conditions", "Conditions and exceptions that limit when a rule applies.", "if · only if · unless · except · provided that · subject to"),
    "quantifier": ("Quantifiers", "Scope words, numeric bounds and deadlines.", "all · only · some · at least · no more than · before · by"),
    "numeric": ("Values", "Numbers, percentages, amounts, dates, weekdays and times.", "75% · Rs. 500 · 15 November · Friday · 5 pm"),
}

st.set_page_config(page_title="ClauseGuard", page_icon="§", layout="wide")


# --------------------------------------------------------------------------- styles

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
  --bg: oklch(1 0 0);
  --surface: oklch(0.982 0.003 355);
  --surface-2: oklch(0.962 0.005 355);
  --line: oklch(0.915 0.006 355);
  --line-strong: oklch(0.83 0.008 355);
  --ink: oklch(0.22 0.012 355);
  --ink-2: oklch(0.38 0.012 355);
  --ink-3: oklch(0.49 0.010 355);

  --high: oklch(0.54 0.205 355);
  --high-bg: oklch(0.962 0.028 355);
  --medium: oklch(0.52 0.115 62);
  --medium-bg: oklch(0.965 0.045 82);
  --review: oklch(0.49 0.085 250);
  --review-bg: oklch(0.963 0.018 250);

  --c-negation: oklch(0.53 0.17 300);
  --c-modality: oklch(0.53 0.10 195);
  --c-condition: oklch(0.50 0.14 258);
  --c-quantifier: oklch(0.53 0.12 135);
  --c-numeric: oklch(0.60 0.14 52);

  --radius: 10px;
  --sans: 'IBM Plex Sans', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif;
  --mono: 'IBM Plex Mono', ui-monospace, 'SF Mono', Menlo, monospace;
  --ease: cubic-bezier(0.22, 1, 0.36, 1);
}

html, body, [data-testid="stAppViewContainer"], .stApp { background: var(--bg); }
html, body, .stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea,
.stApp button, .stApp [data-baseweb] { font-family: var(--sans); }
.stApp { color: var(--ink); }

#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; height: 0; }
.block-container { max-width: 1180px; padding: 2.25rem 1.5rem 4rem; }
@media (max-width: 640px) { .block-container { padding: 1.25rem 1rem 3rem; } }

/* ---- masthead */
.cg-mast { display: flex; align-items: flex-end; justify-content: space-between; gap: 1.5rem;
  flex-wrap: wrap; padding-bottom: 1.25rem; border-bottom: 1px solid var(--line); margin-bottom: 0.25rem; }
.cg-brand { display: flex; align-items: center; gap: 0.75rem; }
.cg-glyph { width: 38px; height: 38px; border-radius: 9px; background: var(--ink); color: var(--bg);
  display: grid; place-items: center; font: 600 1.25rem/1 var(--sans); position: relative; }
.cg-glyph::after { content: ""; position: absolute; right: -3px; bottom: -3px; width: 11px; height: 11px;
  border-radius: 50%; background: var(--high); border: 2px solid var(--bg); }
.cg-title { font: 600 1.6rem/1.1 var(--sans); letter-spacing: -0.02em; margin: 0; color: var(--ink); }
.cg-sub { margin: 0.2rem 0 0; color: var(--ink-2); font-size: 1rem; }
.cg-meta-tags { display: flex; gap: 0.5rem; flex-wrap: wrap; }
.cg-tag { font: 500 0.75rem/1 var(--mono); color: var(--ink-2); border: 1px solid var(--line);
  border-radius: 999px; padding: 0.4rem 0.65rem; background: var(--surface); }
.cg-caveat { color: var(--ink-2); font-size: 0.9rem; margin: 0.9rem 0 0.25rem; max-width: 80ch; }
.cg-caveat strong { color: var(--ink); font-weight: 600; }

/* ---- tabs */
.stTabs [data-baseweb="tab-list"] { gap: 1.75rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { padding: 0.9rem 0 0.7rem; background: transparent; }
.stTabs [data-baseweb="tab"] p { font-size: 0.95rem; font-weight: 500; color: var(--ink-3); }
.stTabs [aria-selected="true"] p { color: var(--ink); }
.stTabs [data-baseweb="tab-highlight"] { background: var(--ink); height: 2px; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* ---- inputs */
.cg-field-label { display: flex; align-items: baseline; justify-content: space-between; gap: 1rem;
  margin: 0.5rem 0 0.4rem; }
.cg-field-label b { font-weight: 600; font-size: 0.95rem; color: var(--ink); }
.cg-field-label span { font-size: 0.82rem; color: var(--ink-3); }
.stTextArea textarea { font: 400 1.02rem/1.6 var(--sans) !important; color: var(--ink) !important;
  background: var(--bg) !important; padding: 0.85rem 1rem !important; }
.stTextArea [data-baseweb="textarea"] { border: 1px solid var(--line-strong) !important; border-radius: var(--radius) !important;
  background: var(--bg) !important; transition: border-color 160ms var(--ease), box-shadow 160ms var(--ease); }
.stTextArea [data-baseweb="textarea"]:focus-within { border-color: var(--ink) !important;
  box-shadow: 0 0 0 3px oklch(0.22 0.012 355 / 0.10) !important; }
.stTextArea textarea::placeholder { color: var(--ink-3) !important; opacity: 1; }
.stSelectbox [data-baseweb="select"] > div { border-radius: var(--radius); border-color: var(--line-strong);
  background: var(--bg); min-height: 42px; }
.stSelectbox label p, .stRadio label p { font-size: 0.85rem; color: var(--ink-2); font-weight: 500; }

/* ---- buttons */
.stButton button, .stDownloadButton button { border-radius: 8px; font-weight: 500; min-height: 42px;
  padding: 0 1.15rem; transition: background 160ms var(--ease), border-color 160ms var(--ease), transform 120ms var(--ease); }
.stButton button[kind="primary"] { background: var(--ink); border: 1px solid var(--ink); color: var(--bg); }
.stButton button[kind="primary"]:hover { background: oklch(0.32 0.014 355); border-color: oklch(0.32 0.014 355); }
.stButton button[kind="secondary"], .stDownloadButton button { background: var(--bg); color: var(--ink);
  border: 1px solid var(--line-strong); }
.stButton button[kind="secondary"]:hover, .stDownloadButton button:hover { border-color: var(--ink); color: var(--ink); background: var(--surface); }
.stButton button:active, .stDownloadButton button:active { transform: translateY(1px); }
.stButton button:focus-visible, .stDownloadButton button:focus-visible { outline: 2px solid var(--ink); outline-offset: 2px; }
.stButton button p, .stDownloadButton button p { font-weight: 500; }

/* ---- status */
.cg-status { display: flex; gap: 0.9rem; align-items: flex-start; padding: 1rem 1.15rem; border-radius: var(--radius);
  border: 1px solid var(--line); background: var(--surface); margin-top: 0.4rem; }
.cg-status .dot { width: 10px; height: 10px; border-radius: 50%; margin-top: 0.42rem; flex: none; background: var(--ink-3); }
.cg-status strong { display: block; font-weight: 600; font-size: 1.05rem; color: var(--ink); }
.cg-status span { color: var(--ink-2); font-size: 0.9rem; }
.cg-status.sev-high { background: var(--high-bg); border-color: oklch(0.86 0.06 355); }
.cg-status.sev-high .dot { background: var(--high); box-shadow: 0 0 0 4px oklch(0.54 0.205 355 / 0.15); }
.cg-status.sev-medium { background: var(--medium-bg); border-color: oklch(0.87 0.07 80); }
.cg-status.sev-medium .dot { background: var(--medium); }
.cg-status.sev-review { background: var(--review-bg); border-color: oklch(0.87 0.03 250); }
.cg-status.sev-review .dot { background: var(--review); }
.cg-counts { font: 500 0.8rem/1 var(--mono); color: var(--ink-2); margin-top: 0.55rem; display: flex; gap: 0.9rem; flex-wrap: wrap; }
.cg-counts i { font-style: normal; color: var(--ink); }

/* ---- side by side */
.cg-section-h { display: flex; align-items: baseline; justify-content: space-between; gap: 1rem; flex-wrap: wrap;
  margin: 2rem 0 0.75rem; }
.cg-section-h h3 { font: 600 1.1rem/1.3 var(--sans); margin: 0; color: var(--ink); letter-spacing: -0.005em; padding: 0; }
.cg-section-h p { margin: 0; font-size: 0.85rem; color: var(--ink-3); }
.cg-compare { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); border: 1px solid var(--line);
  border-radius: var(--radius); overflow: hidden; }
.cg-pane { padding: 1.15rem 1.3rem 1.3rem; background: var(--bg); }
.cg-pane + .cg-pane { border-left: 1px solid var(--line); background: var(--surface); }
@media (max-width: 700px) { .cg-pane + .cg-pane { border-left: 0; border-top: 1px solid var(--line); } }
.cg-pane-label { font: 500 0.78rem/1 var(--mono); color: var(--ink-3); margin-bottom: 0.7rem; }
.cg-sentence { font-size: 1.12rem; line-height: 1.85; color: var(--ink); margin: 0; text-wrap: pretty; }
.cg-absent { margin-top: 0.9rem; padding-top: 0.8rem; border-top: 1px dashed var(--line-strong);
  display: flex; flex-wrap: wrap; gap: 0.4rem; align-items: center; font-size: 0.82rem; color: var(--ink-2); }
.cg-chip { font: 500 0.8rem/1 var(--mono); padding: 0.32rem 0.5rem; border-radius: 6px; text-decoration: line-through;
  text-decoration-thickness: 1px; }

/* markers */
mark.cg-mark { background: transparent; color: inherit; padding: 0.05em 0.12em; border-radius: 3px;
  border-bottom: 2px solid var(--cat, var(--line-strong)); }
mark.cg-negation { --cat: var(--c-negation); background: oklch(0.96 0.025 300); }
mark.cg-modality { --cat: var(--c-modality); background: oklch(0.965 0.02 195); }
mark.cg-condition { --cat: var(--c-condition); background: oklch(0.96 0.022 258); }
mark.cg-quantifier { --cat: var(--c-quantifier); background: oklch(0.965 0.025 135); }
mark.cg-numeric { --cat: var(--c-numeric); background: oklch(0.965 0.03 60); }
mark.cg-context { background: transparent; border-bottom-style: dashed; }
mark.cg-flag-high { box-shadow: inset 0 0 0 1.5px var(--high); background: var(--high-bg); border-bottom-color: var(--high); font-weight: 500; }
mark.cg-flag-medium { box-shadow: inset 0 0 0 1.5px var(--medium); background: var(--medium-bg); border-bottom-color: var(--medium); font-weight: 500; }
mark.cg-flag-review { box-shadow: inset 0 0 0 1.5px var(--review); background: var(--review-bg); border-bottom-color: var(--review); }

.cg-legend { display: flex; flex-wrap: wrap; gap: 0.5rem 1.1rem; margin-top: 0.75rem; font-size: 0.8rem; color: var(--ink-2); }
.cg-legend span { display: inline-flex; align-items: center; gap: 0.4rem; }
.cg-sw { width: 14px; height: 10px; border-radius: 2px; border-bottom: 2px solid var(--cat); display: inline-block; }
.cg-sw.flag { border: 1.5px solid var(--high); background: var(--high-bg); border-radius: 3px; }

/* ---- findings */
.cg-findings { border: 1px solid var(--line); border-radius: var(--radius); overflow: hidden; }
.cg-finding { padding: 1.1rem 1.3rem 1.2rem; background: var(--bg); animation: cg-in 220ms var(--ease) both; }
.cg-finding + .cg-finding { border-top: 1px solid var(--line); }
.cg-finding header { display: flex; align-items: center; gap: 0.65rem; flex-wrap: wrap; margin-bottom: 0.55rem; }
.cg-pill { font: 600 0.72rem/1 var(--sans); padding: 0.32rem 0.55rem; border-radius: 6px; letter-spacing: 0.01em; }
.cg-pill.sev-high, .cg-chip.sev-high { color: var(--high); background: var(--high-bg); box-shadow: inset 0 0 0 1px oklch(0.85 0.07 355); }
.cg-pill.sev-medium, .cg-chip.sev-medium { color: var(--medium); background: var(--medium-bg); box-shadow: inset 0 0 0 1px oklch(0.86 0.07 80); }
.cg-pill.sev-review, .cg-chip.sev-review { color: var(--review); background: var(--review-bg); box-shadow: inset 0 0 0 1px oklch(0.87 0.03 250); }
.cg-cat { display: inline-flex; align-items: center; gap: 0.4rem; font-size: 0.85rem; font-weight: 500; color: var(--ink-2); }
.cg-cat i { width: 8px; height: 8px; border-radius: 50%; background: var(--cat, var(--ink-3)); }
.cg-rule { font: 400 0.76rem/1 var(--mono); color: var(--ink-3); margin-left: auto; }
.cg-expl { font-size: 1rem; line-height: 1.55; color: var(--ink); margin: 0 0 0.85rem; max-width: 85ch; }
.cg-ev { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 0.6rem; margin-bottom: 0.85rem; }
.cg-ev > div { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 0.6rem 0.8rem; }
.cg-ev small { display: block; font: 500 0.72rem/1 var(--mono); color: var(--ink-3); margin-bottom: 0.4rem; }
.cg-ev p { margin: 0; font-size: 0.93rem; line-height: 1.5; color: var(--ink-2); }
.cg-ev p b { color: var(--ink); font-weight: 600; background: oklch(0.93 0.01 355); padding: 0 0.15em; border-radius: 3px; }
.cg-ev p.none { color: var(--ink-3); font-style: italic; }
.cg-why { display: grid; grid-template-columns: max-content 1fr; gap: 0.35rem 1rem; margin: 0; font-size: 0.88rem; }
.cg-why dt { color: var(--ink-3); font-weight: 500; }
.cg-why dd { margin: 0; color: var(--ink-2); max-width: 80ch; }
@media (max-width: 560px) { .cg-why { grid-template-columns: 1fr; } .cg-why dd { margin-bottom: 0.4rem; } .cg-rule { margin-left: 0; } }

.cg-none { padding: 1.4rem 1.3rem; border: 1px dashed var(--line-strong); border-radius: var(--radius); color: var(--ink-2); font-size: 0.95rem; }
.cg-notes { margin: 1rem 0 0; padding: 0; list-style: none; font-size: 0.85rem; color: var(--ink-2); }
.cg-notes li { padding: 0.2rem 0; }
.cg-stale { font-size: 0.85rem; color: var(--medium); margin-top: 0.6rem; }

/* ---- empty state / reference lists */
.cg-ref { border: 1px solid var(--line); border-radius: var(--radius); overflow: hidden; margin-top: 1.25rem; }
.cg-ref-row { display: grid; grid-template-columns: 11rem 1fr minmax(14rem, 1.1fr); gap: 1rem; padding: 0.85rem 1.2rem; align-items: baseline; }
.cg-ref-row + .cg-ref-row { border-top: 1px solid var(--line); }
.cg-ref-row b { font-weight: 600; display: inline-flex; gap: 0.5rem; align-items: center; }
.cg-ref-row b i { width: 8px; height: 8px; border-radius: 50%; background: var(--cat); }
.cg-ref-row span { color: var(--ink-2); font-size: 0.92rem; }
.cg-ref-row code { font: 400 0.8rem/1.5 var(--mono); color: var(--ink-2); background: none; padding: 0; }
@media (max-width: 760px) { .cg-ref-row { grid-template-columns: 1fr; gap: 0.25rem; } }
.cg-intro code, .cg-prov code { font: 400 0.85em var(--mono); background: var(--surface-2); color: var(--ink);
  padding: 0.08rem 0.3rem; border-radius: 4px; }
[data-testid="stHeaderActionElements"] { display: none; }
.cg-intro { font-size: 1rem; color: var(--ink-2); max-width: 70ch; margin: 1.5rem 0 0; line-height: 1.6; }

/* ---- benchmark */
.cg-prov { font: 400 0.8rem/1.6 var(--mono); color: var(--ink-3); margin: 0.25rem 0 0; }
.cg-table { width: 100%; border-collapse: collapse; font-size: 0.92rem; margin: 0; border: 0; }
.cg-table th, .cg-table td { border: 0 !important; border-bottom: 1px solid var(--line) !important; }
.cg-table tr:last-child td { border-bottom: 0 !important; }
.cg-table th { text-align: left; font-weight: 500; color: var(--ink-3); font-size: 0.8rem; padding: 0.65rem 0.9rem;
  border-bottom: 1px solid var(--line); background: var(--surface); }
.cg-table td { padding: 0.6rem 0.9rem; border-bottom: 1px solid var(--line); color: var(--ink); }
.cg-table tr:last-child td { border-bottom: 0; }
.cg-table td.num, .cg-table th.num { text-align: right; font-family: var(--mono); font-variant-numeric: tabular-nums; white-space: nowrap; }
.cg-table td.muted { color: var(--ink-3); }
.cg-table tr.total td { font-weight: 600; background: var(--surface); }
.cg-tablewrap { border: 1px solid var(--line); border-radius: var(--radius); overflow-x: auto; }
.cg-bar { display: inline-block; height: 6px; border-radius: 3px; background: var(--ink); vertical-align: middle; margin-right: 0.5rem; }
.cg-bar.warn { background: var(--medium); }
.cg-bar-track { display: inline-block; width: 56px; height: 6px; border-radius: 3px; background: var(--surface-2); vertical-align: middle; margin-right: 0.6rem; position: relative; }
.cg-bar-track .cg-bar { position: absolute; left: 0; top: 0; margin: 0; }

.cg-cm { display: grid; grid-template-columns: auto 1fr 1fr; border: 1px solid var(--line); border-radius: var(--radius); overflow: hidden; font-size: 0.85rem; }
.cg-cm > div { padding: 0.75rem 0.9rem; border-bottom: 1px solid var(--line); border-right: 1px solid var(--line); }
.cg-cm > div:nth-child(3n) { border-right: 0; }
.cg-cm > div:nth-last-child(-n+3) { border-bottom: 0; }
.cg-cm .h { background: var(--surface); color: var(--ink-3); font-weight: 500; }
.cg-cm .v { font: 500 1.3rem/1.2 var(--mono); color: var(--ink); }
.cg-cm .v small { display: block; font: 400 0.72rem/1.3 var(--sans); color: var(--ink-3); }
.cg-cm .v.good { background: oklch(0.975 0.015 160); }
.cg-cm .v.bad { background: var(--high-bg); }

.cg-callout { border: 1px solid var(--line); background: var(--surface); border-radius: var(--radius); padding: 0.9rem 1.1rem;
  font-size: 0.9rem; color: var(--ink-2); line-height: 1.6; margin-top: 1rem; }
.cg-callout b { color: var(--ink); font-weight: 600; }
.cg-err { font-size: 0.88rem; }
.cg-err td:first-child { font-family: var(--mono); font-size: 0.78rem; color: var(--ink-3); white-space: nowrap; }
.cg-kind { font: 600 0.7rem/1 var(--sans); padding: 0.25rem 0.4rem; border-radius: 4px; }
.cg-kind.fp { color: var(--medium); background: var(--medium-bg); }
.cg-kind.fn { color: var(--high); background: var(--high-bg); }

/* ---- method */
.cg-steps { list-style: none; counter-reset: s; margin: 0; padding: 0; border: 1px solid var(--line); border-radius: var(--radius); overflow: hidden; }
.cg-steps li { counter-increment: s; display: grid; grid-template-columns: 2rem 1fr; gap: 0.75rem; padding: 0.85rem 1.2rem; }
.cg-steps li + li { border-top: 1px solid var(--line); }
.cg-steps li::before { content: counter(s); font: 500 0.85rem/1.6 var(--mono); color: var(--ink-3); }
.cg-steps b { font-weight: 600; } .cg-steps span { color: var(--ink-2); font-size: 0.92rem; }
.cg-steps code, .cg-prose code { font: 400 0.82rem var(--mono); background: var(--surface-2); padding: 0.1rem 0.3rem; border-radius: 4px; color: var(--ink); }
.cg-prose { max-width: 75ch; color: var(--ink-2); line-height: 1.65; font-size: 0.95rem; }
.cg-prose li { margin: 0.3rem 0; }

/* ---- example chips (st.pills) */
[data-testid="stPills"] { margin-bottom: 0.4rem; }
[data-testid="stPills"] button { border-radius: 999px; border: 1px solid var(--line-strong); background: var(--bg);
  color: var(--ink-2); font-size: 0.84rem; min-height: 32px; padding: 0 0.8rem;
  transition: border-color 160ms var(--ease), background 160ms var(--ease), color 160ms var(--ease); }
[data-testid="stPills"] button:hover { border-color: var(--ink); color: var(--ink); background: var(--surface); }
[data-testid="stPills"] button[kind="pillsActive"], [data-testid="stPills"] button[data-testid="stBaseButton-pillsActive"] {
  background: var(--ink); border-color: var(--ink); color: var(--bg); }
[data-testid="stPills"] button p { font-size: 0.84rem; }

/* ---- segmented control */
[data-testid="stButtonGroup"] button { min-height: 34px; font-size: 0.85rem; }

/* ---- expanders */
[data-testid="stExpander"] details { border: 0; border-top: 1px solid var(--line); border-radius: 0; background: var(--bg); }
[data-testid="stExpander"] summary { padding: 0.85rem 0.25rem; }
[data-testid="stExpander"] summary:hover { background: transparent; }
[data-testid="stExpander"] summary:hover p { color: var(--ink-2); }
[data-testid="stExpanderDetails"] { padding: 0.1rem 0 1.4rem; }
[data-testid="stExpander"] summary p { font-size: 0.93rem; color: var(--ink); }
[data-testid="stExpander"] summary strong { font-weight: 600; }

/* ---- uploader + dataframe */
[data-testid="stFileUploaderDropzone"] { border: 1px dashed var(--line-strong); border-radius: var(--radius); background: var(--surface); }
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: var(--radius); overflow: hidden; }

/* ---- passage alignment map */
.cg-map { border: 1px solid var(--line); border-radius: var(--radius); overflow: hidden; }
.cg-map-head, .cg-map-row { display: grid; grid-template-columns: 2.2rem 8.5rem 1fr 1.5rem 1fr; gap: 0.9rem; padding: 0.75rem 1.1rem; }
.cg-map-head { background: var(--surface); font-size: 0.78rem; color: var(--ink-3); font-weight: 500; border-bottom: 1px solid var(--line); }
.cg-map-row + .cg-map-row { border-top: 1px solid var(--line); }
.cg-map-n { font: 500 0.85rem/1.6 var(--mono); color: var(--ink-3); }
.cg-map-kind { display: flex; flex-direction: column; align-items: flex-start; gap: 0.35rem; }
.cg-map-o p, .cg-map-r p { margin: 0 0 0.25rem; font-size: 0.9rem; line-height: 1.5; color: var(--ink); }
.cg-map-o p.none, .cg-map-r p.none { color: var(--ink-3); }
.cg-map-arrow { color: var(--ink-3); padding-top: 0.1rem; }
.cg-kindtag { font: 500 0.72rem/1 var(--mono); padding: 0.28rem 0.45rem; border-radius: 5px; background: var(--surface-2); color: var(--ink-2); }
.cg-kindtag.k-dropped, .cg-kindtag.k-added { background: var(--medium-bg); color: var(--medium); }
.cg-pill.sev-none { color: var(--ink-3); background: var(--surface); box-shadow: inset 0 0 0 1px var(--line); }
@media (max-width: 760px) {
  .cg-map-head { display: none; }
  .cg-map-row { grid-template-columns: 2rem 1fr; }
  .cg-map-kind { flex-direction: row; }
  .cg-map-o, .cg-map-r, .cg-map-arrow { grid-column: 2; }
  .cg-map-arrow { padding: 0; transform: rotate(90deg); width: fit-content; }
}

/* ---- footer */
.cg-foot { margin-top: 3.5rem; padding-top: 1.1rem; border-top: 1px solid var(--line); display: flex; flex-wrap: wrap;
  gap: 0.5rem 1.5rem; justify-content: space-between; font-size: 0.8rem; color: var(--ink-3); }
.cg-foot code { font: 400 0.78rem var(--mono); background: none; color: var(--ink-2); padding: 0; }

@keyframes cg-in { from { opacity: 0.001; transform: translateY(4px); } to { opacity: 1; transform: none; } }
@media (prefers-reduced-motion: reduce) { .cg-finding { animation: none; } * { transition: none !important; } }
</style>
"""


def put(markup: str) -> None:
    """Render an HTML fragment. Lines are joined so Markdown never sees indentation."""
    st.markdown(" ".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


e = html.escape


# --------------------------------------------------------------------------- pieces

def masthead() -> None:
    put(f"""
    <div class="cg-mast">
      <div class="cg-brand">
        <div class="cg-glyph" aria-hidden="true">§</div>
        <div>
          <h1 class="cg-title">ClauseGuard</h1>
          <p class="cg-sub">Audit critical meaning changes in rewritten instructions.</p>
        </div>
      </div>
      <div class="cg-meta-tags">
        <span class="cg-tag">rule-based · no API</span>
        <span class="cg-tag">engine {e(ENGINE_VERSION)}</span>
      </div>
    </div>
    <p class="cg-caveat"><strong>Warnings are review recommendations, not proof.</strong>
    ClauseGuard compares lexical markers and values between two sentences. It can miss changes
    it has no rule for, and it can flag faithful paraphrases.</p>
    """)


def section(title: str, hint: str = "") -> None:
    put(f'<div class="cg-section-h"><h3>{e(title)}</h3><p>{e(hint)}</p></div>')


def status_block(status: str, counts: dict[str, int]) -> str:
    top = next((s for s in ("HIGH", "MEDIUM", "REVIEW") if counts[s]), None)
    detail = {
        "HIGH": "At least one warning concerns negation, a condition, a reversal or a changed value.",
        "MEDIUM": "Warnings concern obligation strength, scope, bounds or a dropped sentence.",
        "REVIEW": "Only low-confidence observations; these are often harmless paraphrases.",
        None: "None of the configured checks fired. This does not prove the meaning is preserved.",
    }[top]
    cls = f"sev-{top.lower()}" if top else ""
    return f"""
    <div class="cg-status {cls}" role="status">
      <div class="dot"></div>
      <div><strong>{e(status)}</strong><span>{e(detail)}</span>
        <div class="cg-counts"><span><i>{counts['HIGH']}</i> high</span><span><i>{counts['MEDIUM']}</i> medium</span>
        <span><i>{counts['REVIEW']}</i> review</span></div>
      </div>
    </div>"""


def legend() -> str:
    items = "".join(
        f'<span><i class="cg-sw" style="--cat: var(--c-{c})"></i>{CATEGORY_INFO[c][0]}</span>' for c in CATEGORIES
    )
    return (f'<div class="cg-legend">{items}'
            f'<span><i class="cg-sw flag"></i>Outlined: cited as evidence by a warning</span>'
            f'<span>Hover a mark to see its normalised label</span></div>')


def comparison(result: AnalysisResult) -> str:
    missing = missing_in_rewrite(result)
    chips = "".join(
        f'<span class="cg-chip sev-{f.severity.value.lower()}" title="{e(f.explanation)}">{e(f.original_evidence.text)}</span>'
        for f in missing
    )
    absent = f'<div class="cg-absent">Not found in rewrite {chips}</div>' if missing else ""
    return f"""
    <div class="cg-compare">
      <div class="cg-pane"><div class="cg-pane-label">original</div>
        <p class="cg-sentence">{highlight(result, "original")}</p></div>
      <div class="cg-pane"><div class="cg-pane-label">rewritten</div>
        <p class="cg-sentence">{highlight(result, "rewritten")}</p>{absent}</div>
    </div>{legend()}"""


def _evidence_html(ev: Evidence | None) -> str:
    if ev is None:
        return '<p class="none">No counterpart found</p>'
    ctx = ev.context or ev.text
    idx = ctx.find(ev.text)
    if idx < 0:
        idx = ctx.lower().find(ev.text.lower())
    if idx >= 0:
        body = e(ctx[:idx]) + f"<b>{e(ctx[idx:idx + len(ev.text)])}</b>" + e(ctx[idx + len(ev.text):])
    else:
        body = f"<b>{e(ev.text)}</b> · {e(ctx)}"
    return f"<p>“{body}”</p>"


def finding_html(f: Finding, i: int) -> str:
    sev = f.severity.value.lower()
    cat_name = CATEGORY_INFO.get(f.category, (f.category.title(),))[0]
    cat_var = f"var(--c-{f.category})" if f.category in CATEGORIES else "var(--ink-3)"
    return f"""
    <article class="cg-finding" style="animation-delay:{min(i, 8) * 35}ms">
      <header><span class="cg-pill sev-{sev}" title="{e(SEVERITY_POLICY[f.severity.value])}">{f.severity.value.title()}</span>
        <span class="cg-cat" style="--cat:{cat_var}"><i></i>{e(cat_name)}</span>
        <code class="cg-rule">{e(f.rule_id)}</code></header>
      <p class="cg-expl">{e(f.explanation)}</p>
      <div class="cg-ev"><div><small>original evidence</small>{_evidence_html(f.original_evidence)}</div>
        <div><small>rewritten evidence</small>{_evidence_html(f.rewritten_evidence)}</div></div>
      <dl class="cg-why"><dt>Why flagged</dt><dd>{e(f.reason)}</dd>
        <dt>Review</dt><dd>{e(f.suggested_action)}</dd></dl>
    </article>"""


def findings_html(findings: list[Finding]) -> str:
    return '<div class="cg-findings">' + "".join(finding_html(f, i) for i, f in enumerate(findings)) + "</div>"


def markers_frame(result: AnalysisResult) -> pd.DataFrame:
    rows = []
    for side, a in (("original", result.original_analysis), ("rewritten", result.rewritten_analysis)):
        if a is None:
            continue
        for m in a.markers:
            rows.append({"sentence": side, "category": m.category, "label": m.label, "text": m.text,
                         "detail": m.focus if m.category == "negation" else (m.detail or "")})
        for n in a.numbers:
            rows.append({"sentence": side, "category": "numeric", "label": n.kind, "text": n.text,
                         "detail": n.normalized})
    return pd.DataFrame(rows)


def empty_state() -> None:
    put('<p class="cg-intro">Paste an instruction and its rewrite, or load one of the examples above. '
        'ClauseGuard extracts the markers below from both sentences and reports every one that was '
        'removed, added or changed, with the evidence it relied on.</p>')
    rows = "".join(
        f'<div class="cg-ref-row" style="--cat: var(--c-{c})"><b><i></i>{name}</b><span>{desc}</span>'
        f'<code>{e(ex)}</code></div>'
        for c, (name, desc, ex) in CATEGORY_INFO.items()
    )
    put(f'<div class="cg-ref">{rows}</div>')


# --------------------------------------------------------------------------- compare tab

def _example_label(ex_id: str) -> str:
    ex = EXAMPLE_BY_ID[ex_id]
    return f"{ex['id']} · {ex['title']}"


def _run_compare() -> None:
    o, r = st.session_state["original"], st.session_state["rewritten"]
    st.session_state["result"] = analyze(o, r)
    st.session_state["analysed_pair"] = (o, r)


def _load_example() -> None:
    ex = EXAMPLE_BY_ID.get(st.session_state.get("example"))
    if ex is None:  # the chip was toggled off
        return
    st.session_state["original"], st.session_state["rewritten"] = ex["original"], ex["rewritten"]
    _run_compare()


def _swap() -> None:
    st.session_state["original"], st.session_state["rewritten"] = (
        st.session_state["rewritten"], st.session_state["original"])
    st.session_state["example"] = None
    if st.session_state.get("result") is not None:
        _run_compare()


def _clear() -> None:
    st.session_state["original"] = st.session_state["rewritten"] = ""
    st.session_state["example"] = None
    st.session_state["result"] = None


def render_result(result: AnalysisResult, *, show_status: bool = True) -> None:
    """Status, side-by-side highlighting, warnings and extracted markers for one pair."""
    if show_status:
        put(status_block(result.status.value, result.severity_counts()))
    if result.original_analysis is not None:
        section("Side by side", "Markers detected in each sentence")
        put(comparison(result))
    n = len(result.findings)
    section("Warnings", f"{n} finding{'s' if n != 1 else ''}, most severe first")
    if result.findings:
        put(findings_html(result.findings))
    else:
        put('<div class="cg-none">No configured check fired for this pair. The sentences may still differ in ways '
            'ClauseGuard does not check — see “Method & limits”.</div>')


def compare_tab() -> None:
    st.session_state.setdefault("original", "")
    st.session_state.setdefault("rewritten", "")

    put('<div class="cg-field-label"><b>Examples</b><span>run through the live engine when selected</span></div>')
    st.pills("Examples", list(EXAMPLE_BY_ID), format_func=_example_label, key="example",
             on_change=_load_example, label_visibility="collapsed")

    left, right = st.columns(2, gap="medium")
    with left:
        put('<div class="cg-field-label"><b>Original</b><span>source of truth</span></div>')
        st.text_area("Original sentence", key="original", height=120, label_visibility="collapsed",
                     placeholder="e.g. Students below 75% attendance cannot appear for the examination unless the HOD approves.")
    with right:
        put('<div class="cg-field-label"><b>Rewritten</b><span>summary or paraphrase to audit</span></div>')
        st.text_area("Rewritten sentence", key="rewritten", height=120, label_visibility="collapsed",
                     placeholder="e.g. Students below 75% attendance cannot appear for the examination.")

    result: AnalysisResult | None = st.session_state.get("result")
    run, swap, clear, _, dl_json, dl_md = st.columns([1.5, 0.75, 0.75, 1.2, 1.1, 1.25])
    with run:
        st.button("Compare sentences", type="primary", width="stretch", on_click=_run_compare)
    with swap:
        st.button("Swap", width="stretch", on_click=_swap, help="Exchange the original and the rewrite")
    with clear:
        st.button("Clear", width="stretch", on_click=_clear)
    with dl_json:
        st.download_button("JSON", data=result.to_json() if result else "{}", icon=":material/download:",
                           file_name="clauseguard-analysis.json", mime="application/json",
                           disabled=result is None, width="stretch")
    with dl_md:
        st.download_button("Report", data=result_to_markdown(result) if result else "", icon=":material/description:",
                           file_name="clauseguard-report.md", mime="text/markdown",
                           disabled=result is None, width="stretch", help="Markdown report with all evidence")

    if result is None:
        empty_state()
        return
    if st.session_state.get("analysed_pair") != (st.session_state["original"], st.session_state["rewritten"]):
        put('<p class="cg-stale">The inputs changed since the last comparison. Select “Compare sentences” to refresh.</p>')

    render_result(result)
    if result.notes:
        put('<ul class="cg-notes">' + "".join(f"<li>{e(n)}</li>" for n in result.notes) + "</ul>")
    if result.original_analysis is not None:
        with st.expander("Extracted markers and values"):
            frame = markers_frame(result)
            if frame.empty:
                st.caption("Nothing was extracted from either sentence.")
            else:
                st.dataframe(frame, hide_index=True, width="stretch")


# --------------------------------------------------------------------------- passage tab

KIND_LABEL = {"aligned": "Aligned", "merged": "Merged", "split": "Split", "regrouped": "Regrouped",
              "dropped": "Dropped", "added": "Added"}


def _load_passage_sample() -> None:
    st.session_state["p_original"] = PASSAGE_EXAMPLE["original"]
    st.session_state["p_rewritten"] = PASSAGE_EXAMPLE["rewritten"]
    st.session_state["p_result"] = analyze_passage(PASSAGE_EXAMPLE["original"], PASSAGE_EXAMPLE["rewritten"])


def _run_passage() -> None:
    st.session_state["p_result"] = analyze_passage(st.session_state["p_original"], st.session_state["p_rewritten"])


def _counts_text(findings: list[Finding]) -> str:
    c = {s: sum(f.severity.value == s for f in findings) for s in ("HIGH", "MEDIUM", "REVIEW")}
    parts = [f"{c[s]} {s.lower()}" for s in ("HIGH", "MEDIUM", "REVIEW") if c[s]]
    return " · ".join(parts) if parts else "no warnings"


def alignment_map(res: PassageResult) -> str:
    rows = []
    for n, seg in enumerate(res.segments, start=1):
        top = next((s for s in ("HIGH", "MEDIUM", "REVIEW") if any(f.severity.value == s for f in seg.findings)), None)
        sev = f'<span class="cg-pill sev-{top.lower()}">{top.title()}</span>' if top else '<span class="cg-pill sev-none">Clear</span>'
        o = "".join(f"<p>{e(x)}</p>" for x in seg.original_sentences) or '<p class="none">—</p>'
        r = "".join(f"<p>{e(x)}</p>" for x in seg.rewritten_sentences) or '<p class="none">—</p>'
        rows.append(f'<div class="cg-map-row"><div class="cg-map-n">{n}</div>'
                    f'<div class="cg-map-kind"><span class="cg-kindtag k-{seg.kind}">{KIND_LABEL[seg.kind]}</span>{sev}</div>'
                    f'<div class="cg-map-o">{o}</div><div class="cg-map-arrow">→</div><div class="cg-map-r">{r}</div></div>')
    return ('<div class="cg-map"><div class="cg-map-head"><div>#</div><div>Alignment</div><div>Original</div><div></div>'
            '<div>Rewritten</div></div>' + "".join(rows) + "</div>")


def passage_tab() -> None:
    st.session_state.setdefault("p_original", "")
    st.session_state.setdefault("p_rewritten", "")
    put('<p class="cg-intro" style="margin-top:1rem">Paste a whole notice or paragraph and its rewrite. ClauseGuard '
        'splits both into sentences, aligns them by shared vocabulary (including sentences that were merged or '
        'split), compares every aligned group, and reports sentences that were dropped or added.</p>')
    left, right = st.columns(2, gap="medium")
    with left:
        put('<div class="cg-field-label"><b>Original passage</b><span>one rule per sentence or line</span></div>')
        st.text_area("Original passage", key="p_original", height=200, label_visibility="collapsed",
                     placeholder="Paste the source notice, circular or policy section…")
    with right:
        put('<div class="cg-field-label"><b>Rewritten passage</b><span>summary or simplified version</span></div>')
        st.text_area("Rewritten passage", key="p_rewritten", height=200, label_visibility="collapsed",
                     placeholder="Paste the rewrite to audit…")

    res: PassageResult | None = st.session_state.get("p_result")
    run, sample, _, dl_json, dl_md = st.columns([1.5, 1.5, 1.2, 1.1, 1.25])
    with run:
        st.button("Compare passages", type="primary", width="stretch", on_click=_run_passage)
    with sample:
        st.button("Load sample passage", width="stretch", on_click=_load_passage_sample)
    with dl_json:
        st.download_button("JSON", data=res.to_json() if res else "{}", icon=":material/download:",
                           file_name="clauseguard-passage.json", mime="application/json",
                           disabled=res is None, width="stretch", key="p_dl_json")
    with dl_md:
        st.download_button("Report", data=passage_to_markdown(res) if res else "", icon=":material/description:",
                           file_name="clauseguard-passage.md", mime="text/markdown",
                           disabled=res is None, width="stretch", key="p_dl_md")
    if res is None:
        return

    put(status_block(res.status.value, res.severity_counts()))
    kinds = [s.kind for s in res.segments]
    compared = sum(k not in ("dropped", "added") for k in kinds)
    section("Alignment", f"{len(res.original_sentences)} original · {len(res.rewritten_sentences)} rewritten sentences · "
            f"{compared} groups compared · {kinds.count('dropped')} dropped · {kinds.count('added')} added")
    put(alignment_map(res))

    section("Details by group", "Groups with HIGH or MEDIUM warnings are expanded")
    for n, seg in enumerate(res.segments, start=1):
        flagged = any(f.severity.value in ("HIGH", "MEDIUM") for f in seg.findings)
        label = f"**{n} · {KIND_LABEL[seg.kind]}** — {_counts_text(seg.findings)}"
        with st.expander(label, expanded=flagged):
            if seg.result is not None:
                if seg.result.original_analysis is not None:
                    put(comparison(seg.result))
                if seg.result.findings:
                    put('<div style="height:.9rem"></div>' + findings_html(seg.result.findings))
                else:
                    put('<div class="cg-none" style="margin-top:.9rem">No configured check fired for this group.</div>')
            else:
                put(findings_html(seg.findings))
    put('<ul class="cg-notes">' + "".join(f"<li>{e(n)}</li>" for n in res.notes) + "</ul>")


# --------------------------------------------------------------------------- batch tab

BATCH_LIMIT = 2000
TEMPLATE_CSV = ("id,original,rewritten,expected_drift\n"
                "1,Candidates must submit the form before Friday.,Candidates may submit the form before Friday.,1\n"
                "2,Submit the application before 15 November.,The application must be submitted before 15 November.,0\n")


def _column(df: pd.DataFrame, *names: str) -> str | None:
    lower = {c.lower().strip(): c for c in df.columns}
    return next((lower[n] for n in names if n in lower), None)


@st.cache_data(show_spinner=False, max_entries=8)
def run_batch(pairs: tuple[tuple[str, str, str], ...]) -> pd.DataFrame:
    rows = []
    for pid, o, r in pairs:
        res = analyze(o, r)
        c = res.severity_counts()
        rows.append({"id": pid, "status": res.status.value, "high": c["HIGH"], "medium": c["MEDIUM"],
                     "review": c["REVIEW"], "drift": int(res.has_drift()),
                     "categories": ", ".join(sorted(res.predicted_categories())),
                     "rules": ", ".join(f.rule_id for f in res.findings), "original": o, "rewritten": r})
    return pd.DataFrame(rows)


def _use_benchmark() -> None:
    st.session_state["batch_source"] = "benchmark"


def batch_tab() -> None:
    put('<p class="cg-intro" style="margin-top:1rem">Audit many sentence pairs at once. Upload a CSV with an '
        '<code>original</code> and a <code>rewritten</code> column (optionally <code>id</code> and '
        '<code>expected_drift</code> as 0/1 to score the detector against your own labels).</p>')
    up, bench, tmpl = st.columns([2.6, 1.4, 1.2], vertical_alignment="bottom")
    with up:
        uploaded = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
    with bench:
        st.button("Use bundled benchmark", width="stretch", on_click=_use_benchmark,
                  help=f"Runs all pairs in data/benchmark.csv")
    with tmpl:
        st.download_button("CSV template", data=TEMPLATE_CSV, file_name="clauseguard-template.csv",
                           mime="text/csv", width="stretch", icon=":material/download:")

    if uploaded is not None:
        st.session_state["batch_source"] = "upload"
        try:
            df = pd.read_csv(uploaded, keep_default_na=False)
        except Exception as exc:  # malformed CSV
            put(f'<div class="cg-none">Could not read the CSV file: {e(str(exc))}</div>')
            return
        source = e(uploaded.name)
    elif st.session_state.get("batch_source") == "benchmark":
        df = pd.read_csv(ROOT / "data" / "benchmark.csv", keep_default_na=False)
        source = "data/benchmark.csv"
    else:
        return

    o_col = _column(df, "original", "original_text", "source")
    r_col = _column(df, "rewritten", "rewritten_text", "rewrite", "summary")
    if o_col is None or r_col is None:
        put('<div class="cg-none">The CSV needs an <code>original</code> and a <code>rewritten</code> column. '
            'Download the template for the expected format.</div>')
        return
    if len(df) > BATCH_LIMIT:
        put(f'<p class="cg-stale">Only the first {BATCH_LIMIT} rows are analysed.</p>')
        df = df.head(BATCH_LIMIT)
    id_col = _column(df, "id", "example_id")
    ids = df[id_col].astype(str) if id_col else pd.Series(range(1, len(df) + 1)).astype(str)
    pairs = tuple(zip(ids, df[o_col].astype(str), df[r_col].astype(str)))
    with st.spinner(f"Analysing {len(pairs)} pairs…"):
        out = run_batch(pairs)

    counts = out["status"].value_counts()
    section("Summary", f"{len(out)} pairs from {source}")
    status_rows = "".join(
        f"<tr><td>{e(sv.value)}</td><td class='num'>{int(counts.get(sv.value, 0))}</td>"
        f"<td class='num'>{_bar(counts.get(sv.value, 0) / len(out))}{counts.get(sv.value, 0) / len(out):.0%}</td></tr>"
        for sv in Status)
    status_table = ("<div class='cg-tablewrap'><table class='cg-table'><thead><tr><th>Status</th><th class='num'>Pairs</th>"
                    "<th class='num'>Share</th></tr></thead><tbody>" + status_rows + "</tbody></table></div>")

    gold_col = _column(df, "expected_drift", "label", "drift")
    gold = pd.to_numeric(df[gold_col], errors="coerce") if gold_col else None
    if gold is not None and gold.notna().all() and set(gold.unique()) <= {0, 1}:
        g, pr = gold.astype(int).to_numpy(), out["drift"].to_numpy()
        m = prf(int(((g == 1) & (pr == 1)).sum()), int(((g == 0) & (pr == 1)).sum()),
                int(((g == 1) & (pr == 0)).sum()), int(((g == 0) & (pr == 0)).sum()))
        c1, c2 = st.columns([1.2, 1], gap="large")
        with c1:
            put(status_table)
            put(f"<p class='cg-prov'>Scored against <code>{e(gold_col)}</code>: precision {_pct(m['precision'])} · "
                f"recall {_pct(m['recall'])} · F1 {_pct(m['f1'])} · accuracy {_pct(m['accuracy'])}</p>")
        with c2:
            put(confusion(m))
    else:
        put(status_table)

    section("Results", "Select a row to inspect it")
    show = st.segmented_control("Show", ["All", "Drift", "Review", "None"], default="All",
                                key="batch_filter", label_visibility="collapsed")
    view = out
    if show == "Drift":
        view = out[out["status"] == Status.DRIFT.value]
    elif show == "Review":
        view = out[out["status"] == Status.REVIEW.value]
    elif show == "None":
        view = out[out["status"] == Status.NONE.value]
    event = st.dataframe(
        view[["id", "status", "high", "medium", "review", "categories", "original", "rewritten", "rules"]],
        hide_index=True, width="stretch", height=380, on_select="rerun", selection_mode="single-row",
        column_config={
            "id": st.column_config.TextColumn("ID", width="small"),
            "status": st.column_config.TextColumn("Status", width="medium"),
            "high": st.column_config.NumberColumn("High", width="small"),
            "medium": st.column_config.NumberColumn("Med.", width="small"),
            "review": st.column_config.NumberColumn("Rev.", width="small"),
            "categories": st.column_config.TextColumn("Categories"),
            "original": st.column_config.TextColumn("Original", width="large"),
            "rewritten": st.column_config.TextColumn("Rewritten", width="large"),
            "rules": st.column_config.TextColumn("Rules"),
        })
    st.download_button("Download results CSV", data=out.to_csv(index=False), file_name="clauseguard-batch.csv",
                       mime="text/csv", icon=":material/download:")

    selected = event.selection.rows if event and event.selection else []
    if selected:
        row = view.iloc[selected[0]]
        section(f"Pair {row['id']}", "Re-analysed with the same engine")
        render_result(analyze(row["original"], row["rewritten"]))


# --------------------------------------------------------------------------- benchmark tab

def _pct(x) -> str:
    return "—" if x is None else f"{x:.3f}"


def _bar(x, warn: bool = False) -> str:
    if x is None:
        return ""
    if warn:  # on benign rows a flag is a false alarm, so it gets the warning colour
        return f'<span class="cg-bar-track"><span class="cg-bar warn" style="width:{x * 56:.0f}px"></span></span>'
    return f'<span class="cg-bar-track"><span class="cg-bar" style="width:{x * 56:.0f}px"></span></span>'


def metric_table(s: dict) -> str:
    def row(name: str, m: dict) -> str:
        return (f"<tr><td>{name}</td><td class='num'>{_bar(m['precision'])}{_pct(m['precision'])}</td>"
                f"<td class='num'>{_bar(m['recall'])}{_pct(m['recall'])}</td>"
                f"<td class='num'>{_pct(m['f1'])}</td><td class='num'>{_pct(m.get('accuracy'))}</td></tr>")
    return ("<div class='cg-tablewrap'><table class='cg-table'><thead><tr><th>Decision rule</th>"
            "<th class='num'>Precision</th><th class='num'>Recall</th><th class='num'>F1</th>"
            "<th class='num'>Accuracy</th></tr></thead><tbody>"
            + row("HIGH or MEDIUM = drift <span style='color:var(--ink-3)'>(primary)</span>", s["pair_level"])
            + row("Any warning incl. REVIEW = drift", s["pair_level_strict"])
            + "</tbody></table></div>")


def confusion(m: dict) -> str:
    return f"""
    <div class="cg-cm">
      <div class="h"></div><div class="h">Predicted drift</div><div class="h">Predicted no drift</div>
      <div class="h">Labelled drift</div><div class="v good">{m['tp']}<small>true positive</small></div>
        <div class="v {'bad' if m['fn'] else ''}">{m['fn']}<small>false negative</small></div>
      <div class="h">Labelled no drift</div><div class="v {'bad' if m['fp'] else ''}">{m['fp']}<small>false positive</small></div>
        <div class="v good">{m['tn']}<small>true negative</small></div>
    </div>"""


def category_table(cats: dict) -> str:
    rows = ""
    for name, m in cats.items():
        label = "Micro average" if name == "micro_avg" else CATEGORY_INFO[name][0]
        cls = " class='total'" if name == "micro_avg" else ""
        rows += (f"<tr{cls}><td>{label}</td><td class='num'>{m['support']}</td><td class='num'>{m['tp']}</td>"
                 f"<td class='num'>{m['fp']}</td><td class='num'>{m['fn']}</td><td class='num'>{_pct(m['precision'])}</td>"
                 f"<td class='num'>{_pct(m['recall'])}</td><td class='num'>{_pct(m['f1'])}</td></tr>")
    return ("<div class='cg-tablewrap'><table class='cg-table'><thead><tr><th>Category</th><th class='num'>Support</th>"
            "<th class='num'>TP</th><th class='num'>FP</th><th class='num'>FN</th><th class='num'>P</th>"
            "<th class='num'>R</th><th class='num'>F1</th></tr></thead><tbody>" + rows + "</tbody></table></div>")


BENIGN = {"benign_paraphrase", "unchanged"}


def mutation_table(muts: dict) -> str:
    rows = "".join(
        f"<tr><td>{e(k.replace('_', ' '))}</td><td class='num'>{v['pairs']}</td><td class='num'>{v['flagged_as_drift']}</td>"
        f"<td class='num'>{_bar(v['flag_rate'], warn=k in BENIGN)}{_pct(v['flag_rate'])}</td></tr>"
        for k, v in sorted(muts.items())
    )
    return ("<div class='cg-tablewrap'><table class='cg-table'><thead><tr><th>Mutation type</th><th class='num'>Pairs</th>"
            "<th class='num'>Flagged</th><th class='num'>Flag rate</th></tr></thead><tbody>" + rows + "</tbody></table></div>"
            "<p class='cg-prov'>Amber bars mark rewrites labelled as meaning-preserving, where every flag is a false alarm.</p>")


def errors_table(errors: list[dict], split: str) -> str:
    rows = [x for x in errors if split == "all" or x["split"] == split]
    if not rows:
        return "<div class='cg-none'>No misclassified pairs in this split.</div>"
    body = "".join(
        f"<tr><td>{e(x['example_id'])}</td><td><span class='cg-kind {'fn' if x['expected_drift'] else 'fp'}'>"
        f"{'FN' if x['expected_drift'] else 'FP'}</span></td>"
        f"<td>{e(x['original_text'])}<br><span style='color:var(--ink-3)'>→ {e(x['rewritten_text'])}</span></td>"
        f"<td class='muted'>{e(x['findings'] or 'no findings')}</td></tr>"
        for x in rows
    )
    return ("<div class='cg-tablewrap'><table class='cg-table cg-err'><thead><tr><th>ID</th><th>Error</th>"
            "<th>Pair</th><th>Findings</th></tr></thead><tbody>" + body + "</tbody></table></div>")


def benchmark_tab() -> None:
    report = st.session_state.get("live_report")
    source = "live run in this session"
    if report is None:
        if not RESULTS_PATH.exists():
            put("<div class='cg-none'>No saved results yet. Run <code>python scripts/evaluate.py</code> "
                "or use the button below.</div>")
        else:
            report = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
            source = "reports/evaluation_results.json"

    top_l, top_r = st.columns([3, 1.2])
    with top_r:
        if st.button("Re-run evaluation", width="stretch",
                     help="Runs scripts/evaluate.py's evaluate() on data/benchmark.csv in this session"):
            from scripts.evaluate import DATA, evaluate
            with st.spinner("Running the engine on every benchmark pair…"):
                st.session_state["live_report"], _ = evaluate(pd.read_csv(DATA, keep_default_na=False))
            st.rerun()
    if report is None:
        return

    splits = report["splits"]
    with top_l:
        options = [k for k in ("test", "dev", "all") if k in splits]
        names = {"test": "Test · held out", "dev": "Dev · used for tuning", "all": "All pairs"}
        split = st.radio("Split", options, format_func=names.get, horizontal=True, label_visibility="collapsed")
        put(f"<p class='cg-prov'>source: {e(source)} · generated {e(report['generated_at'])} · "
            f"engine {e(report['engine_version'])} · dataset sha256 {e(report['dataset_sha256'][:12])}…</p>")

    s = splits[split]
    section("Pair-level detection", f"{s['pairs']} sentence pairs")
    c1, c2 = st.columns([1.6, 1], gap="large")
    with c1:
        put(metric_table(s))
    with c2:
        put(confusion(s["pair_level"]))

    section("By category", "Counted once per pair; HIGH/MEDIUM findings only")
    put(category_table(s["category_level"]))

    section("By mutation type", "How often each kind of rewrite was flagged as drift")
    put(mutation_table(s["by_mutation_type"]))

    section("Misclassified pairs", "Pair-level errors under the primary decision rule")
    put(errors_table(report["errors"], split))

    put("""<div class="cg-callout"><b>Reading these numbers.</b> The benchmark is small (152 pairs from 52 base
    sentences) and was written by the same author as the rules, so scores overstate how the detector would do on
    unseen text. The dev split was used to tune the rules; the test split was evaluated once after the rules were
    frozen. Pairs labelled <i>out of scope</i> (e.g. ‘and’ → ‘or’) are kept on purpose to show what ClauseGuard
    cannot see.</div>""")


# --------------------------------------------------------------------------- method tab

def method_tab() -> None:
    section("How a comparison works")
    put("""
    <ol class="cg-steps">
      <li><b>Normalise</b><span>Collapse whitespace, unify quotes and dashes, keep character offsets for evidence. <code>extractors.normalize_text</code></span></li>
      <li><b>Extract</b><span>Regular expressions find negation, modality, condition and quantifier markers and map each to a normalised label; numbers, amounts, dates and times are parsed into comparison keys. <code>extractors.analyze_sentence</code></span></li>
      <li><b>Compare</b><span>Each category rule takes the multiset difference of labels between the two sentences, pairs removed and added markers, and assigns a severity from a fixed table. <code>rules.py</code></span></li>
      <li><b>Merge known paraphrases</b><span>Validated patterns such as <code>unless X</code> ≈ <code>without X</code> turn two opposing warnings into one REVIEW note. <code>analyzer.py</code></span></li>
      <li><b>Report</b><span>Findings are sorted by severity; the overall status is derived from the most severe one.</span></li>
    </ol>""")

    section("Passage mode")
    put("""<p class="cg-prose">For paragraphs, the text is first split into sentences (line breaks and bullets also
    separate items; abbreviations such as <code>Rs.</code>, <code>No.</code> and <code>p.m.</code> do not). Each sentence
    is linked to its most similar sentence on the other side, using the mean of the Jaccard overlap and the overlap
    coefficient of content-word stems (threshold 0.3). Linked sentences form groups, so merged and split sentences
    are compared together. Unlinked sentences are reported as dropped or added: MEDIUM if they contain rule markers
    or values, otherwise REVIEW. <code>src/passage.py</code></p>""")

    section("Command line")
    put("""<p class="cg-prose">The same engine runs without the UI:
    <code>python -m src "original" "rewrite"</code>, with <code>--json</code> or <code>--markdown</code> for
    structured output and <code>--passage file1.txt file2.txt</code> for paragraphs. The exit code is 0 (no checks
    triggered), 1 (potential drift) or 2 (review only), so it can be used in scripts.</p>""")

    section("Severity")
    rows = "".join(f"<tr><td><span class='cg-pill sev-{k.lower()}'>{k.title()}</span></td><td>{e(v)}</td></tr>"
                   for k, v in SEVERITY_POLICY.items())
    put(f"<div class='cg-tablewrap'><table class='cg-table'><tbody>{rows}</tbody></table></div>")
    put("<p class='cg-prose' style='margin-top:.6rem'>Severity is a heuristic for how much attention a warning "
        "deserves. It is not a probability that the meaning changed.</p>")

    section("What ClauseGuard does not check")
    put("""
    <ul class="cg-prose">
      <li>Changes of subject, object or actor (<i>students</i> → <i>staff</i>) and changes to ordinary content words.</li>
      <li>Conjunctions and logical structure (<i>and</i> → <i>or</i>, <i>either … or</i>), and which clause a marker belongs to.</li>
      <li>Synonyms and antonyms outside its small lexicon (<i>closed</i>/<i>open</i> is listed; most pairs are not).</li>
      <li>Ranges (<i>three to five</i>), month/year conversions and relative dates (<i>next Monday</i>). Exact conversions between minutes, hours, days and weeks are recognised.</li>
      <li>Ability vs permission (<i>can</i>), epistemic vs deontic <i>may</i>, and future <i>will</i>.</li>
      <li>Paragraph structure beyond sentence alignment: in the Passage tab, sentences are paired by shared vocabulary, so a heavily reworded sentence can be reported as dropped and re-added.</li>
    </ul>""")


def footer() -> None:
    put(f'<div class="cg-foot"><span>ClauseGuard · rule-based NLP · engine <code>{e(ENGINE_VERSION)}</code> · '
        'no external API</span><span>Warnings are review recommendations, not proof of a change in meaning.</span></div>')


# --------------------------------------------------------------------------- page

put(CSS)
masthead()
compare, passage, batch, bench, method = st.tabs(["Compare", "Passage", "Batch audit", "Benchmark", "Method & limits"])
with compare:
    compare_tab()
with passage:
    passage_tab()
with batch:
    batch_tab()
with bench:
    benchmark_tab()
with method:
    method_tab()
footer()
