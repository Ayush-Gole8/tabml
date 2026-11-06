"""Professional PDF report generation for TabML-Guard.

This is the pipeline's headline deliverable: a paginated, print-ready document
assembled from ``results.json`` (its only data source, so the numbers can never
disagree with the run).

Typography — Type 1 only
------------------------
The report uses **exclusively the PDF base-14 fonts**, which are Type 1 and
resident in every conforming PDF viewer:

    Times-Roman / Times-Bold / Times-Italic   body prose
    Helvetica-Bold                            headings and eyebrow labels
    Courier / Courier-Bold                    every number, table cell, config

Nothing is embedded and ``registerFont`` is never called, so the document is
small and renders identically everywhere. Courier for all numerics is deliberate:
being monospaced, decimal points align down a column without any per-cell
alignment tricks.

Page furniture
--------------
A4 portrait. Every page after the title carries a running header (wordmark left,
dataset right, teal rule beneath) and a footer (timestamp left, ``Page N of M``
right). The true page total comes from a two-pass build. A small ``~ Ayush Gole``
watermark appears on the final page only.
"""
from __future__ import annotations

import datetime
import math
from pathlib import Path
from typing import Any, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# --------------------------------------------------------------------------- #
# Palette — matches the pipeline's visual identity
# --------------------------------------------------------------------------- #
INK = colors.HexColor("#14213d")        # ink-navy: headings, header bar
TEAL = colors.HexColor("#0f766e")       # accent: rules, sub-headings, eyebrows
BODY = colors.HexColor("#1f2430")       # near-black body text
MUTED = colors.HexColor("#5b6472")      # captions, footers
RULE = colors.HexColor("#c9d2de")       # hairlines
PANEL = colors.HexColor("#f4f6f9")      # table/panel fill
ZEBRA = colors.HexColor("#fafbfc")      # alternating table rows
GOOD = colors.HexColor("#15803d")
WARN = colors.HexColor("#b45309")
BAD = colors.HexColor("#b91c1c")

PAGE_W, PAGE_H = A4
MARGIN = 20 * mm
CONTENT_W = PAGE_W - 2 * MARGIN

# Type 1 base-14 font names (never embedded)
F_BODY = "Times-Roman"
F_BODY_B = "Times-Bold"
F_BODY_I = "Times-Italic"
F_HEAD = "Helvetica-Bold"
F_SANS = "Helvetica"
F_MONO = "Courier"
F_MONO_B = "Courier-Bold"


# --------------------------------------------------------------------------- #
# Formatting helpers (mirror report.py so both renderers agree)
# --------------------------------------------------------------------------- #
def _is_nan(v: Any) -> bool:
    return isinstance(v, float) and math.isnan(v)


def _num(v: Any, nd: int = 4) -> str:
    """Format a number; en-dash for missing/NaN."""
    if v is None or _is_nan(v):
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f == int(f) and abs(f) >= 1000:
        return f"{int(f):,}"
    return f"{f:.{nd}f}"


def _pct(v: Any, nd: int = 1) -> str:
    if v is None or _is_nan(v):
        return "-"
    return f"{float(v) * 100:.{nd}f}%"


def _secs(v: Any) -> str:
    if v is None or _is_nan(v):
        return "-"
    return f"{float(v):.1f}s"


def esc(x: Any) -> str:
    """Escape for reportlab's mini-HTML paragraph markup."""
    return (str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


# --------------------------------------------------------------------------- #
# Paragraph styles
# --------------------------------------------------------------------------- #
def _styles() -> dict:
    ss = getSampleStyleSheet()
    s = {}
    s["title"] = ParagraphStyle(
        "tg-title", parent=ss["Normal"], fontName=F_HEAD, fontSize=25, leading=29,
        textColor=INK, spaceAfter=4,
    )
    s["subtitle"] = ParagraphStyle(
        "tg-subtitle", parent=ss["Normal"], fontName=F_BODY, fontSize=12, leading=16,
        textColor=MUTED, spaceAfter=2,
    )
    s["eyebrow"] = ParagraphStyle(
        "tg-eyebrow", parent=ss["Normal"], fontName=F_HEAD, fontSize=7.5, leading=10,
        textColor=TEAL, spaceAfter=2,
    )
    s["h1"] = ParagraphStyle(
        "tg-h1", parent=ss["Normal"], fontName=F_HEAD, fontSize=13.5, leading=17,
        textColor=INK, spaceBefore=2, spaceAfter=5,
    )
    s["h2"] = ParagraphStyle(
        "tg-h2", parent=ss["Normal"], fontName=F_HEAD, fontSize=10.5, leading=14,
        textColor=TEAL, spaceBefore=8, spaceAfter=3,
    )
    s["body"] = ParagraphStyle(
        "tg-body", parent=ss["Normal"], fontName=F_BODY, fontSize=10, leading=14.2,
        textColor=BODY, alignment=TA_LEFT, spaceAfter=5,
    )
    s["small"] = ParagraphStyle(
        "tg-small", parent=ss["Normal"], fontName=F_BODY, fontSize=8.5, leading=11.5,
        textColor=MUTED, spaceAfter=3,
    )
    s["caption"] = ParagraphStyle(
        "tg-caption", parent=ss["Normal"], fontName=F_BODY_I, fontSize=8, leading=10.5,
        textColor=MUTED, alignment=TA_CENTER, spaceBefore=2, spaceAfter=7,
    )
    s["mono"] = ParagraphStyle(
        "tg-mono", parent=ss["Normal"], fontName=F_MONO, fontSize=8.2, leading=11,
        textColor=BODY, spaceAfter=3,
    )
    s["verdict_big"] = ParagraphStyle(
        "tg-vbig", parent=ss["Normal"], fontName=F_MONO_B, fontSize=21, leading=24,
        textColor=INK,
    )
    s["verdict_lbl"] = ParagraphStyle(
        "tg-vlbl", parent=ss["Normal"], fontName=F_HEAD, fontSize=7.5, leading=10,
        textColor=TEAL, spaceAfter=1,
    )
    s["toc"] = ParagraphStyle(
        "tg-toc", parent=ss["Normal"], fontName=F_BODY, fontSize=10, leading=15,
        textColor=BODY,
    )
    s["sign"] = ParagraphStyle(
        "tg-sign", parent=ss["Normal"], fontName=F_BODY_I, fontSize=9, leading=12,
        textColor=colors.HexColor("#9aa4b2"), alignment=TA_CENTER,
    )
    return s


ST = _styles()


# --------------------------------------------------------------------------- #
# Flowable builders
# --------------------------------------------------------------------------- #
def _para(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, ST[style])


def _rule(color=RULE, thickness: float = 0.6, space: float = 4) -> Table:
    """A horizontal hairline spanning the content width."""
    t = Table([[""]], colWidths=[CONTENT_W], rowHeights=[0.1])
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), thickness, color),
        ("TOPPADDING", (0, 0), (-1, -1), space),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return t


def _section(sid: str, eyebrow: str, title: str) -> list:
    """Section opener: eyebrow label, title, teal rule."""
    return [
        Spacer(1, 7),
        _para(f"{esc(eyebrow)}", "eyebrow"),
        _para(esc(title), "h1"),
        _rule(TEAL, 0.9, 1),
        Spacer(1, 6),
    ]


def _table(headers: list, rows: list, *, col_widths=None, highlight_row: int = -1,
           right_from: int = 1, font_size: float = 8.2) -> Table:
    """A styled data table: sans headers, Courier numerics, zebra rows.

    Cells at/after ``right_from`` are right-aligned monospace, so decimals line up.
    """
    data = [[str(h) for h in headers]] + [[str(c) for c in r] for r in rows]
    if col_widths is None:
        first = CONTENT_W * (0.30 if len(headers) > 2 else 0.5)
        rest = (CONTENT_W - first) / max(1, len(headers) - 1)
        col_widths = [first] + [rest] * (len(headers) - 1)

    t = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    style = [
        # header row
        ("BACKGROUND", (0, 0), (-1, 0), PANEL),
        ("TEXTCOLOR", (0, 0), (-1, 0), INK),
        ("FONTNAME", (0, 0), (-1, 0), F_SANS),
        ("FONTSIZE", (0, 0), (-1, 0), font_size - 0.4),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, TEAL),
        # body
        ("FONTNAME", (0, 1), (right_from - 1, -1), F_BODY),
        ("FONTNAME", (right_from, 1), (-1, -1), F_MONO),
        ("FONTSIZE", (0, 1), (-1, -1), font_size),
        ("TEXTCOLOR", (0, 1), (-1, -1), BODY),
        ("ALIGN", (right_from, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, RULE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ZEBRA]),
    ]
    if highlight_row >= 0:
        r = highlight_row + 1
        style += [
            ("BACKGROUND", (0, r), (-1, r), colors.HexColor("#e6f4f1")),
            ("FONTNAME", (0, r), (right_from - 1, r), F_BODY_B),
            ("FONTNAME", (right_from, r), (-1, r), F_MONO_B),
        ]
    t.setStyle(TableStyle(style))
    return t


def _callout(title: str, body: str, kind: str = "note") -> Table:
    """A boxed note with a coloured left edge."""
    edge = {"note": TEAL, "good": GOOD, "warn": WARN, "bad": BAD}.get(kind, TEAL)
    inner = []
    if title:
        inner.append(_para(f"<b>{esc(title)}</b>", "h2"))
    inner.append(_para(body, "body"))
    t = Table([[inner]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PANEL),
        ("LINEBEFORE", (0, 0), (0, -1), 2.6, edge),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


#: Tallest a figure may be. The usable frame height is roughly
#: PAGE_H - 2*MARGIN - header/footer furniture; capping well under that leaves
#: room for the caption and guarantees reportlab never raises "too large on page".
MAX_FIG_H = (PAGE_H - 2 * MARGIN - 30 * mm) * 0.62


def _figure(fig_dir: Path, name: str, caption: str, max_w: float = None) -> list:
    """Embed a PNG scaled to fit both the content width and the frame height.

    Scaling on width alone is not enough: a tall figure that fits the width can
    still overflow the frame, which reportlab treats as a fatal LayoutError.
    """
    path = fig_dir / f"{name}.png"
    if not path.exists():
        return []
    try:
        from reportlab.lib.utils import ImageReader

        iw, ih = ImageReader(str(path)).getSize()
        if not iw or not ih:
            return []
        avail_w = max_w or (CONTENT_W * 0.86)
        scale = min(1.0, avail_w / iw, MAX_FIG_H / ih)
        w, h = iw * scale, ih * scale
        img = Image(str(path), width=w, height=h)
        img.hAlign = "CENTER"
        # Keep the figure with its caption so they never split across a page.
        return [Spacer(1, 4), KeepTogether([img, _para(esc(caption), "caption")])]
    except Exception:
        return []


def _kv_table(pairs: list, label_w: float = 0.34) -> Table:
    """Two-column key/value table: serif labels, Courier values."""
    lw = CONTENT_W * label_w
    t = Table([[k, v] for k, v in pairs], colWidths=[lw, CONTENT_W - lw], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), F_BODY),
        ("FONTNAME", (1, 0), (1, -1), F_MONO),
        ("FONTSIZE", (0, 0), (-1, -1), 8.6),
        ("TEXTCOLOR", (0, 0), (0, -1), MUTED),
        ("TEXTCOLOR", (1, 0), (1, -1), BODY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("LINEBELOW", (0, 0), (-1, -2), 0.25, RULE),
    ]))
    return t


# --------------------------------------------------------------------------- #
# Insight derivation (plain-English verdicts)
# --------------------------------------------------------------------------- #
def _metric_keys(task: str) -> list:
    if task == "classification":
        return ["roc_auc", "accuracy", "f1", "precision", "recall"]
    return ["rmse", "mae", "r2"]


def _merged_leaderboard(ctx: dict) -> list:
    for stage in ("tuning", "boosting", "baselines"):
        lb = (ctx.get(stage) or {}).get("leaderboard")
        if lb:
            return lb
    return []


def _score_of(lb: list, name: str) -> Optional[float]:
    for row in lb:
        if row.get("name") == name:
            v = row.get("primary")
            return None if v is None or _is_nan(v) else float(v)
    return None


def _family(name: str, ctx: dict) -> str:
    """Which stage a model came from — used for the 'simple model wins' insight.

    Stage 3 does not publish a ``models`` list (only ``records`` + a leaderboard),
    so tuned models are recognised by their ``<booster>_<strategy>`` name instead.
    """
    if not name:
        return "unknown"

    tun = ctx.get("tuning") or {}
    strategies = {r.get("strategy") for r in (tun.get("records") or [])
                  if isinstance(r, dict)}
    if name == tun.get("tuned_best") or name.rsplit("_", 1)[-1] in strategies:
        return "tuned"

    for stage, label in (("baselines", "baseline"), ("boosting", "ensemble")):
        names = [m.get("name") for m in (ctx.get(stage) or {}).get("models", []) or []]
        if name in names:
            return label
    if name.startswith(("logreg", "linear", "ridge", "lasso", "elastic")):
        return "baseline"
    return "ensemble"


def _better(a: Optional[float], b: Optional[float], primary: str) -> bool:
    """Is a better than b for this metric? (lower is better for error metrics)"""
    if a is None or b is None:
        return False
    lower_better = primary in {"rmse", "mae"}
    return (a < b) if lower_better else (a > b)


def _strategy_score(tuning: dict, lb: list, strategy: str,
                    primary: str) -> Optional[float]:
    """Held-out score for one tuning strategy.

    Stage 3 does not store a test score on its ``records`` entries - each
    strategy's winning configuration is refit once and registered on the
    leaderboard as ``<booster>_<strategy>`` (e.g. ``lightgbm_random``). This
    resolves that name and returns the score, so the report never has to guess.
    """
    booster = (tuning or {}).get("booster") or ""
    if not booster:
        return None
    v = _score_of(lb, f"{booster}_{strategy}")
    if v is None and strategy == "default":
        # some runs register the untuned model under the bare booster name
        v = _score_of(lb, booster)
    return v


def _insights(ctx: dict, task: str, primary: str) -> dict:
    """Derive the headline verdict, status pills and recommendation."""
    lb = _merged_leaderboard(ctx)
    champion = lb[0]["name"] if lb else ""
    champ_score = _score_of(lb, champion) if champion else None
    pills, rec_bits = [], []
    headline = ""

    fam = _family(champion, ctx)
    complexity_note = ""
    if fam == "baseline":
        pills.append(("Simple model wins", "good"))
        complexity_note = (
            "A regularized linear baseline matched or beat every ensemble here. "
            "That is a finding, not a failure: it says the signal in this dataset is "
            "largely additive, and it argues for shipping the cheaper, more "
            "interpretable model."
        )
    elif fam == "tuned":
        # No pill here: whether tuning actually *paid off* is decided by the
        # cost/benefit block below, which compares against the default config.
        pass
    else:
        pills.append(("Ensemble wins", "neutral"))

    # --- tuning cost/benefit -------------------------------------------------
    tun = ctx.get("tuning") or {}
    recs = tun.get("records") or []
    by = {r.get("strategy"): r for r in recs if isinstance(r, dict)}
    default_s = _strategy_score(tun, lb, "default", primary)
    best_tuned, best_tuned_name = None, ""
    for k in by:
        if k == "default":
            continue
        v = _strategy_score(tun, lb, k, primary)
        if v is not None and (best_tuned is None or _better(v, best_tuned, primary)):
            best_tuned, best_tuned_name = float(v), k
    if default_s is not None and best_tuned is not None:
        gain = abs(best_tuned - float(default_s))
        rel = gain / max(1e-12, abs(float(default_s)))
        tuning_helped = _better(best_tuned, float(default_s), primary)
        if not tuning_helped:
            pills.append(("Defaults beat tuning", "warn"))
            rec_bits.append(
                "No search beat the library defaults on the held-out period, so the "
                "default configuration is the pragmatic production choice."
            )
        elif rel < 0.01:
            pills.append(("Tuning: marginal gain", "warn"))
            rec_bits.append(
                "Hyperparameter tuning moved the score by less than 1% over defaults, "
                "so the default configuration is a defensible production choice."
            )
        else:
            pills.append(("Tuning: real gain", "good"))
            rec_bits.append(
                f"Tuning improved the score by {rel * 100:.1f}% over defaults "
                f"(best strategy: {best_tuned_name})."
            )
    # is Bayesian worth it at equal budget? (the strategy is recorded as 'bayes')
    bay_key = "bayes" if "bayes" in by else "bayesian"
    rnd, bay = by.get("random"), by.get(bay_key)
    if rnd and bay:
        rs = _strategy_score(tun, lb, "random", primary)
        bs = _strategy_score(tun, lb, bay_key, primary)
        rt, bt = rnd.get("wall_seconds"), bay.get("wall_seconds")
        if None not in (rs, bs, rt, bt) and not any(_is_nan(x) for x in (rs, bs, rt, bt)):
            if _better(bs, rs, primary):
                ratio = float(bt) / max(1e-9, float(rt))
                rec_bits.append(
                    f"At an equal budget Bayesian search edged out random search, "
                    f"but cost {ratio:.1f}x the wall-clock."
                )
            else:
                rec_bits.append(
                    "At an equal budget Bayesian search did not beat random search - "
                    "on a space this size, random is the pragmatic default."
                )

    # --- drift ---------------------------------------------------------------
    dr = ctx.get("drift") or {}
    trend = dr.get("trend") or {}
    fd = dr.get("feature_drift") or {}
    worsened = bool(trend.get("worse_over_time"))
    max_psi = fd.get("max_mean_psi")
    inputs_moved = max_psi is not None and not _is_nan(max_psi) and float(max_psi) > 0.25
    if worsened and inputs_moved:
        pills.append(("Drift detected", "bad"))
        rec_bits.append("Both the inputs and the accuracy moved over time: schedule retraining.")
    elif inputs_moved and not worsened:
        pills.append(("Inputs drifted, performance held", "warn"))
        rec_bits.append(
            "The input distribution shifted materially (PSI above 0.25) while accuracy "
            "held up - monitor, but no retrain is urgent yet."
        )
    elif worsened:
        pills.append(("Performance decayed", "warn"))
    elif dr.get("status") == "ok":
        pills.append(("Stable over time", "good"))

    # --- reliability triage --------------------------------------------------
    ex = ctx.get("explain") or {}
    rel = ex.get("reliability") or {}
    flag = rel.get("flag_rule") or {}
    ecr = flag.get("error_capture_rate")
    if ecr is not None and not _is_nan(ecr):
        pills.append((f"Triage flag catches {_pct(ecr, 0)} of errors", "neutral"))
        rec_bits.append(
            f"Routing the least-confident {_pct(flag.get('coverage'), 0)} of predictions to "
            f"human review would catch {_pct(ecr, 0)} of all errors."
        )

    if champion and champ_score is not None:
        headline = (
            f"{champion} is the recommended model, scoring {_num(champ_score)} "
            f"{primary} on a strictly held-out, time-ordered test set."
        )

    return {
        "champion": champion,
        "champion_score": champ_score,
        "headline": headline,
        "pills": pills,
        "complexity_note": complexity_note,
        "recommendation": " ".join(rec_bits),
        "leaderboard": lb,
    }


# --------------------------------------------------------------------------- #
# Page furniture: header, footer, watermark
# --------------------------------------------------------------------------- #
class _Doc(BaseDocTemplate):
    """Two-pass document: pass 1 counts pages, pass 2 prints 'Page N of M'."""

    def __init__(self, filename: str, *, dataset: str, timestamp: str, **kw):
        super().__init__(filename, pagesize=A4,
                         leftMargin=MARGIN, rightMargin=MARGIN,
                         topMargin=MARGIN + 8 * mm, bottomMargin=MARGIN + 4 * mm,
                         title=kw.pop("doc_title", "TabML-Guard Report"),
                         author="Ayush Gole", **kw)
        self.dataset = dataset
        self.timestamp = timestamp
        self.total_pages = 0          # filled after pass 1
        self._is_final_pass = False

        frame = Frame(self.leftMargin, self.bottomMargin,
                      self.width, self.height, id="body",
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates([
            PageTemplate(id="cover", frames=[frame], onPage=self._cover_furniture),
            PageTemplate(id="body", frames=[frame], onPage=self._page_furniture),
        ])

    # -- pass control ------------------------------------------------------- #
    def run_two_pass(self, story_factory) -> int:
        """Build once to learn the page count, then rebuild with 'of M' filled in.

        ``story_factory`` must return a *fresh* list of flowables on each call:
        reportlab flowables carry layout state and cannot be safely reused across
        two builds.
        """
        self._is_final_pass = False
        self.build(story_factory())
        total = self.page
        # second pass, now that the true total is known
        self.total_pages = total
        self._is_final_pass = True
        self.page = 0
        self.build(story_factory())
        return self.page

    # -- furniture ---------------------------------------------------------- #
    def _footer(self, canv, page_label: str) -> None:
        canv.setFont(F_BODY, 7.6)
        canv.setFillColor(MUTED)
        canv.drawString(MARGIN, MARGIN * 0.62, f"Generated {self.timestamp}")
        canv.drawRightString(PAGE_W - MARGIN, MARGIN * 0.62, page_label)
        canv.setStrokeColor(RULE)
        canv.setLineWidth(0.4)
        canv.line(MARGIN, MARGIN * 0.62 + 9, PAGE_W - MARGIN, MARGIN * 0.62 + 9)

    def _page_label(self) -> str:
        if self._is_final_pass and self.total_pages:
            return f"Page {self.page} of {self.total_pages}"
        return f"Page {self.page}"

    def _cover_furniture(self, canv, doc) -> None:
        canv.saveState()
        self._footer(canv, self._page_label())
        canv.restoreState()

    def _page_furniture(self, canv, doc) -> None:
        canv.saveState()
        # running header
        y = PAGE_H - MARGIN - 2 * mm
        canv.setFont(F_HEAD, 8.4)
        canv.setFillColor(INK)
        canv.drawString(MARGIN, y, "TabML-Guard")
        canv.setFont(F_BODY_I, 8.2)
        canv.setFillColor(MUTED)
        canv.drawRightString(PAGE_W - MARGIN, y, self.dataset)
        canv.setStrokeColor(TEAL)
        canv.setLineWidth(0.8)
        canv.line(MARGIN, y - 3.4 * mm, PAGE_W - MARGIN, y - 3.4 * mm)
        self._footer(canv, self._page_label())
        # watermark: final page only
        if self._is_final_pass and self.total_pages and self.page == self.total_pages:
            canv.setFont(F_BODY_I, 9)
            canv.setFillColor(colors.HexColor("#b9c2cf"))
            canv.drawCentredString(PAGE_W / 2, MARGIN * 0.62 + 20, "~ Ayush Gole")
            canv.saveState()
            canv.translate(PAGE_W / 2, PAGE_H / 2)
            canv.rotate(38)
            canv.setFont(F_BODY_I, 46)
            canv.setFillColor(colors.Color(0.58, 0.63, 0.70, alpha=0.07))
            canv.drawCentredString(0, 0, "~ Ayush Gole")
            canv.restoreState()
        canv.restoreState()


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def _cover(ctx: dict, ins: dict, title: str, dataset: str, task: str,
           primary: str, timestamp: str) -> list:
    prof = ctx.get("profile", {}) or {}
    out: list = [Spacer(1, 26)]
    out.append(_para("EXPLAINABLE - DRIFT-AWARE - COST-ACCOUNTED", "eyebrow"))
    out.append(_para(esc(title), "title"))
    out.append(_rule(INK, 1.5, 3))
    out.append(Spacer(1, 8))
    sub = f"{dataset} &middot; {_num(prof.get('n_rows'))} rows &times; " \
          f"{_num(prof.get('n_features'))} features &middot; {esc(task)}"
    out.append(_para(sub, "subtitle"))
    out.append(Spacer(1, 20))

    # verdict panel
    score_txt = _num(ins["champion_score"]) if ins["champion_score"] is not None else "-"
    left = [
        _para("RECOMMENDED MODEL", "verdict_lbl"),
        _para(esc(ins["champion"] or "-"), "h1"),
        Spacer(1, 5),
        _para(f"{esc(primary.upper())} (HELD-OUT)", "verdict_lbl"),
        _para(esc(score_txt), "verdict_big"),
    ]
    pill_lines = [_para(f"&bull; {esc(t)}", "small") for t, _ in ins["pills"]]
    right = [_para("AT A GLANCE", "verdict_lbl")] + (pill_lines or [_para("-", "small")])
    panel = Table([[left, right]], colWidths=[CONTENT_W * 0.52, CONTENT_W * 0.48])
    panel.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PANEL),
        ("BOX", (0, 0), (-1, -1), 0.7, RULE),
        ("LINEBEFORE", (0, 0), (0, -1), 3, TEAL),
        ("LINEAFTER", (0, 0), (0, -1), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 11),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    out.append(panel)
    out.append(Spacer(1, 14))

    if ins["headline"]:
        out.append(_callout("Headline finding", esc(ins["headline"]), "note"))
        out.append(Spacer(1, 8))
    if ins["recommendation"]:
        out.append(_callout("Recommendation", esc(ins["recommendation"]), "good"))
        out.append(Spacer(1, 8))
    if ins["complexity_note"]:
        out.append(_callout("On model complexity", esc(ins["complexity_note"]), "warn"))

    out.append(Spacer(1, 16))
    out.append(_para(
        "This report was produced automatically by the TabML-Guard pipeline. Every "
        "number below comes from a single run on a strictly held-out, time-ordered "
        "test split - no test data was used for model selection or tuning.", "small"))
    return out


def _toc(entries: list) -> list:
    """Table of contents with dot leaders and real page numbers."""
    out = _section("TOC", "CONTENTS", "Table of contents")
    rows = [[esc(name), str(page)] for name, page in entries]
    t = Table(rows, colWidths=[CONTENT_W * 0.86, CONTENT_W * 0.14], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), F_BODY),
        ("FONTNAME", (1, 0), (1, -1), F_MONO),
        ("FONTSIZE", (0, 0), (-1, -1), 9.4),
        ("TEXTCOLOR", (0, 0), (-1, -1), BODY),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.4),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, RULE),
    ]))
    out.append(t)
    return out


def _skipped(stage: dict, label: str) -> list:
    """Uniform note when a stage did not produce results."""
    status = (stage or {}).get("status", "missing")
    reason = (stage or {}).get("reason") or (stage or {}).get("error") or ""
    body = f"This stage reported status <b>{esc(status)}</b>."
    if reason:
        body += f" {esc(reason)}"
    body += (" The pipeline is designed to degrade gracefully: a missing optional "
             "dependency or an inapplicable stage is skipped with a note rather "
             "than aborting the run.")
    return [_callout(f"{label} unavailable", body, "warn")]


def _sec_dataset(ctx: dict, cfg: Optional[Any], task: str) -> list:
    prof = ctx.get("profile", {}) or {}
    out = _section("S0", "STAGE 0", "Dataset, profile and configuration")
    out.append(_para(
        "The pipeline begins by profiling the raw table and deciding, from the target "
        "column alone, whether this is a classification or a regression problem. When a "
        "time column is present the split is chronological - the model trains on the "
        "past and is scored on the future.", "body"))

    span = prof.get("time_span") or []
    pairs = [
        ("Rows", _num(prof.get("n_rows"))),
        ("Features", _num(prof.get("n_features"))),
        ("Target column", str(prof.get("target", "-"))),
        ("Detected task", str(task)),
        ("Missing values", _num(prof.get("missing_total", 0))),
        ("Time column", str(prof.get("time_col") or "none (random split)")),
    ]
    if len(span) == 2:
        pairs.append(("Time span", f"{span[0]}  ->  {span[1]}"))
    ts = prof.get("target_stats") or {}
    if ts:
        bits = "  ".join(f"{k}={_num(v)}" for k, v in list(ts.items())[:4])
        pairs.append(("Target statistics", bits))
    cb = prof.get("class_balance")
    if cb:
        pairs.append(("Class balance", "  ".join(f"{k}={_num(v, 3)}" for k, v in cb.items())))
    out.append(_kv_table(pairs))

    if cfg is not None:
        out.append(_para("Configuration used", "h2"))
        out.append(_para(
            "This is the entire dataset-specific input to the run. Everything else is "
            "generic pipeline code.", "small"))
        try:
            ds = cfg.dataset
            lines = [
                f"dataset.source        {ds.source}",
                f"dataset.name          {ds.name or '-'}",
                f"dataset.path          {ds.path or '-'}",
                f"dataset.target        {ds.target}",
                f"dataset.time_col      {ds.time_col or 'null'}",
                f"dataset.drop_cols     {list(ds.drop_cols) if ds.drop_cols else '[]'}",
                f"split                 train={cfg.split.train_frac}  test={cfg.split.test_frac}",
                f"models.boosters       {list(cfg.models.boosters)}",
                f"models.stacking       {cfg.models.stacking}",
                f"tuning.budget_trials  {cfg.tuning.budget_trials}",
                f"tuning.cv_splits      {cfg.tuning.cv_splits}",
                f"drift.n_chunks        {cfg.drift.n_chunks}",
                f"drift.retrain         {cfg.drift.retrain}",
                f"random_state          {cfg.random_state}",
            ]
            block = Table([[Paragraph("<br/>".join(esc(l).replace(" ", "&nbsp;")
                                                   for l in lines), ST["mono"])]],
                          colWidths=[CONTENT_W])
            block.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), PANEL),
                ("BOX", (0, 0), (-1, -1), 0.5, RULE),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]))
            out.append(block)
        except Exception:
            pass
    return out


def _leaderboard_rows(lb: list, task: str, primary: str) -> tuple:
    keys = _metric_keys(task)
    headers = ["model"] + keys + ["fit"]
    rows = []
    for row in lb:
        m = row.get("metrics", {}) or {}
        cells = [row.get("name", "-")]
        cells += [_num(m.get(k)) for k in keys]
        cells.append(_secs(row.get("fit_seconds")))
        rows.append(cells)
    return headers, rows


def _sec_baselines(ctx: dict, task: str, primary: str) -> list:
    st = ctx.get("baselines") or {}
    out = _section("S1", "STAGE 1", "Regularized linear baselines")
    if st.get("status") != "ok":
        return out + _skipped(st, "Baselines")
    out.append(_para(
        "Before reaching for an ensemble, the pipeline establishes what a well-regularized "
        "linear model can do. This is the honesty check: if a linear model is competitive, "
        "the extra complexity of boosting has to justify itself.", "body"))
    lb = st.get("leaderboard") or []
    headers, rows = _leaderboard_rows(lb, task, primary)
    best_i = next((i for i, r in enumerate(lb) if r.get("name") == st.get("best")), 0)
    out.append(_table(headers, rows, highlight_row=best_i))
    if st.get("note"):
        out.append(Spacer(1, 5))
        out.append(_para(esc(st["note"]), "small"))

    path = st.get("regularization_path") or {}
    if path.get("values"):
        out.append(_para("Regularization path", "h2"))
        out.append(_para(
            "Sweeping the penalty strength on an internal split of the training data only "
            "(the test set stays untouched) makes the bias-variance trade-off visible, and "
            "shows how aggressively L1 drives coefficients to exactly zero.", "body"))
        vals = path.get("values") or []
        nz = path.get("l1_nonzero") or []
        sc = path.get("l1_score") or []
        sc2 = path.get("l2_score") or []
        rows = []
        for i, v in enumerate(vals):
            rows.append([
                _num(v, 4),
                _num(nz[i]) if i < len(nz) else "-",
                _num(sc[i]) if i < len(sc) else "-",
                _num(sc2[i]) if i < len(sc2) else "-",
            ])
        out.append(_table([path.get("param", "C"), "L1 non-zero coefs",
                           "L1 val score", "L2 val score"], rows))
    return out


def _sec_boosting(ctx: dict, task: str, primary: str) -> list:
    st = ctx.get("boosting") or {}
    out = _section("S2", "STAGE 2", "Ensemble and boosting models")
    if st.get("status") != "ok":
        return out + _skipped(st, "Ensembles")
    out.append(_para(
        "Gradient-boosted trees plus a stacking meta-model, all scored through the same "
        "evaluation path as the baselines so the leaderboard rows are directly comparable.",
        "body"))
    skipped = st.get("skipped_boosters") or []
    if skipped:
        out.append(_para(
            f"Skipped (library not installed): {esc(', '.join(map(str, skipped)))}.", "small"))
    lb = st.get("leaderboard") or []
    headers, rows = _leaderboard_rows(lb, task, primary)
    best_i = next((i for i, r in enumerate(lb) if r.get("name") == st.get("best")), 0)
    out.append(_table(headers, rows, highlight_row=best_i))
    out.append(Spacer(1, 5))
    out.append(_para(
        f"Best ensemble: <b>{esc(st.get('best', '-'))}</b>. The strongest single booster "
        f"(<b>{esc(st.get('best_booster', '-'))}</b>) is the one carried into the tuning "
        f"comparison in Stage 3.", "body"))
    return out


def _sec_tuning(ctx: dict, task: str, primary: str) -> list:
    st = ctx.get("tuning") or {}
    out = _section("S3", "STAGE 3", "Hyperparameter optimization: benefit vs cost")
    if st.get("status") != "ok":
        return out + _skipped(st, "Tuning comparison")
    out.append(_para(
        f"Four strategies compete on <b>{esc(st.get('booster', '-'))}</b>: library defaults, "
        f"an exhaustive coarse grid, random search, and Bayesian optimization (Optuna TPE). "
        f"Random and Bayesian are given an <b>equal budget</b> of "
        f"{_num(st.get('budget_trials'))} trials, which makes them a fair head-to-head. "
        f"Crucially, the table reports not just the score but what each strategy <i>cost</i>.",
        "body"))
    recs = st.get("records") or []
    lb = st.get("leaderboard") or []
    rows, best_i, best_v = [], -1, None
    for i, r in enumerate(recs):
        strategy = r.get("strategy", "-")
        v = _strategy_score(st, lb, strategy, primary)
        rows.append([
            strategy,
            _num(v),
            _secs(r.get("wall_seconds")),
            _num(r.get("n_evaluations")),
            _num(r.get("cv_score")),
        ])
        if v is not None and (best_v is None or _better(v, best_v, primary)):
            best_v, best_i = float(v), i
    out.append(_table(["strategy", f"test {primary}", "wall clock",
                       "model fits", "cv score"], rows, highlight_row=best_i))
    out.append(Spacer(1, 4))
    out.append(_para(
        "The <i>cv score</i> column is the internal cross-validated score each search "
        "optimized (scikit-learn's convention is higher-is-better, so error metrics appear "
        "negated). The <i>test</i> column is the winning configuration refit once and scored "
        "on the untouched held-out period - that is the number to trust.", "small"))
    if st.get("verdict"):
        out.append(Spacer(1, 7))
        out.append(_callout("Is tuning worth it?", esc(st["verdict"]), "note"))
    return out


def _sec_explain(ctx: dict, fig_dir: Path, task: str) -> list:
    st = ctx.get("explain") or {}
    out = _section("S4", "STAGE 4", "Explainability and prediction reliability")
    if st.get("status") != "ok":
        return out + _skipped(st, "Explainability")
    out.append(_para(
        f"SHAP attributions for <b>{esc(st.get('model', '-'))}</b>, using the explainer "
        f"matched to the model family (tree, linear, or model-agnostic fallback). Global "
        f"importance answers <i>what drives this model overall</i>; the local example shows "
        f"how a single prediction was reached.", "body"))

    shap_info = st.get("shap") or {}
    gi = shap_info.get("global_importance") or []
    if gi:
        rows = [[str(f), _num(v, 5)] for f, v in gi[:15]]
        out.append(_table(["feature", "mean |SHAP|"], rows,
                          col_widths=[CONTENT_W * 0.62, CONTENT_W * 0.38]))
    out += _figure(fig_dir, "shap_importance", "Global feature importance (mean |SHAP|)")

    local = shap_info.get("local_example") or {}
    contribs = local.get("contributions") or []
    if contribs:
        out.append(_para("A single prediction, explained", "h2"))
        pred = local.get("prediction")
        out.append(_para(
            f"For one held-out row the model predicted <b>{esc(_num(pred))}</b>. These are "
            f"the largest signed contributions that produced it - positive values push the "
            f"prediction up, negative values pull it down.", "body"))
        rows = [[str(f), _num(v, 5)] for f, v in contribs[:10]]
        out.append(_table(["feature", "SHAP contribution"], rows,
                          col_widths=[CONTENT_W * 0.62, CONTENT_W * 0.38]))

    rel = st.get("reliability") or {}
    if rel.get("bins"):
        out.append(_para("Confidence-based triage", "h2"))
        out.append(_para(
            "A model that knows when it is unsure is far more useful in production than one "
            "that is uniformly confident. Binning the held-out predictions by confidence "
            "shows whether confidence actually tracks correctness - and if it does, the "
            "least-confident slice can be routed to a human.", "body"))
        # Bins are quantile bins: Stage 4 reports the bin index and its mean
        # confidence, not band edges, so label the band by its rank.
        n_bins = len(rel["bins"])
        rows = [[
            f"least confident {i}/{n_bins}" if i == 1 else
            (f"most confident {i}/{n_bins}" if i == n_bins else f"{i}/{n_bins}"),
            _num(b.get("n")),
            _num(b.get("mean_confidence"), 3),
            _pct(b.get("accuracy"), 1),
        ] for i, b in enumerate(rel["bins"], 1)]
        out.append(_table(["confidence band", "n", "mean confidence", "accuracy"], rows))
        if rel.get("error_capture_rate") is not None:
            out.append(Spacer(1, 6))
            out.append(_kv_table([
                ("Flagged for review", _pct(rel.get("coverage_flagged"))),
                ("Of all errors, caught", _pct(rel.get("error_capture_rate"))),
                ("Accuracy on auto-approved", _pct(rel.get("accuracy_auto_accepted"), 1)),
                ("Accuracy on flagged", _pct(rel.get("accuracy_flagged"), 1)),
                ("Confidence cut-off", _num(rel.get("confidence_threshold"), 3)),
            ]))
        out += _figure(fig_dir, "reliability", "Reliability curve: confidence vs observed accuracy")
    elif st.get("reliability_note"):
        out.append(Spacer(1, 5))
        out.append(_para(esc(st["reliability_note"]), "small"))
    return out


def _sec_drift(ctx: dict, fig_dir: Path, task: str) -> list:
    st = ctx.get("drift") or {}
    out = _section("S5", "STAGE 5", "Concept drift and statistical significance")
    if st.get("status") != "ok":
        out += _skipped(st, "Drift analysis")
        out += _significance(st.get("significance"), task)
        return out
    out.append(_para(
        f"The held-out period is replayed in {_num(st.get('n_chunks'))} chronological chunks. "
        f"A <b>frozen</b> model (trained once, never updated) is compared against an "
        f"<b>adaptive</b> model that predicts each chunk <i>before</i> learning from it - so "
        f"there is no look-ahead. Retraining mode: <b>{esc(st.get('retrain_mode', '-'))}</b>.",
        "body"))

    chunks = st.get("chunks") or []
    if chunks:
        rows = []
        for c in chunks:
            rows.append([
                _num(c.get("chunk")),
                _num(c.get("n")),
                _num(c.get("static_score")),
                _num(c.get("adaptive_score")),
                _num(c.get("mean_psi"), 3),
            ])
        out.append(_table(["chunk", "rows", "frozen", "adaptive", "mean PSI"], rows))
        out.append(_para(
            "Chunk 1 is a built-in consistency check: both models are identical there by "
            "construction, since neither has seen the chunk yet.", "small"))

    out += _figure(fig_dir, "drift_performance",
                   "Frozen vs adaptive performance across the held-out period")
    out += _figure(fig_dir, "drift_psi",
                   "Input drift (PSI) against the training window")

    trend = st.get("trend") or {}
    fd = st.get("feature_drift") or {}
    out.append(_kv_table([
        ("Mean score, frozen model", _num(st.get("static_mean"))),
        ("Mean score, adaptive model", _num(st.get("adaptive_mean"))),
        ("First third -> last third", f"{_num(trend.get('first_third'))}  ->  {_num(trend.get('last_third'))}"),
        ("Performance worsened", "yes" if trend.get("worsened") else "no"),
        ("Worst-drifting feature", str(fd.get("worst_feature", "-"))),
        ("Peak mean PSI", _num(fd.get("max_mean_psi"), 3)),
    ]))
    if st.get("verdict"):
        out.append(Spacer(1, 7))
        out.append(_callout("Drift verdict", esc(st["verdict"]),
                            "bad" if trend.get("worsened") else "good"))
    out += _significance(st.get("significance"), task)
    return out


def _significance(sig: Optional[dict], task: str) -> list:
    if not sig:
        return []
    test = "McNemar's test" if task == "classification" else "Wilcoxon signed-rank test"
    out = [_para("Statistical significance", "h2")]
    out.append(_para(
        f"A score difference is only meaningful if it survives a significance test. Paired "
        f"predictions on the same held-out rows are compared with the {test}. A p-value "
        f"below 0.05 means the difference is unlikely to be chance; above it, the two models "
        f"are statistically indistinguishable - which is itself worth reporting.", "body"))
    rows = []
    for label, r in sig.items():
        if not isinstance(r, dict):
            continue
        p = r.get("p_value")
        rows.append([
            str(label).replace("_", " "),
            str(r.get("better_model", "-")),
            "-" if p is None else f"{float(p):.4g}",
            "significant" if r.get("significant") else "not significant",
        ])
    if rows:
        out.append(_table(["comparison", "better model", "p-value", "conclusion"], rows,
                          right_from=2))
    for label, r in sig.items():
        if isinstance(r, dict) and r.get("interpretation"):
            out.append(Spacer(1, 3))
            out.append(_para(f"<b>{esc(str(label).replace('_', ' '))}:</b> "
                             f"{esc(r['interpretation'])}", "small"))
    return out


def _sec_limits() -> list:
    out = _section("NOTES", "APPENDIX A", "Scope and limitations")
    for title, body in [
        ("Tuning is measured, not assumed",
         "The point of Stage 3 is to quantify the cost/benefit of tuning, not to prove that "
         "tuning always wins. On well-conditioned datasets a regularized baseline can match a "
         "tuned booster, and this report says so when it happens."),
        ("Drift is detected, not explained",
         "PSI and prequential error are cheap, reliable indicators that a distribution has "
         "moved. Neither attributes a cause; deeper causal drift analysis is out of scope."),
        ("Single-machine scale",
         "The pipeline targets datasets that fit in memory on one machine. It is not a "
         "distributed training system."),
        ("One split, one seed",
         "Results come from a single temporal split at a fixed random seed. That is the right "
         "choice for a time-ordered problem - shuffled k-fold would leak the future - but it "
         "means confidence intervals across resamples are not reported, which is why the "
         "significance tests above matter."),
    ]:
        out.append(_para(title, "h2"))
        out.append(_para(esc(body), "body"))
    return out


def _sec_glossary() -> list:
    out = _section("TERMS", "APPENDIX B", "Glossary")
    terms = [
        ("ROC-AUC", "Probability a random positive is ranked above a random negative. "
                    "0.5 is chance, 1.0 is perfect. Insensitive to class imbalance."),
        ("RMSE / MAE", "Average prediction error in the target's own units. Lower is better; "
                       "RMSE punishes large misses harder than MAE."),
        ("R-squared", "Share of the target's variance the model explains. 0 means no better "
                      "than predicting the mean."),
        ("Regularization (L1/L2)", "A penalty on coefficient size that trades a little bias "
                                   "for less variance. L1 can zero coefficients outright, "
                                   "performing feature selection; L2 shrinks them smoothly."),
        ("Stacking", "Training a small meta-model on the base models' predictions so it can "
                     "learn which to trust where."),
        ("Bayesian optimization", "Tuning that builds a surrogate model of the search space "
                                  "to choose each next trial, rather than sampling blindly."),
        ("SHAP value", "A feature's signed contribution to one prediction, derived from "
                       "cooperative game theory. Contributions sum to the prediction."),
        ("PSI", "Population Stability Index: how far a feature's distribution has moved from "
                "its training reference. Rules of thumb: below 0.1 stable, 0.1-0.25 "
                "moderate, above 0.25 material."),
        ("Prequential evaluation", "Test-then-train: each chunk is predicted before the model "
                                   "is allowed to learn from it, so no future information leaks."),
        ("McNemar's test", "A paired test for two classifiers on identical rows, based only on "
                           "the cases where they disagree."),
    ]
    rows = [[t, d] for t, d in terms]
    tbl = Table(rows, colWidths=[CONTENT_W * 0.26, CONTENT_W * 0.74], hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), F_HEAD),
        ("FONTSIZE", (0, 0), (0, -1), 8.4),
        ("TEXTCOLOR", (0, 0), (0, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -2), 0.25, RULE),
    ]))
    # wrap descriptions as paragraphs for wrapping
    wrapped = [[Paragraph(f"<b>{esc(t)}</b>", ST["small"]), Paragraph(esc(d), ST["small"])]
               for t, d in terms]
    tbl = Table(wrapped, colWidths=[CONTENT_W * 0.26, CONTENT_W * 0.74], hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("LINEBELOW", (0, 0), (-1, -2), 0.25, RULE),
    ]))
    out.append(tbl)
    out.append(Spacer(1, 14))
    out.append(_rule(RULE, 0.5, 2))
    out.append(Spacer(1, 6))
    out.append(_para("Produced by the TabML-Guard pipeline.", "sign"))
    return out


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def build(context: dict, output_dir: str, title: str = "TabML-Guard Report",
          cfg: Optional[Any] = None) -> str:
    """Assemble the PDF report from the run context. Returns its path.

    ``output_dir`` is the *run directory* for this dataset (``results/<slug>/``),
    which is also where ``figures/`` lives.
    """
    from . import paths

    task = context.get("task", "classification")
    primary = (context.get("baselines") or {}).get("primary_metric") \
        or ("roc_auc" if task == "classification" else "rmse")
    ins = _insights(context, task, primary)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    run_out = Path(output_dir)
    run_out.mkdir(parents=True, exist_ok=True)
    fig_dir = run_out / "figures"

    if cfg is not None:
        slug = paths.dataset_slug(cfg, context, title)
        out_path = run_out / f"report_{slug}.pdf"
        try:
            dataset_label = str(cfg.dataset.name or cfg.dataset.path or slug)
        except Exception:
            dataset_label = slug
    else:
        slug = paths.dataset_slug(None, context, title)
        out_path = run_out / f"report_{slug}.pdf"
        dataset_label = slug

    # ---- section builders --------------------------------------------------
    # Each entry is (toc name, builder). Builders are called afresh for every
    # build pass, because reportlab flowables cannot be reused across builds.
    section_defs = [
        ("Dataset, profile and configuration", lambda: _sec_dataset(context, cfg, task)),
        ("Regularized linear baselines", lambda: _sec_baselines(context, task, primary)),
        ("Ensemble and boosting models", lambda: _sec_boosting(context, task, primary)),
        ("Hyperparameter optimization: benefit vs cost",
         lambda: _sec_tuning(context, task, primary)),
        ("Explainability and prediction reliability",
         lambda: _sec_explain(context, fig_dir, task)),
        ("Concept drift and statistical significance",
         lambda: _sec_drift(context, fig_dir, task)),
        ("Appendix A - Scope and limitations", _sec_limits),
        ("Appendix B - Glossary", _sec_glossary),
    ]

    def _make_story(toc_entries: list) -> list:
        """Assemble a complete, single-use story with the given TOC page numbers."""
        story: list = []
        story += _cover(context, ins, title, dataset_label, task, primary, timestamp)
        story.append(_NextTemplate("body"))
        story.append(PageBreak())
        story += _toc(toc_entries)
        story.append(PageBreak())
        for i, (name, builder) in enumerate(section_defs):
            story.append(_Marker(name))
            story.extend(builder())
            if i < len(section_defs) - 1:
                story.append(Spacer(1, 10))
        return story

    # ---- measuring pass: learn which page each section starts on -----------
    # Built with placeholder TOC numbers of the same width, so the layout the
    # probe measures matches the final layout.
    placeholder = [(name, "-") for name, _ in section_defs]
    section_pages: dict = {}
    tmp = run_out / f".probe_{slug}.pdf"
    try:
        _Marker.SINK = section_pages
        probe = _Doc(str(tmp), dataset=dataset_label, timestamp=timestamp,
                     doc_title=title)
        probe.build(_make_story(placeholder))
    except Exception:
        section_pages = {}
    finally:
        _Marker.SINK = None
        try:
            tmp.unlink()
        except Exception:
            pass

    toc_entries = [(name, section_pages.get(name, "-")) for name, _ in section_defs]

    # ---- final two-pass build (fills in "Page N of M") ---------------------
    doc = _Doc(str(out_path), dataset=dataset_label, timestamp=timestamp, doc_title=title)
    doc.run_two_pass(lambda: _make_story(toc_entries))
    return str(out_path)


# --------------------------------------------------------------------------- #
# Small flowables used above (defined last to keep the reading order natural)
# --------------------------------------------------------------------------- #
from reportlab.platypus import Flowable, NextPageTemplate as _NextTemplate  # noqa: E402


class _Marker(Flowable):
    """Zero-height flowable that records which page it landed on.

    Used only during the measuring pass so the table of contents can show real
    page numbers. When ``SINK`` is None (the real build) it does nothing.
    """

    SINK: Optional[dict] = None

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.width = 0
        self.height = 0

    def wrap(self, availWidth, availHeight):
        return (0, 0)

    def draw(self):
        sink = _Marker.SINK
        if sink is not None:
            page = self.canv.getPageNumber()
            sink.setdefault(self.name, page)
