"""Verify Cat's Eye scroll-stopper detection, tight clip defaults, calibrated 4D viral scoring, and UI hooks."""
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ["PEAKCUT_DATA"] = tempfile.mkdtemp(prefix="pc-viral-")
os.environ["DATABASE_URL"] = ""
os.environ["PEAKCUT_MULTISHOT"] = "0"
sys.path.insert(0, ".")

import numpy as np  # noqa: E402
from studio import brain, creators, jobs  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
fails = []


def check(name, ok, extra=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  {extra}" if extra and not ok else ""))
    if not ok:
        fails.append(name)


def make_words(text, start=0.0, step=0.35):
    out, t = [], start
    for token in text.split():
        dur = max(0.18, step * 0.8)
        out.append({"w": token, "s": round(t, 2), "e": round(t + dur, 2)})
        t += step
    return out


# 1. Default clip length checks (35s instead of 60s)
check("jobs.DEFAULTS max_len is 35", jobs.DEFAULTS["max_len"] == 35, f"got {jobs.DEFAULTS['max_len']}")
check("creators.DEFAULTS max_len is 35", creators.DEFAULTS["max_len"] == 35, f"got {creators.DEFAULTS['max_len']}")

index_html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
check("UI o_maxlen stepper defaults to 35", 'stepper("o_maxlen", "Longest", c?.max_len ?? 35' in index_html)
check("UI renders Scroll-stopper pill", "Scroll-stopper" in index_html and "ss-pill" in index_html)
check("UI renders 4D score-grid and escaped scroll_stopper", ".score-grid" in index_html and "esc(c.scroll_stopper)" in index_html)

# 2. score_clip() contrast: punchy 22s scroll-stopper vs bloated 58s dead-air intro
punchy_text = ("WAIT did you just see what happened right there?! No way he actually clutched that "
               "with one HP left while the entire lobby was rushing him at once! Look at the replay, "
               "nobody in the server even knew where he jumped from and he wiped the whole squad! What a finish!")
punchy_words = make_words(punchy_text, start=0.08, step=0.45)  # ~45 words over ~20s -> ~2.25 wps, ends at ~20.3s
z_punchy = np.zeros(23, dtype=np.float32)
z_punchy[0] = 2.8
z_punchy[1] = 2.1
z_punchy[18] = 2.9
z_punchy[19] = 3.1
punchy_sc = brain.score_clip(0.0, 21.0, words=punchy_words, db_z=z_punchy)
check(
    "punchy 21s clip with 0-2s hook & climax scores >= 85, 1M+ Strangers reach, loop_seamless & share_trigger",
    (
        punchy_sc["score"] >= 85
        and punchy_sc["scores"]["hook"] >= 80
        and punchy_sc["scores"]["shareability"] >= 80
        and punchy_sc["scores"]["focus"] >= 85
        and punchy_sc["tier"] == "viral"
        and punchy_sc["audience_reach"] == "1M+ Strangers"
        and punchy_sc["loop_seamless"] is True
        and bool(punchy_sc["share_trigger"])
        and bool(punchy_sc["scroll_stopper"])
    ),
    str(punchy_sc),
)

# Multi-beat wandering clip ("delivers one clear thing" check)
multi_beat_words = make_words(
    "Look at this play right here! Anyway moving on to the next thing by the way on another note let us change the subject.",
    start=0.2, step=0.7,
)
multi_beat_sc = brain.score_clip(0.0, 22.0, words=multi_beat_words, db_z=np.zeros(23, dtype=np.float32), ai_clip={"score": 90, "focus_score": 90})
check(
    "multi-beat topic-wandering clip is capped on focus <= 55 ('one clear thing' rule)",
    multi_beat_sc["scores"]["focus"] <= 55,
    str(multi_beat_sc),
)

# Insider-only stream lore clip ("instantly understandable to strangers" 1M+ vs 100K rule)
insider_words = make_words(
    "Wait remember on yesterday's stream in the discord as I told chat earlier only channel members and regulars know part 2!",
    start=0.1, step=0.65,
)
insider_sc = brain.score_clip(0.0, 18.0, words=insider_words, db_z=z_punchy, ai_clip={"score": 92, "clarity_score": 92})
check(
    "insider-lore clip is capped on clarity <= 55 and blocked from 1M+ Strangers reach",
    insider_sc["scores"]["clarity"] <= 55 and insider_sc["audience_reach"] != "1M+ Strangers",
    str(insider_sc),
)

bloated_text = ("hey guys welcome back so anyway today we are going to talk a little bit about some "
                "settings maybe later um like yeah okay so let me explain slowly what might happen eventually "
                "when the match starts maybe tomorrow or later on in the afternoon")
bloated_words = make_words(bloated_text, start=2.2, step=1.05)  # starts after 2.2s dead air, slow ~1 wps
z_flat = np.zeros(60, dtype=np.float32)
bloated_sc = brain.score_clip(0.0, 58.0, words=bloated_words, db_z=z_flat)
bloated_ai = brain.score_clip(
    0.0, 58.0, words=bloated_words, db_z=z_flat,
    ai_clip={"score": 95, "hook_score": 95, "pacing_score": 92, "payoff_score": 90, "clarity_score": 95},
)
check(
    "bloated 58s dead-air clip scores sharply lower than punchy clip (>= 25 pt gap)",
    punchy_sc["score"] - bloated_sc["score"] >= 25,
    f"punchy={punchy_sc['score']}, bloated={bloated_sc['score']}",
)
check(
    "objective calibration caps inflated AI hook/overall score on dead-air bloated clip",
    bloated_ai["scores"]["hook"] <= 55 and bloated_ai["score"] < 65,
    str(bloated_ai),
)

# 3. fallback_by_loudness(): tight clips even when max_len=60, differentiated scores, real spoken hook
db_fb = np.full(140, -28.0, dtype=np.float32)
# Peak 1 around t=20s (very strong burst + urgent opening speech)
db_fb[19:22] = -6.0
db_fb[35:37] = -4.0
# Peak 2 around t=95s (moderate bump + slow filler)
db_fb[95:97] = -14.0

w_fb = (
    make_words("Wait look at that insane shot! Nobody survived that explosion at the end!", start=18.2, step=0.4)
    + make_words("uh yeah okay maybe so it was alright whatever happens next is fine", start=92.5, step=0.9)
)
picks = brain.fallback_by_loudness(db_fb, w_fb, 2, 60, 140.0)
check(
    "fallback_by_loudness picks 2 tight clips (<= 35s even with max_len=60)",
    len(picks) == 2 and all(14.0 <= (p["end"] - p["start"]) <= 35.0 for p in picks),
    str([(p["start"], p["end"], p["score"]) for p in picks]),
)
check(
    "fallback_by_loudness produces differentiated non-static scores and real spoken hooks",
    picks[0]["score"] > picks[1]["score"] and all(p["hook"] != "WAIT FOR IT" and bool(p["hook"]) for p in picks),
    str([(p["score"], p["hook"]) for p in picks]),
)
check(
    "fallback_by_loudness omits shots when PEAKCUT_MULTISHOT=0 and exposes 6-pillar scores/reach/scroll_stopper",
    all(
        p["shots"] == []
        and set(p["scores"]) == {"hook", "pacing", "payoff", "shareability", "focus", "clarity"}
        and p["tier"] in ("viral", "strong", "good", "weak")
        and p["audience_reach"] in ("1M+ Strangers", "100K+ Audience", "Core Followers", "Low Reach")
        and "scroll_stopper" in p
        for p in picks
    ),
    str(picks[0]),
)
compat_picks = brain.fallback_by_loudness(np.array([1.0] * 30), [{"w": "hello", "s": 2.0, "e": 3.0}], 1, 15, 30.0)
check("fallback_by_loudness 5-arg signature remains backward compatible", len(compat_picks) == 1)

# 4. _snap(): trims leading dead silence, keeps loud non-verbal reaction, caps far sentence look-ahead to <= 5s
w_snap = (
    make_words("This opens cleanly right here! First sentence finishes cleanly.", start=12.5, step=0.5)
    + make_words("Second sentence ends right near twenty seven.", start=20.0, step=0.9)
    + make_words("Third sentence runs much further down the timeline past the limit.", start=28.0, step=1.2)
)
lines_snap = brain.make_lines(w_snap)
z_silent = np.zeros(45, dtype=np.float32)
s_trim, e_trim = brain._snap({"start": 10.0, "end": 26.8}, lines_snap, w_snap, 12, 35, 45.0, z=z_silent)
check(
    "_snap trims dead opening air forward to first spoken word",
    12.2 <= s_trim <= 12.6,
    f"got s_trim={s_trim}, e_trim={e_trim}",
)

z_scream = np.zeros(45, dtype=np.float32)
z_scream[10] = 3.2
s_keep, _ = brain._snap({"start": 10.0, "end": 26.8}, lines_snap, w_snap, 12, 35, 45.0, z=z_scream)
check(
    "_snap preserves early start when loud non-verbal reaction opens the clip",
    s_keep <= 10.2,
    f"got s_keep={s_keep}",
)

# Look-ahead > 5s test: AI stopped at 21.0; next sentence ends at 34.0 (+13s), previous ended at 19.5
w_far = (
    make_words("Alpha beta gamma delta epsilon zeta.", start=0.0, step=0.5)  # ends ~2.9
    + make_words("Short mid sentence ending right near nineteen seconds.", start=15.0, step=0.55)  # ends ~19.3
    + make_words("Very long rambling sentence that does not finish until thirty four seconds from now yes.", start=19.8, step=1.0)
)
s_cap, e_cap = brain._snap({"start": 0.0, "end": 21.0}, brain.make_lines(w_far), w_far, 10, 40, 40.0)
check(
    "_snap refuses +13s look-ahead bloat and picks earlier sentence end",
    e_cap <= 21.0,
    f"got e_cap={e_cap}",
)

# 5. find_clips() mocked AI calibration, weak-filler drop, link stripping in scroll_stopper
orig_ask = brain.ask_ai
w_full = (
    make_words("WAIT did you see that play?! Unbelievable clutch right there!", start=0.1, step=0.42)
    + make_words("Second burst ends cleanly before thirty seconds.", start=6.0, step=1.5)
    + make_words("hey guys welcome back so anyway today we look slowly at nothing much at all.", start=42.5, step=1.1)
    + make_words("uh mumble", start=105.0, step=1.2)
)
db_full = np.full(120, -30.0, dtype=np.float32)
db_full[0:3] = -5.0
db_full[16:18] = -6.0


def mock_ask(prompt, schema=None, purpose="AI call", images=None):
    return [
        {
            "start": 0.0, "end": 18.0, "score": 88, "hook_score": 92, "pacing_score": 88,
            "payoff_score": 86, "clarity_score": 88,
            "scroll_stopper": "WAIT did you see that play?! https://spam.gg @handle",
            "hook": "INSANE CLUTCH", "title": "Unreal play", "description": "d",
            "hashtags": ["gaming"], "reason": "Instant 0-3s shock and climax",
        },
        {
            "start": 40.0, "end": 98.0, "score": 95, "hook_score": 95, "pacing_score": 90,
            "payoff_score": 90, "clarity_score": 95,
            "scroll_stopper": "hey guys welcome back",
            "hook": "LONG RAMBLE", "title": "Slow start", "description": "d",
            "hashtags": ["stream"], "reason": "Inflated score",
        },
        {
            "start": 100.0, "end": 116.0, "score": 25, "hook_score": 20, "pacing_score": 30,
            "payoff_score": 20, "clarity_score": 30,
            "scroll_stopper": "uh mumble",
            "hook": "WEAK FILLER", "title": "Weak", "description": "d",
            "hashtags": ["clip"], "reason": "Dead moment",
        },
    ], "mock-model", 42


brain.ask_ai = mock_ask
try:
    ai_clips, info = brain.find_clips(w_full, db_full, 120.0, max_clips=6, min_len=15, max_len=60)
finally:
    brain.ask_ai = orig_ask

check("find_clips drops weak <45 filler when strong clips exist", len(ai_clips) == 2 and all(c["hook"] != "WEAK FILLER" for c in ai_clips), str(ai_clips))
check(
    "find_clips ranks strong punchy clip cleanly above bloated dead-air clip",
    ai_clips[0]["hook"] == "INSANE CLUTCH" and ai_clips[0]["score"] > ai_clips[1]["score"] + 15,
    str([(c["hook"], c["score"], c["scores"]) for c in ai_clips]),
)
check(
    "find_clips strips links and @handles from scroll_stopper",
    "https://" not in ai_clips[0]["scroll_stopper"] and "@handle" not in ai_clips[0]["scroll_stopper"],
    ai_clips[0]["scroll_stopper"],
)

print("ALL PASSED" if not fails else f"{len(fails)} failed")
shutil.rmtree(os.environ["PEAKCUT_DATA"], ignore_errors=True)
sys.exit(1 if fails else 0)
