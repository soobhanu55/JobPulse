"""Checks for the German-level rules, the part of the analysis most likely to be silently wrong.
    python test_rules.py
"""
from analyze import german_level

CASES = {
    "Sehr gute Deutsch- und Englischkenntnisse in Wort und Schrift": "fluent (C1+)",
    "Fließende Deutschkenntnisse (C1) sind erforderlich": "fluent (C1+)",
    "You are fluent in German and English": "fluent (C1+)",
    "German (C1) and English required": "fluent (C1+)",
    "Gute Deutschkenntnisse": "good (B1-B2)",
    "Good command of German": "good (B1-B2)",
    "Deutschkenntnisse sind von Vorteil": "optional",
    "German is a plus, English is our company language": "optional",
    "Fluent English required": "not mentioned",
}

if __name__ == "__main__":
    bad = [(t, want, german_level(t)) for t, want in CASES.items() if german_level(t) != want]
    for t, want, got in bad:
        print(f"FAIL {t!r}: want {want}, got {got}")
    print(f"{len(CASES) - len(bad)}/{len(CASES)} rule checks pass")
    raise SystemExit(1 if bad else 0)
