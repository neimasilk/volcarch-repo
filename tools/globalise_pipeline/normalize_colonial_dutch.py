"""
Colonial Dutch Spelling Normalizer
=====================================
Maps pre-1947 Dutch spelling to modern Dutch for NLP processing.

Three normalization layers:
1. Systematic orthographic rules (ij→y, oe→oo, tj→c, etc.)
2. Colonial place-name dictionary (Soerabaja → Surabaya)
3. VOC-era abbreviation expansion (M=r → Mijnheer)

Usage:
    from normalize_colonial_dutch import ColonialDutchNormalizer
    normalizer = ColonialDutchNormalizer()
    modern = normalizer.normalize("Soerabaja")  # "Surabaya"
    modern = normalizer.normalize("den 15=e Januarij 1786")  # "den 15e Januari 1786"
"""

import re
from typing import Dict, List, Tuple


class ColonialDutchNormalizer:
    """Normalize colonial/historical Dutch to modern Dutch."""

    def __init__(self):
        self.place_names = self._build_place_dict()
        self.abbreviations = self._build_abbreviation_dict()
        self.spelling_rules = self._build_spelling_rules()

    def _build_place_dict(self) -> Dict[str, str]:
        """Colonial → modern Indonesian/Dutch place name mapping."""
        return {
            # Java
            "Soerabaja": "Surabaya",
            "Soerabaya": "Surabaya",
            "Batavia": "Jakarta",
            "Buitenzorg": "Bogor",
            "Samarang": "Semarang",
            "Cheribon": "Cirebon",
            "Grissee": "Gresik",
            "Japara": "Jepara",
            "Pekalongan": "Pekalongan",  # unchanged
            "Banjoemas": "Banyumas",
            "Kediri": "Kediri",  # unchanged
            "Madoera": "Madura",
            "Pasoeroean": "Pasuruan",
            "Probolinggo": "Probolinggo",  # unchanged
            "Soerakarta": "Surakarta",
            "Djokjakarta": "Yogyakarta",
            "Djokja": "Yogya",
            "Malang": "Malang",  # unchanged
            "Toeban": "Tuban",
            "Tegal": "Tegal",  # unchanged
            "Bantam": "Banten",
            "Trowoelan": "Trowulan",
            "Modjokerto": "Mojokerto",
            "Singosari": "Singhasari",
            "Sidoardjo": "Sidoarjo",
            "Blitar": "Blitar",  # unchanged
            "Bondowoso": "Bondowoso",  # unchanged
            "Loemadjang": "Lumajang",
            # Sumatra
            "Padang": "Padang",  # unchanged
            "Palembang": "Palembang",  # unchanged
            "Djambi": "Jambi",
            "Atjeh": "Aceh",
            "Medan": "Medan",  # unchanged
            "Benkoelen": "Bengkulu",
            "Lampong": "Lampung",
            # Other islands
            "Celebes": "Sulawesi",
            "Makassar": "Makassar",  # unchanged (was also Macassar)
            "Macassar": "Makassar",
            "Borneo": "Kalimantan",
            "Amboina": "Ambon",
            "Ternaten": "Ternate",
            "Banda": "Banda",  # unchanged
            "Ceilon": "Ceylon",
            "Ceijlon": "Ceylon",
            "Cijlon": "Ceylon",
            "Malacca": "Malaka",
            # Historic
            "Nederlandsch Indie": "Nederlands-Indie",
            "Nederlandsch-Indie": "Nederlands-Indie",
        }

    def _build_abbreviation_dict(self) -> Dict[str, str]:
        """VOC-era abbreviation expansion."""
        return {
            r"M=r": "Mijnheer",
            r"UE:": "Uw Edelheid",
            r"UE\.": "Uw Edelheid",
            r"Gouv\.": "Gouverneur",
            r"Gen\.": "Generaal",
            r"Comp\.": "Compagnie",
            r"Bat\.": "Batavia",
            r"Res\.": "Residentie",
            r"Reg\.": "Regentschap",
            r"dd\.": "de dato",
            r"No\.": "Nummer",
            r"Besl\.": "Besluit",
        }

    def _build_spelling_rules(self) -> List[Tuple[str, str]]:
        """Systematic colonial Dutch → modern Dutch spelling rules.

        These are applied as regex substitutions in order.
        Conservative: only apply clear, systematic changes.
        """
        return [
            # Month names (most common and unambiguous)
            (r"\bJanuarij\b", "Januari"),
            (r"\bFebruarij\b", "Februari"),
            (r"\bMeij\b", "Mei"),
            (r"\bJunij\b", "Juni"),
            (r"\bJulij\b", "Juli"),
            (r"\bAugustij\b", "Augustus"),
            (r"\bDecemberij\b", "December"),

            # Date format normalization
            (r"(\d+)\s*=\s*([edstn]+)\b", r"\1\2"),  # 15=e → 15e

            # sch at word end → s (NOT applied — too aggressive, many exceptions)
            # Final -ij endings in specific words
            (r"\bpartij\b", "partij"),    # unchanged (correct modern Dutch)
            (r"\bbij\b", "bij"),          # unchanged

            # tj → c in Indonesian words
            (r"\btjandi\b", "candi"),
            (r"\bTjandi\b", "Candi"),
            (r"\bdessa\b", "desa"),
            (r"\bDessa\b", "Desa"),
            (r"\bkampong\b", "kampung"),
            (r"\bnegorij\b", "negeri"),
            (r"\bregentschap\b", "regentschap"),  # unchanged (Dutch word)

            # oe → u in Indonesian words (careful — don't change Dutch words)
            (r"\bSoenda\b", "Sunda"),
            (r"\bsoendaneesch\b", "Sundanees"),
            (r"\bKoedoes\b", "Kudus"),
            (r"\bOedjong\b", "Ujung"),
            (r"\bToeban\b", "Tuban"),
            (r"\bLoemadjang\b", "Lumajang"),
            (r"\bModjokerto\b", "Mojokerto"),
            (r"\bPasoeroean\b", "Pasuruan"),

            # dj → j in Indonesian words
            (r"\bDjokjakarta\b", "Yogyakarta"),
            (r"\bDjambi\b", "Jambi"),
            (r"\bDjedong\b", "Jedong"),
            (r"\bDjoeroedesain\b", "Jurudesain"),
        ]

    def normalize_places(self, text: str) -> str:
        """Apply place-name dictionary mapping."""
        for colonial, modern in self.place_names.items():
            text = re.sub(r'\b' + re.escape(colonial) + r'\b', modern, text)
        return text

    def normalize_abbreviations(self, text: str) -> str:
        """Expand VOC-era abbreviations."""
        for pattern, expansion in self.abbreviations.items():
            text = re.sub(pattern, expansion, text)
        return text

    def normalize_spelling(self, text: str) -> str:
        """Apply systematic spelling rules."""
        for pattern, replacement in self.spelling_rules:
            text = re.sub(pattern, replacement, text)
        return text

    def normalize(self, text: str, level="full") -> str:
        """Full normalization pipeline.

        Levels:
            "light": only place names + abbreviations (safe, minimal changes)
            "medium": + date formats + month names
            "full": + all spelling rules (may introduce errors on ambiguous words)
        """
        if level in ("light", "medium", "full"):
            text = self.normalize_places(text)
            text = self.normalize_abbreviations(text)

        if level in ("medium", "full"):
            # Date format
            text = re.sub(r'(\d+)\s*=\s*([edstn]+)\b', r'\1\2', text)
            # Month names
            for pattern, replacement in self.spelling_rules[:7]:
                text = re.sub(pattern, replacement, text)

        if level == "full":
            text = self.normalize_spelling(text)

        return text

    def get_normalization_report(self, text: str) -> dict:
        """Report what would change without actually changing."""
        changes = []
        for colonial, modern in self.place_names.items():
            if colonial in text and colonial != modern:
                changes.append({"type": "place", "from": colonial, "to": modern})
        for pattern, expansion in self.abbreviations.items():
            if re.search(pattern, text):
                changes.append({"type": "abbreviation", "from": pattern, "to": expansion})
        return {"n_changes": len(changes), "changes": changes}


def test_normalizer():
    """Run test suite on normalizer."""
    norm = ColonialDutchNormalizer()

    tests = [
        # (input, expected_output, level)
        ("Soerabaja", "Surabaya", "light"),
        ("Batavia", "Jakarta", "light"),
        ("Celebes", "Sulawesi", "light"),
        ("den 15=e Januarij 1786", "den 15e Januari 1786", "medium"),
        ("M=r Willem Arnold Alting", "Mijnheer Willem Arnold Alting", "light"),
        ("tjandi Singosari", "candi Singhasari", "full"),
        ("dessa Toeban", "desa Tuban", "full"),
        ("de kampong bij Loemadjang", "de kampung bij Lumajang", "full"),
        ("Djokjakarta", "Yogyakarta", "full"),
        ("Ceijlon", "Ceylon", "light"),
    ]

    print("Colonial Dutch Normalizer — Test Suite")
    print("=" * 60)

    passed = 0
    for input_text, expected, level in tests:
        result = norm.normalize(input_text, level=level)
        ok = result == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] '{input_text}' -> '{result}' (expected: '{expected}') [{level}]")

    print(f"\n{passed}/{len(tests)} tests passed")

    # Demo on real VOC text
    print("\n--- Demo on real VOC text ---")
    voc_sample = (
        "Aan zijn Hoog Edelheid Den Hoog Edelen Groot Achtbaaren Heer. "
        "M=r Willem Arnold Alting. wegens den staat der vrije vereenigde "
        "Nederlanden en de Generaale Nederlandsche Oost Indische Compagnie "
        "Gouverneur Generaal Batavia den 15=e Meij 1786."
    )
    print(f"\nOriginal:\n  {voc_sample}")
    print(f"\nLight:\n  {norm.normalize(voc_sample, 'light')}")
    print(f"\nMedium:\n  {norm.normalize(voc_sample, 'medium')}")
    print(f"\nFull:\n  {norm.normalize(voc_sample, 'full')}")

    return passed == len(tests)


if __name__ == "__main__":
    test_normalizer()
