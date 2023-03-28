from typing import FrozenSet, List, Union
from pythainlp.corpus import get_corpus
from pythainlp.spell import NorvigSpellChecker
from pythainlp.util import normalize

_THAI_THAILAND_PROVINCES = set()
_THAI_THAILAND_PROVINCES_DETAILS = list()
_THAI_THAILAND_PROVINCES_FILENAME = "thailand_provinces_th.csv"

def provinces(details: bool = False) -> Union[FrozenSet[str], List[str]] :

    global _THAI_THAILAND_PROVINCES, _THAI_THAILAND_PROVINCES_DETAILS

    if not _THAI_THAILAND_PROVINCES or not _THAI_THAILAND_PROVINCES_DETAILS:
        provs = set()
        prov_details = list()

        for line in get_corpus(_THAI_THAILAND_PROVINCES_FILENAME, as_is=True):
            p = line.split(",")

            prov = dict()
            prov["name_th"] = p[0]
            prov["abbr_th"] = p[1]
            prov["name_en"] = p[2]
            prov["abbr_en"] = p[3]

            provs.add(prov["name_th"])
            prov_details.append(prov)

        _THAI_THAILAND_PROVINCES = frozenset(provs)
        _THAI_THAILAND_PROVINCES_DETAILS = prov_details

    if details:
        return _THAI_THAILAND_PROVINCES_DETAILS

    return _THAI_THAILAND_PROVINCES

def province(province):
    checker = NorvigSpellChecker(custom_dict=provinces())

    province_name = normalize(province)
    province_name = checker.correct(str(province_name))
    """ print("province_pynlp : ", province_name) """

    return province_name