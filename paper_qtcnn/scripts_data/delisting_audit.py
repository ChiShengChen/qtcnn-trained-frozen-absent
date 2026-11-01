"""Match the universe securities missing from the yfinance extension against the
JPX official delisting lists (2022-2026) and classify the delisting reason."""
import re, html, glob
import pandas as pd
from pathlib import Path
W = Path(__file__).resolve().parent.parent
rows = []
for f in sorted(glob.glob(str(W / "cache/extension/jpx_delisted/*.html"))):
    t = open(f, encoding="utf-8").read()
    for r in re.findall(r"<tr[^>]*>(.*?)</tr>", t, flags=re.S):
        cells = [" ".join(html.unescape(re.sub(r"<[^>]+>", " ", c)).split()) for c in re.findall(r"<td[^>]*>(.*?)</td>", r, flags=re.S)]
        if len(cells) >= 5 and re.match(r"\d{4}/\d{2}/\d{2}", cells[0]):
            rows.append(dict(delist_date=cells[0], name=cells[1], code=cells[2], market=cells[3], reason=cells[4]))
d = pd.DataFrame(rows).drop_duplicates()
d["code_num"] = pd.to_numeric(d["code"].str.extract(r"(\d{4})")[0], errors="coerce")
def classify(s):
    if re.search(r"公開買付|ＭＢＯ|MBO|買収|完全子会社|合併|併合|株式交換|株式移転|株式等売渡|子会社化", s): return "acquisition/going-private"
    if re.search(r"破産|民事再生|会社更生|債務超過|上場維持基準|上場廃止基準|監理|特設注意|虚偽", s): return "failure/standards"
    if re.search(r"申請|Ｔｏｋｙｏ|TOKYO PRO|新設", s): return "voluntary/other"
    return "other"
d["class"] = d["reason"].map(classify)
miss = pd.read_csv(W / "cache/extension/universe_missing.csv")["SecuritiesCode"]
m = d[d["code_num"].isin(miss)].sort_values("delist_date")
m.to_csv(W / "cache/extension/universe_missing_delisted.csv", index=False)
print("delisting rows parsed:", len(d), "| universe-missing codes:", len(miss), "| matched:", m["code_num"].nunique())
print(m["class"].value_counts())
print(m["delist_date"].str[:4].value_counts().sort_index())
unmatched = sorted(set(miss) - set(m["code_num"]))
print("unmatched codes:", len(unmatched), unmatched[:30])
print(m["reason"].value_counts().head(12))
