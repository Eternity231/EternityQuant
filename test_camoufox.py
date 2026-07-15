"""测试东财港股分时数据（用户抓包的正确格式）"""
from curl_cffi import requests as curl_requests
import json, pandas as pd

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/146.0.0.0 Safari/537.36",
    "Referer": "https://quote.eastmoney.com/hk/",
}
ut = "fa5fd1943c7b386f172d6893dbfba10b"

print("=== 测试用户抓包格式 ===")

# 1. ulist/get 确认股票存在
r = curl_requests.get(
    "https://push2.eastmoney.com/api/qt/ulist/get",
    params={
        "fltt": "1", "invt": "2",
        "fields": "f14,f12,f13,f1,f2,f4,f3,f152",
        "secids": "116.00700,116.09988,116.01810",
        "ut": ut,
        "pn": "1", "np": "1", "pz": "20",
        "dect": "1", "wbp2u": "|0|0|0|web",
    },
    headers=headers, impersonate="chrome146", timeout=15,
)
data = r.json()
diff = data.get("data", {}).get("diff", [])
names = [f"{d['f12']} {d['f14']}" for d in diff]
print(f"  ulist/get 3只: ✓ {', '.join(names)}")

# 2. trends2 分时数据
for code in ["00700", "09988", "01810"]:
    try:
        r = curl_requests.get(
            "https://push2.eastmoney.com/api/qt/stock/trends2/get",
            params={
                "fields1": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13",
                "fields2": "f51,f52,f53,f54,f55,f56,f57,f58",
                "ut": ut,
                "iscr": "0",
                "ndays": "5",
                "secid": f"116.{code}",
            },
            headers=headers, impersonate="chrome146", timeout=15,
        )
        data = r.json()
        trends = data.get("data", {}).get("trends")
        if trends:
            print(f"  trends2 {code}: ✓ {len(trends)} 条分时")
            print(f"    首条: {trends[0]}")
            print(f"    末条: {trends[-1]}")
        else:
            print(f"  trends2 {code}: rc={data.get('rc')}, data=null, full={json.dumps(data, ensure_ascii=False)[:200]}")
    except Exception as e:
        print(f"  trends2 {code}: FAIL {type(e).__name__}: {str(e)[:60]}")

# 3. 如果 trends2 通，试重采样为 5 分钟 K 线
print("\n=== 如果 trends2 通了，重采样为 5 分钟 K 线 ===")