"""测试港股分钟线数据源 - 绕过 akshare 直接请求东财 + 伪装 Headers"""
import requests, json, pandas as pd
import time

# ============================================================
# 东财港股分钟线（直接请求，绕过 akshare 的反爬问题）
# ============================================================
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Referer": "https://quote.eastmoney.com/hk/00700.html",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
}


def eastmoney_hk_minute(symbol: str, period: str = "5", count: int = 500):
    """直接请求东财港股分钟K线 API

    Args:
        symbol: 5位港股代码，如 "00700"
        period: "1"/"5"/"15"/"30"/"60"
        count: 返回K线根数，最大约 5000
    Returns:
        DataFrame [时间, 开盘, 收盘, 最高, 最低, 成交量, 成交额]
    """
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "klt": period,
        "fqt": "0",
        "secid": f"116.{symbol}",
        "beg": "0",
        "end": str(count),
    }
    r = requests.get(url, params=params, headers=HEADERS, timeout=15)
    r.raise_for_status()
    data_json = r.json()

    if data_json.get("data") is None or data_json["data"].get("klines") is None:
        raise RuntimeError(f"东财返回空数据: {data_json.get('msg', '未知错误')}")

    klines = data_json["data"]["klines"]
    rows = [item.split(",") for item in klines]
    df = pd.DataFrame(rows, columns=[
        "时间", "开盘", "收盘", "最高", "最低", "成交量",
        "成交额", "振幅", "涨跌幅", "涨跌额", "换手率",
    ])
    for col in ["开盘", "收盘", "最高", "最低", "成交量", "成交额"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["时间"] = pd.to_datetime(df["时间"])
    df = df.set_index("时间")
    return df


# ============================================================
# 测试
# ============================================================
print("=" * 60)
print("东财港股5分钟线 (直接请求 + 伪装Headers)")
print("=" * 60)

for code in ["00700", "09988", "01024"]:
    try:
        df = eastmoney_hk_minute(code, period="5", count=300)
        print(f"  {code}: {len(df)} 根5分钟线, {df.index[0]} ~ {df.index[-1]}")
        print(df.tail(3).to_string())
        print()
        break  # 只测一只，通了就不继续
    except Exception as e:
        print(f"  {code}: FAIL - {type(e).__name__}: {str(e)[:80]}")
        time.sleep(1)

print()
print("=" * 60)
print("如果上面失败，备选方案：腾讯港股日K线（已验证可用）")
print("=" * 60)
try:
    url = "http://web.ifzq.gtimg.cn/appstock/app/hkfqkline/get"
    params = {"param": "hk00700,day,2026-01-01,2026-07-15,640,qfq"}
    r = requests.get(url, params=params, headers=HEADERS, timeout=15)
    data = r.json()["data"]["hk00700"]["qfqday"]
    print(f"  腾讯日K线: {len(data)} 根, {data[0][0]} ~ {data[-1][0]}")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {str(e)[:80]}")