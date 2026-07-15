"""使用 curl_cffi 模拟浏览器 TLS 指纹请求东财港股分钟线"""
import json, sys

try:
    from curl_cffi import requests as curl_requests
except ImportError:
    print("✗ 需要安装 curl_cffi: pip install curl_cffi")
    sys.exit(1)


def fetch_eastmoney_hk_minute(symbol: str, period: str = "5"):
    """用 curl_cffi 模拟 Chrome 浏览器 TLS 指纹请求东财 API"""
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "klt": period,
        "fqt": "0",
        "secid": f"116.{symbol}",
        "beg": "0",
        "end": "20500000",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "Referer": "https://quote.eastmoney.com/hk/00700.html",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
    }

    # 使用 Chrome 浏览器指纹
    r = curl_requests.get(
        url,
        params=params,
        headers=headers,
        impersonate="chrome146",
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    print(f"  [debug] dktotal={data.get('data',{}).get('dktotal','?')}, "
          f"klines={len(data.get('data',{}).get('klines',[]))} 根", flush=True)

    klines = data.get("data", {}).get("klines", [])
    if not klines:
        print(f"  [debug] 完整响应: {json.dumps(data, ensure_ascii=False)[:600]}", flush=True)
        raise RuntimeError("klines 为空")

    return klines


def main():
    for code in ["00700", "09988", "01024"]:
        print(f"\n正在请求 {code} 的 5 分钟线...")
        try:
            klines = fetch_eastmoney_hk_minute(code, period="5")
            print(f"  ✓ {code}: {len(klines)} 根K线")
            print(f"  首条: {klines[0]}")
            print(f"  末条: {klines[-1]}")
            break
        except Exception as e:
            print(f"  ✗ {code}: {type(e).__name__}: {str(e)[:120]}")


if __name__ == "__main__":
    main()