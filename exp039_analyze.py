"""exp039 분석 — 조건별 사람 정확도 + **trial 단위** proxy↔사람 상관.

핵심은 두 번째다. 조건 평균으로 상관을 내면 점이 8개뿐이라 아무것도 못 말한다.
trial 단위로 (proxy_i, 정답여부_i) 를 쌓으면 320 trial × 참가자 수 만큼 점이 생기고,
logistic regression 으로 "proxy 가 사람 인식률을 예측하는가" 를 검정할 수 있다.

이게 통과하면 이후 ablation 은 proxy 로만 보고하고 여기를 인용한다
(FID/LPIPS 가 정당화되는 방식과 같다). "사람을 대체한다"고 주장하지 않는다.

참가자 간 상관을 무시하면 p 값이 과하게 작아지므로 **참가자 clustered bootstrap** 으로
신뢰구간을 낸다.

Run: python exp039_analyze.py --results_dir ./collected_results
"""
import argparse, csv, glob, os
import numpy as np

CHANCE = 0.25
PROXIES = ['heldout_cos', 'cosface_cos', 'clip']


def load(d):
    rows = []
    for f in sorted(glob.glob(f'{d}/*.csv')):
        for r in csv.DictReader(open(f)):
            try:
                r['correct'] = int(r['correct'])
                r['rt_ms'] = float(r['rt_ms'])
            except (ValueError, KeyError):
                continue
            for p in PROXIES:
                r[p] = float(r[p]) if r.get(p) not in (None, '', 'nan') else np.nan
            rows.append(r)
    return rows


def binom_p(k, n, p0):
    """정규근사 양측 p (scipy 없이)."""
    if n == 0:
        return float('nan')
    se = (p0 * (1 - p0) / n) ** 0.5
    z = (k / n - p0) / se if se > 0 else 0.0
    return float(2 * 0.5 * np.math.erfc(abs(z) / np.sqrt(2)))


def logistic(X, y, iters=2000, lr=0.1):
    """표준화된 X(절편 포함) 에 대한 간단 경사하강 로지스틱."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        w -= lr * (X.T @ (p - y)) / len(y)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default='./collected_results')
    ap.add_argument('--n_boot', type=int, default=2000)
    a = ap.parse_args()
    rows = load(a.results_dir)
    if not rows:
        print(f'{a.results_dir} 에 CSV 가 없다'); return
    parts = sorted({r['participant'] for r in rows})
    print(f'참가자 {len(parts)}명 · trial {len(rows)}개 (chance {CHANCE:.0%})\n')

    print('=== 조건별 사람 정확도 ===')
    print(f'{"condition":<12}{"acc%":>8}{"±SE":>7}{"n":>6}{"vs chance":>11}{"p":>10}'
          f'{"RT(s)":>8}{"proxy":>9}')
    conds = []
    for c in dict.fromkeys(r['condition'] for r in rows):
        rr = [r for r in rows if r['condition'] == c]
        k, n = sum(r['correct'] for r in rr), len(rr)
        acc = k / n
        se = (acc * (1 - acc) / n) ** 0.5
        pv = binom_p(k, n, CHANCE)
        rt = np.median([r['rt_ms'] for r in rr]) / 1000
        px = np.nanmean([r['heldout_cos'] for r in rr])
        tag = 'higher' if acc > CHANCE else 'LOWER'
        print(f'{c:<12}{100*acc:>8.1f}{100*se:>7.1f}{n:>6}{tag:>11}{pv:>10.2g}{rt:>8.1f}{px:>9.3f}')
        conds.append((c, acc, px))
    print()

    print('=== trial 단위 proxy -> 사람 정답 로지스틱 ===')
    print('  (참가자 clustered bootstrap 95% CI · 계수>0 이면 proxy 높을수록 사람이 잘 맞춤)')
    for pname in PROXIES:
        m = np.array([not np.isnan(r[pname]) for r in rows])
        if m.sum() < 30:
            print(f'  {pname:<13} 유효 trial 부족 ({int(m.sum())})'); continue
        x = np.array([r[pname] for r in rows])[m]
        y = np.array([r['correct'] for r in rows], float)[m]
        pid = np.array([r['participant'] for r in rows])[m]
        mu, sd = x.mean(), x.std() + 1e-9
        X = np.column_stack([np.ones(len(x)), (x - mu) / sd])
        w = logistic(X, y)
        bs = []
        for _ in range(a.n_boot):
            pick = np.random.choice(parts, len(parts), replace=True)
            idx = np.concatenate([np.where(pid == p)[0] for p in pick])
            if len(idx) < 30:
                continue
            bs.append(logistic(X[idx], y[idx])[1])
        lo, hi = np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan)
        sig = '유의' if (lo > 0 or hi < 0) else '유의하지 않음'
        # 상관계수도 같이 (점이진 상관)
        r_pb = np.corrcoef(x, y)[0, 1]
        print(f'  {pname:<13} beta={w[1]:+.3f}  95%CI [{lo:+.3f}, {hi:+.3f}]  {sig}'
              f'   r_pb={r_pb:+.3f}  n={len(x)}')
    print()

    print('=== 조건 평균 수준 상관 (참고용 — 점이 조건 수만큼뿐이라 약하다) ===')
    px = np.array([c[2] for c in conds]); ac = np.array([c[1] for c in conds])
    ok = ~np.isnan(px)
    if ok.sum() >= 3:
        print(f'  pearson r = {np.corrcoef(px[ok], ac[ok])[0,1]:+.3f}  (n={int(ok.sum())} 조건)')
    print('\n해석 지침: trial 단위 beta 의 CI 가 0 을 넘지 않아야 "proxy 가 사람과 정렬된다"고 '
          '쓸 수 있다. 넘으면 proxy 를 바꾸거나, proxy 로 ablation 을 대체하지 말아야 한다.')


if __name__ == '__main__':
    main()
