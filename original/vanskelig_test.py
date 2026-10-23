"""Stresstest for pong_ai.py. Legg filen ved siden av pong_ai.py.
Kjør: python3 vanskelig_test.py  (eller legg modellfiler etter kommandoen).
Ingen læring. Modellfilene endres ikke. Bruker bare standardbiblioteket uten GUI.
"""
import argparse
import csv
import hashlib
import random
from pathlib import Path
from pong_ai import Agent, Arena, BALL_RADIUS, PADDLE_SPEED

# Faste regler for alle lag. Ikke endre mellom modellene i samme konkurranse.
LEVELS = (("Vanskelig", 5), ("Ekspert", 7), ("Ekstrem", 9), ("Stresstest", 12))
DEFAULT_SEED = 7319
DEFAULT_SHOTS = 250  # per nivå: totalt 1000 baller


def trajectory(x, y, vx, vy):
    """Samme ballbevegelse og veggsprett som Arena, fram til høyre racket."""
    plane = 730 - BALL_RADIUS
    steps = 0
    while x < plane:
        x += vx
        y += vy
        if y < BALL_RADIUS:
            y = 2 * BALL_RADIUS - y
            vy = abs(vy)
        elif y > 600 - BALL_RADIUS:
            y = 2 * (600 - BALL_RADIUS) - y
            vy = -abs(vy)
        steps += 1
    return steps, y


def make_shots(count, seed):
    rng = random.Random(seed)
    shots = []
    for level, speed in LEVELS:
        accepted = 0
        while accepted < count:
            x = rng.uniform(100, 460)
            y = rng.uniform(12, 588)
            paddle = rng.uniform(50, 550)
            vy = rng.choice((-7, -6, -4, -2, 2, 4, 6, 7))
            steps, target = trajectory(x, y, speed, vy)
            # En fasitkontroller som vet framtidig treffpunkt skal kunne treffe.
            lo = max(50, paddle - steps * PADDLE_SPEED)
            hi = min(550, paddle + steps * PADDLE_SPEED)
            closest = min(hi, max(lo, target))
            if abs(closest - target) > 50 + BALL_RADIUS:
                continue
            shots.append((level, x, y, speed, vy, paddle, steps, target))
            accepted += 1
    return shots


def evaluate(model, shots, seed, gui=False):
    agent = Agent.load(model)
    rows = []
    window = None
    if gui:
        from pong_ai import Window
        window = Window()
    try:
        for index, (level, x, y, vx, vy, paddle, steps, target) in enumerate(shots):
            arena = Arena(seed)
            arena.x, arena.ball_y = x, y
            arena.vx, arena.vy = vx, vy
            arena.y[1] = paddle
            # Egen fast tilfeldighetssekvens per ball, uavhengig av tidligere baller.
            agent.rng.seed(seed + index + 100000)
            unknown = 0
            for _ in range(steps):
                state = arena.state(1)
                unknown += state not in agent.q
                arena.step((0, agent.action(state, training=False)))
                if window and not window.draw(arena, f'{level} | ball {index+1}/{len(shots)} | laering AV'):
                    raise RuntimeError('Test avbrutt. Delresultatet skal ikke brukes.')
            rows.append({'level': level, 'shot': index + 1,
                         'hit': int(arena.hits[1] > 0),
                         'error_px': abs(target - arena.y[1]),
                         'unknown_steps': unknown, 'steps': steps})
    finally:
        if window:
            window.close()
    return rows


def summary(model, rows):
    hits = sum(r['hit'] for r in rows)
    error = sum(r['error_px'] for r in rows) / len(rows)
    print(f'\n{model}')
    for level, _ in LEVELS:
        part = [r for r in rows if r['level'] == level]
        n = sum(r['hit'] for r in part)
        print(f'  {level:12}: {n:3}/{len(part)} treff ({100*n/len(part):5.1f} %)')
    print(f'  TOTALT: {hits}/{len(rows)} treff ({100*hits/len(rows):.2f} %)')
    print(f'  Gjennomsnittlig avstand fra racketmidten ved passering: {error:.2f} px')
    total_steps = sum(r['steps'] for r in rows)
    print(f"  Ukjente tilstander: {100*sum(r['unknown_steps'] for r in rows)/total_steps:.1f} % av stegene")
    return hits, error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('models', nargs='*')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--shots', type=int, default=DEFAULT_SHOTS)
    parser.add_argument('--gui', action='store_true', help='Vis enkeltballene (hele testen tar lang tid).')
    parser.add_argument('--output', default='resultater/vanskelig_test')
    args = parser.parse_args()
    if args.shots < 1:
        parser.error('--shots må være positivt')
    models = args.models
    if not models:
        print('Vanskelig test. Enter på tom linje starter testen.')
        while True:
            value = input('Modellfil (f.eks. modeller/lag1.json): ').strip().strip('"').strip("'")
            if not value:
                break
            models.append(value)
    if not models:
        parser.error('Oppgi minst én modellfil.')
    shots = make_shots(args.shots, args.seed)
    print(f'\n{len(shots)} baller. Seed {args.seed}. Læring og utforsking AV.')
    print('Høyere hastigheter er en stresstest utenfor vanlig treningsmiljø.')
    rank = []
    details = []
    for model in models:
        before = hashlib.sha256(Path(model).read_bytes()).hexdigest()
        rows = evaluate(model, shots, args.seed, args.gui)
        if hashlib.sha256(Path(model).read_bytes()).hexdigest() != before:
            raise RuntimeError('Modellfilen er endret under testen.')
        hits, error = summary(model, rows)
        rank.append({'model': model, 'hits': hits, 'shots': len(shots),
                     'hit_percent': 100*hits/len(shots), 'mean_error_px': error,
                     'seed': args.seed})
        details.extend(dict(model=model, **r) for r in rows)
    rank.sort(key=lambda r: (-r['hits'], r['mean_error_px']))
    print('\nRANGERING: flest treff, deretter lavest gjennomsnittlig avstand.')
    for i, r in enumerate(rank, 1):
        print(f"{i}. {r['model']}: {r['hits']} treff, {r['mean_error_px']:.4f} px")
    print('Identiske resultater er delt plass, selv om filene listes etter hverandre.')
    for suffix, records in (('_rangering.csv', rank), ('_detaljer.csv', details)):
        path = Path(args.output + suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        print(f'Lagret {path}')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        raise SystemExit(f'Feil: {exc}')
