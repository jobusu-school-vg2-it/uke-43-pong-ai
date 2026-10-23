"""Pong med tabellbasert Q-learning. Python 3.10 eller nyere.

Elever endrer bare ALPHA, GAMMA og EPSILON i grunnoppgaven.
GUI trenger pygame. Trening, test og kamp uten GUI bruker standardbiblioteket.
"""
import argparse
import csv
import json
import math
import random
from pathlib import Path

# ELEVENE KAN ENDRE DISSE TRE VERDIENE
ALPHA = 0.2
GAMMA = 0.96
EPSILON = 0.15

# Default
# ALPHA = 0.10
# GAMMA = 0.95
# EPSILON = 0.10

# Felles spillregler og tilstandsbeskrivelse skal ikke endres.
WIDTH, HEIGHT = 800, 600
PADDLE_HEIGHT, PADDLE_SPEED = 100, 6
BALL_RADIUS, BALL_SPEED = 8, 5
ACTIONS = (-1, 0, 1)
FORMAT = 'pong-q-v1'
STATE_SPEC = 'own-perspective-x80-relative-y20-own-y60-vxsign-vy3-v1'


class Arena:
    def __init__(self, seed=1, mirrored=False):
        self.rng = random.Random(seed)
        self.mirrored = mirrored
        self.ticks = 0
        self.y = [HEIGHT / 2, HEIGHT / 2]
        self.score = [0, 0]
        self.hits = [0, 0]
        self.misses = [0, 0]
        self.serve()

    def serve(self):
        self.x, self.ball_y = WIDTH / 2, self.rng.uniform(180, 420)
        self.vx = self.rng.choice((-BALL_SPEED, BALL_SPEED))
        if self.mirrored: self.vx *= -1
        self.vy = self.rng.choice((-4, -3, -2, 2, 3, 4))

    def state(self, side):
        # Venstre spiller speiles slik at begge modeller bruker samme perspektiv.
        x = self.x if side == 1 else WIDTH - self.x
        vx = self.vx if side == 1 else -self.vx
        return (int(x // 80), int((self.ball_y - self.y[side]) // 20),
                int(self.y[side] // 60), 1 if vx > 0 else -1,
                max(-3, min(3, round(self.vy / 3))))

    def step(self, actions):
        self.ticks += 1
        reward = [0.0, 0.0]
        for side, action in enumerate(actions):
            old = abs(self.ball_y - self.y[side])
            self.y[side] = max(50, min(550, self.y[side] + action * PADDLE_SPEED))
            approaching = self.vx < 0 if side == 0 else self.vx > 0
            if approaching:
                # Felles hjelpebelønning gjør tidlig læring synlig.
                reward[side] += 0.02 * (old - abs(self.ball_y - self.y[side]))
        previous_x = self.x
        self.x += self.vx
        self.ball_y += self.vy
        if self.ball_y < BALL_RADIUS:
            self.ball_y = 2 * BALL_RADIUS - self.ball_y
            self.vy = abs(self.vy)
        elif self.ball_y > HEIGHT - BALL_RADIUS:
            self.ball_y = 2 * (HEIGHT - BALL_RADIUS) - self.ball_y
            self.vy = -abs(self.vy)
        for side, plane in ((0, 70 + BALL_RADIUS), (1, 730 - BALL_RADIUS)):
            crossing = (self.vx < 0 and previous_x > plane >= self.x) if side == 0 else (self.vx > 0 and previous_x < plane <= self.x)
            if crossing and abs(self.ball_y - self.y[side]) <= 50 + BALL_RADIUS:
                self.x = plane
                self.vx = abs(self.vx) if side == 0 else -abs(self.vx)
                self.vy = max(-7, min(7, (self.ball_y - self.y[side]) * 0.12))
                self.hits[side] += 1
                reward[side] += 2
        terminal = self.x < -BALL_RADIUS or self.x > WIDTH + BALL_RADIUS
        if terminal:
            loser = 0 if self.x < 0 else 1
            self.misses[loser] += 1
            self.score[1 - loser] += 1
            reward[loser] -= 5
            reward[1 - loser] += 5
            self.serve()
        return reward, terminal


class Agent:
    def __init__(self, seed=1, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON):
        self.q = {}
        self.rng = random.Random(seed)
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.steps, self.seed = 0, seed

    def action(self, state, training=False):
        if training and self.rng.random() < self.epsilon:
            return self.rng.choice(ACTIONS)
        values = self.q.get(state, [0.0, 0.0, 0.0])
        # Like verdier håndteres likt for alle. RNG har fast seed under test.
        best = max(values)
        return self.rng.choice([a for a, v in zip(ACTIONS, values) if v == best])

    def learn(self, state, action, reward, next_state, terminal):
        values = self.q.setdefault(state, [0.0, 0.0, 0.0])
        future = 0.0 if terminal else max(self.q.get(next_state, [0.0, 0.0, 0.0]))
        index = action + 1
        values[index] += self.alpha * (reward + self.gamma * future - values[index])
        self.steps += 1

    def save(self, filename):
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {'format': FORMAT, 'state_spec': STATE_SPEC, 'steps': self.steps,
                'seed': self.seed, 'alpha': self.alpha, 'gamma': self.gamma,
                'epsilon': self.epsilon,
                'q': [[list(s), values] for s, values in sorted(self.q.items())]}
        path.write_text(json.dumps(data, separators=(',', ':'), allow_nan=False), encoding='utf-8')

    @classmethod
    def load(cls, filename):
        data = json.loads(Path(filename).read_text(encoding='utf-8'))
        if data.get('format') != FORMAT or data.get('state_spec') != STATE_SPEC:
            raise ValueError('Modellen har feil format eller tilstandsbeskrivelse.')
        obj = cls(data['seed'], data['alpha'], data['gamma'], data['epsilon'])
        for state, values in data['q']:
            if len(state) != 5 or len(values) != 3 or not all(isinstance(x, int) for x in state) or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in values):
                raise ValueError('Ugyldig oppføring i modellen.')
            obj.q[tuple(state)] = values
        obj.steps = data['steps']
        return obj


def opponent(arena, side=0):
    # Felles trenings- og testmotstander. Reagerer bare når ballen kommer mot den.
    if arena.ticks % 2: return 0
    approaching = arena.vx < 0 if side == 0 else arena.vx > 0
    if not approaching:
        return 0
    delta = arena.ball_y - arena.y[side]
    return 1 if delta > 12 else -1 if delta < -12 else 0


class Window:
    def __init__(self):
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError('GUI trenger pygame. Kjør python -m pip install -r requirements.txt') from exc
        self.pg = pygame
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Pong AI')
        self.font = pygame.font.Font(None, 25)
        self.clock = pygame.time.Clock()

    def draw(self, arena, label):
        pg = self.pg
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                return False
        self.screen.fill((15, 20, 30))
        for x, y in zip((50, 730), arena.y):
            pg.draw.rect(self.screen, (230, 235, 245), (x, int(y - 50), 20, 100))
        pg.draw.circle(self.screen, (80, 210, 180), (int(arena.x), int(arena.ball_y)), BALL_RADIUS)
        for i, line in enumerate([f'Venstre {arena.score[0]} : {arena.score[1]} Hoyre', label, 'ESC avslutter. Spiller: W/S eller pil opp/ned.']):
            self.screen.blit(self.font.render(line, True, (240, 240, 240)), (15, 10 + i * 26))
        pg.display.flip()
        self.clock.tick(60)
        return True

    def human(self):
        pg = self.pg
        keys = pg.key.get_pressed()
        return int(keys[pg.K_s] or keys[pg.K_DOWN]) - int(keys[pg.K_w] or keys[pg.K_UP])

    def close(self):
        self.pg.quit()


def train(args):
    agent = Agent.load(args.resume) if args.resume else Agent(args.seed)
    arena = Arena(args.seed)
    window = Window() if args.gui else None
    try:
        for i in range(args.steps):
            state = arena.state(1)
            action = agent.action(state, training=True)
            reward, terminal = arena.step((opponent(arena), action))
            agent.learn(state, action, reward[1], arena.state(1), terminal)
            if window and not window.draw(arena, f'Trening steg {agent.steps} | epsilon {agent.epsilon} | tilstander {len(agent.q)}'):
                break
            if not window and (i + 1) % 100000 == 0:
                print(f'{i + 1}/{args.steps} steg', flush=True)
    finally:
        if window: window.close()
        agent.save(args.output)
    print(f'Lagret {args.output}: {agent.steps} steg, {len(agent.q)} tilstander, {Path(args.output).stat().st_size / 1048576:.2f} MB')


def test(args):
    agent = Agent.load(args.model)
    rows = []
    for seed in range(args.seed, args.seed + args.rounds):
        arena = Arena(seed)
        agent.rng.seed(seed + 10000)
        for _ in range(args.steps):
            arena.step((opponent(arena), agent.action(arena.state(1))))
        rows.append({'seed': seed, 'steps': args.steps, 'won': arena.score[1], 'lost': arena.score[0], 'hits': arena.hits[1], 'misses': arena.misses[1]})
    won, lost = sum(r['won'] for r in rows), sum(r['lost'] for r in rows)
    hits, misses = sum(r['hits'] for r in rows), sum(r['misses'] for r in rows)
    print(f'Vunne poeng: {won} av {won+lost} ({100*won/max(1,won+lost):.1f}%). Treffandel: {100*hits/max(1,hits+misses):.1f}%.')
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def play(args):
    agent = Agent.load(args.model) if args.model else Agent(args.seed)
    arena, window = Arena(args.seed), Window()
    try:
        while window.draw(arena, 'Menneske mot AI | laering av'):
            arena.step((window.human(), agent.action(arena.state(1))))
    finally:
        window.close()


def match(args):
    agents = [Agent.load(args.a), Agent.load(args.b)]
    totals = [0, 0]
    for leg in range(2):
        sides = agents if leg == 0 else list(reversed(agents))
        # Samme starter og seed i begge delkamper; modellene bytter side.
        for agent in agents: agent.rng.seed(args.seed + 20000)
        arena = Arena(args.seed)
        window = Window() if args.gui else None
        complete = True
        try:
            for i in range(args.seconds * 60):
                arena.step(tuple(agent.action(arena.state(side)) for side, agent in enumerate(sides)))
                if window and not window.draw(arena, f'Delkamp {leg+1}/2 | {args.seconds-i//60} sek igjen | laering av'):
                    complete = False; break
        finally:
            if window: window.close()
        if not complete:
            print('Kamp avbrutt. Resultatet skal ikke brukes.'); return None
        scores = arena.score if leg == 0 else list(reversed(arena.score))
        totals = [a + b for a, b in zip(totals, scores)]
        print(f'Delkamp {leg+1}, modell A:B = {scores[0]}:{scores[1]}')
    print(f'Samlet A:B = {totals[0]}:{totals[1]}')
    return totals


def menu():
    print('\n1 Spill mot utrent AI\n2 Tren med GUI\n3 Tren uten GUI\n4 Test modell\n5 Spill mot modell\n6 AI mot AI\n0 Avslutt')
    choice = input('Velg: ').strip()
    if choice == '0': return None
    if choice == '1': return ['play']
    if choice in ('2', '3'):
        steps = input('Treningssteg [30000 med GUI / 300000 uten]: ').strip() or ('30000' if choice == '2' else '300000')
        output = input('Modellfil [modeller/lag1.json]: ').strip() or 'modeller/lag1.json'
        resume = input('Fortsett fra modellfil (Enter = ny tom modell): ').strip()
        seed = input('Tilfeldig startverdi [1]: ').strip() or '1'
        return ['train', '--steps', steps, '--output', output, '--seed', seed] + (['--gui'] if choice == '2' else []) + (['--resume', resume] if resume else [])
    if choice in ('4', '5'):
        return ['test' if choice == '4' else 'play', '--model', input('Modellfil: ').strip()]
    if choice == '6':
        return ['match', '--a', input('Modell A: ').strip(), '--b', input('Modell B: ').strip(), '--gui']
    raise ValueError('Velg et tall fra menyen.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='mode')
    t = subs.add_parser('train'); t.add_argument('--steps', type=int, default=300000); t.add_argument('--seed', type=int, default=1)
    t.add_argument('--output', default='modeller/lag1.json'); t.add_argument('--resume'); t.add_argument('--gui', action='store_true')
    t.set_defaults(func=train)
    t = subs.add_parser('test'); t.add_argument('--model', required=True); t.add_argument('--steps', type=int, default=12000)
    t.add_argument('--seed', type=int, default=100); t.add_argument('--rounds', type=int, default=5); t.add_argument('--csv'); t.set_defaults(func=test)
    t = subs.add_parser('play'); t.add_argument('--model'); t.add_argument('--seed', type=int, default=1); t.set_defaults(func=play)
    t = subs.add_parser('match'); t.add_argument('--a', required=True); t.add_argument('--b', required=True)
    t.add_argument('--seconds', type=int, default=90); t.add_argument('--seed', type=int, default=500); t.add_argument('--gui', action='store_true'); t.set_defaults(func=match)
    args = parser.parse_args(argv)
    if args.mode is None:
        command = menu()
        if command is not None: main(command)
        return
    for name in ('steps', 'seconds', 'rounds'):
        if hasattr(args, name) and getattr(args, name) <= 0: parser.error(f'{name} må være positiv.')
    if not all(0 <= x <= 1 for x in (ALPHA, GAMMA, EPSILON)): parser.error('Læringsparametere må være mellom 0 og 1.')
    args.func(args)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        print(f'Feil: {error}')
        raise SystemExit(1)
