# Pong AI – start her

Jobb to og to. Begge bruker samme program. Test trente AI mot hverandre.

## Du trenger

- Python **3.12 eller 3.13** anbefales for denne pakken. Programkoden krever minst 3.10.
- Pygame 2.6.1 for spillvinduet. Ingen GPU, Pandas eller andre AI-biblioteker trengs.
- Internett ved installasjon. Deretter fungerer pakken lokalt.

Pakken er testet med Python 3.12 og Pygame 2.6.1 på Linux. Mac og Windows må prøves. Ved installasjonsproblemer på nyere Python, bruk 3.12 eller 3.13 fremfor å kompilere Pygame selv.

## Windows


```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe pong_ai.py
```

Har du Python 3.13, bytt `-3.12` med `-3.13`. Direkte bruk av miljøets python.exe gjør at du ikke trenger å endre PowerShells policy for aktivering.

## Mac med Homebrew

Åpne terminalen i mappen med pong_ai.py. Bruk Python 3.13 eksplisitt; vanlig python3 kan peke på 3.14. Kjør én linje om gangen og stopp hvis en linje gir feil.

```bash
brew install python@3.13
"$(brew --prefix python@3.13)/bin/python3.13" -m venv .venv313
source .venv313/bin/activate
python3 --version
python3 -m pip install --only-binary=:all: -r requirements.txt
python3 pong_ai.py
```

Versjonskontrollen skal vise 3.13.x. Etter aktivering vises normalt (.venv313) foran ledeteksten. Bare et aktivt miljø er ikke nok hvis det ble opprettet med feil Python-versjon.

Neste gang, fra samme mappe:

```bash
source .venv313/bin/activate
python3 pong_ai.py
```

Har du allerede Python 3.13 installert uten Homebrew, kan du bruke den installasjonens python3.13 til å opprette .venv313. 

## Linux

Åpne terminalen i den utpakkede mappen. Kontroller først `python3 --version`. Oppskriften nedenfor forutsetter Python 3.12 eller 3.13. 

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --only-binary=:all: -r requirements.txt
python pong_ai.py
```

Etter at terminalen er lukket, aktiver miljøet på nytt.

## Menyen

1. Spill mot utrent AI.
2. Tren med GUI.
3. Tren uten GUI.
4. Test en lagret modell.
5. Spill mot en lagret modell.
6. AI mot AI med sidebytte.

En menyhandling kjøres om gangen. Start programmet på nytt for neste handling. W/S eller piltastene styrer venstre racket. ESC eller lukking av vinduet avslutter. Ved avbrutt trening lagres modellen. En avbrutt konkurransekamp er ugyldig.

## Første treningsforsøk

1. Begge spiller mot utrent AI.
2. Se trening med GUI i 5–10 minutter. Menyens standard på 30 000 steg varer omtrent 8 minutter ved 60 steg per sekund.
3. Tren videre uten GUI ved å oppgi den lagrede modellen under «Fortsett fra modellfil».
4. Antall steg ved fortsettelse er **ekstra steg**. 30 000 med GUI pluss 270 000 uten GUI gir 300 000 totalt.
5. Test modellen og spill mot den.

Ved nye sammenligningsforsøk lar du feltet for fortsettelse være tomt. Bruk samme totale steggrense. Gi hver modell et eget filnavn, ellers overskrives filen. Fortsettelse bruker parameterverdiene i den lagrede modellen; nye endringer øverst i scriptet gjelder bare nye modeller.

## Endre bare disse verdiene

Øverst i pong_ai.py finner du:

```python
ALPHA = 0.10
GAMMA = 0.95
EPSILON = 0.10
```

Prøv alpha 0.05/0.10/0.20, gamma 0.90/0.95/0.99 eller epsilon 0.05/0.10/0.30. Endre én verdi om gangen. Bruk punktum. Spillfysikk, belønninger og tilstandsbeskrivelse holdes felles. 

## Kommandoer når miljøet er aktivert

På Windows kan `python` nedenfor erstattes med `.\.venv\Scripts\python.exe`.

```bash
python pong_ai.py play
python pong_ai.py train --gui --steps 30000 --output modeller/lag1.json
python pong_ai.py train --resume modeller/lag1.json --steps 270000 --output modeller/lag1.json
python pong_ai.py train --steps 300000 --seed 1 --output modeller/forsok2.json
python pong_ai.py test --model modeller/lag1.json --csv resultater/lag1.csv
python pong_ai.py play --model modeller/lag1.json
python pong_ai.py match --a modeller/lag1.json --b modeller/forsok2.json --gui
```

Testen kjører fem startverdier, 100–104, med 12 000 steg hver. Vunnede poeng betyr scoringer, ikke antall vunnede kamper. Læring og utforsking er automatisk avslått. Tilfeldig valg mellom like Q-verdier har fast startverdi under testen. Testen endrer ikke modellfilen.

AI mot AI kjører to delkamper à 90 simulerte sekunder. Bruk `--seconds 180` for finalen. Uten `--gui` kjøres samme antall steg raskere.

## Lever

- Den valgte JSON-modellen.
- En kort begrunnelse for modellvalget.




