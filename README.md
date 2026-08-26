# hpmpc — självhostad ML-styrning av luft/vatten-värmepump

Modellprediktiv styrning (MPC) för en luft/vatten-värmepump på golvvärme. Systemet lär
sig ditt hus från Home Assistants historik och styr pumpen genom att manipulera vilken
utetemperatur den *tror* att den ser — via ditt digitala motstånd.

Allt körs lokalt: numpy, scipy och scikit-learn på en Raspberry Pi eller NUC. Ingen
molntjänst, ingen API-nyckel, inget som lämnar huset. Enda utgående trafiken är till din
egen Home Assistant.

**Tar hänsyn till:** utetemperatur, innetemperatur, vindstyrka, sol/molnighet och elpris —
plus värmepumpens COP, husets tröghet och betongplattans värmelagring.

---

## Innehåll

- [Idén](#idén)
- [Arkitektur](#arkitektur)
- [Snabbstart utan Home Assistant](#snabbstart-utan-home-assistant)
- [Komma igång på riktigt](#komma-igång-på-riktigt)
- [Excitation — det viktigaste steget](#excitation--det-viktigaste-steget)
- [Träning och vad siffrorna betyder](#träning-och-vad-siffrorna-betyder)
- [Backtest](#backtest)
- [Drift](#drift)
- [Säkerhet](#säkerhet)
- [Hårdvara: det digitala motståndet](#hårdvara-det-digitala-motståndet)
- [Vad kan man realistiskt spara?](#vad-kan-man-realistiskt-spara)
- [Begränsningar](#begränsningar)
- [Utveckling](#utveckling)

---

## Idén

En värmepump med väderkompenserad framledning har ingen aning om elpriset, och den vet
inte att solen ska skina om tre timmar. Den läser bara sin utegivare och slår upp
framledningstemperaturen i sin värmekurva.

Om du kan ljuga för utegivaren kan du styra hela pumpen utan att röra dess reglering:

```
offset −3 K  →  pumpen tror att det är kallare  →  högre framledning  →  mer värme
offset +3 K  →  pumpen tror att det är varmare  →  lägre framledning  →  mindre värme
```

Golvvärme i betongplatta är ett värmelager med tidskonstant på 8–30 timmar. Det gör att
man kan **ladda plattan när elen är billig** och **glida på lagrad värme när den är dyr**,
utan att inomhustemperaturen märkbart rör sig. Det är hela affärsidén.

Två detaljer som är lätta att missa och som avgör om det fungerar:

1. **Pumpen medelvärdesbildar sin utegivare** över några timmar. Ett offsetsteg får alltså
   inte omedelbart genomslag på framledningen. Modellen har det filtret som eget tillstånd.
2. **COP styrs av den verkliga uteluften, inte den påhittade.** Framledningen följer den
   falska temperaturen, men förångaren ser den riktiga. Att ladda på natten är billigt i
   kronor men dyrare i kWh, eftersom det är kallast då. Optimeraren väger det mot varandra.

---

## Arkitektur

```
Home Assistant  ──historik──►  dataset  ──►  systemidentifiering  ──►  modell
      │                                                                  │
      │◄─────────── offset / ohm ──────────  MPC-optimerare  ◄───────────┤
      │                                            ▲                     │
      └────────── prognos: väder + elpris ─────────┘                     │
                                                                     observatör
```

### 1. Grey-box-modell av huset (`model/thermal.py`)

En 2R2C-modell med tre tillstånd:

| Tillstånd | Betydelse | Typisk tidskonstant |
|---|---|---|
| `Ti` | inomhusluft och lätt inredning | 1–3 h |
| `Tm` | betongplatta och tung stomme | 8–30 h |
| `Tf` | pumpens egen filtrerade utetemperatur | 1–6 h |

```
Ci·dTi/dt = Him·(Tm−Ti) + Hie_eff·(Te−Ti) + f_sol·Qs + Qint
Cm·dTm/dt = Him·(Ti−Tm) + Hme·(Te−Tm) + Qgolv + (1−f_sol)·Qs
Qgolv     = clip(Hfloor·(Tw−Tm), 0, Qmax)
Hie_eff   = Hie·(1 + k_wind·v)          ← vinden sitter här
```

Tio parametrar, alla fysikaliskt tolkbara. Det är ett medvetet val framför ett rent
neuralt nät: med 3–6 veckors data är ett neuralt nät hopplöst underbestämt och
extrapolerar farligt vid temperaturer det aldrig sett, medan en fysikalisk modell
uppför sig rimligt även utanför träningsdatan. Den *lär sig* fortfarande — men den lär
sig tio tal med betydelse i stället för tiotusen vikter.

### 2. Lärd residual (`residual.py`)

Ovanpå fysiken sitter en gradient-boosting-modell (scikit-learn) som fångar det
2R2C-modellen omöjligt kan veta: morgonduschen, braskaminen, att lågt vintersol
träffar söderfönstren hårdare än en platt globalstrålningssiffra antyder.

Den är medvetet begränsad till **exogena särdrag** — klocka, sol, vind, utetemperatur.
Aldrig husets eget tillstånd och aldrig styrsignalen. Det ger två saker: optimeraren kan
inte utnyttja den i en återkopplingsslinga, och korrektionen kan beräknas en gång per
lösning i stället för en gång per simuleringssteg. Bidraget är hårt begränsat
(`residual_max_correction`, default 0,4 K/h), och modellen kastas automatiskt om den
inte slår fysiken på valideringsdatan.

### 3. Optimeraren (`mpc.py`)

Beslutsvariabel: styckvis konstant offset över 36 timmar i 3-timmarsblock — 12 tal.

Målfunktion:

```
J = Σ pris·P·Δt                                  elkostnad
  + w_komfort · Σ avvikelse²                     mjuk komfortbandsstraff
  + w_hård    · Σ överträdelse²                  hård gräns
  + w_ändring · Σ (Δoffset)²                     lugn styrsignal
  + w_terminal·(Ti_N − börvärde)²                slutläge
  − lagrad_energi_i_plattan · pris / COP         ← avgörande
```

Den sista termen är inte kosmetika. Utan den har optimeraren en uppenbar fusk-strategi:
avsluta horisonten med kall betongplatta. Det ser billigt ut på elmätaren och ångras
tyst vid nästa omplanering, så vinsten realiseras aldrig i sluten loop. I den här koden
prissattes den lagrade värmen till vad den skulle kosta att köpa — då försvinner
incitamentet helt, och äkta förladdning när elen är billig belönas korrekt.

> Under utvecklingen gav den öppna slingan **+13 %** besparing innan den termen fanns —
> och den slutna slingan **−0,9 %**. Efter fixen: **+1,6 %** öppet, **+7,7 %** slutet.
> Den öppna siffran var ren självbedrägeri.

Sökningen är en cross-entropy-metod över batchade utrullningar, efterpolerad med
L-BFGS-B (ändlig-differens-gradienten beräknas som *en* batchad utrullning av 13
störda scheman, inte 13 separata anrop — det är det som gör polisheringen prisvärd på
en Pi). Sökpopulationen sås bland annat med hela rutnätet av konstanta offset, vilket
garanterar att MPC:n aldrig kan bli sämre än att bara låta offseten stå still.

En lösning tar ~0,5–1 s. Bara första blocket verkställs; resten planeras om nästa cykel.

### 4. Regulatorn (`controller.py`)

Betongplattans temperatur mäts inte. Den skattas med en Luenberger-observatör: modellen
propageras en cykel framåt, den uppmätta innetemperaturen adopteras rakt av, och
prediktionsfelet knuffar plattans skattning. Vid kallstart görs en 24-timmars
inkörning mot verklig historik i stället för en gissning.

---

## Snabbstart utan Home Assistant

Hela kedjan körs mot ett syntetiskt hus med känd sanning — bra för att se att det
fungerar innan du kopplar in något:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

hpmpc demo --days 25 --backtest-days 3
```

Den genererar 25 dygn syntetisk historik, identifierar huset, jämför de återfunna
parametrarna mot sanningen och kör ett backtest. Ungefär vad du ska se:

```
Curve:     slope 0.3518, offset 22.956 C, filter 3.0 h, R2 0.9933 - applied
Building:  validation RMSE 0.146 C over 47 windows (persistence 1.176 C)
           UA 221.7 W/K, time constants {'air': 1.95, 'slab': 8.38, 'envelope': 91.57}
COP:       Carnot efficiency 0.4288, standby 30 W, observed SCOP 3.259
```

---

## Komma igång på riktigt

### 1. Förutsättningar

Home Assistant med recorder som sparar minst 3–4 veckor för de aktuella entiteterna
(default är 10 dagar — höj `recorder.purge_keep_days` eller lägg entiteterna i en
egen `recorder`-`include`).

Nödvändigt: **innetemperatur** och **utetemperatur**.
Starkt rekommenderat: **framledningstemperatur** (gör modellanpassningen dramatiskt
bättre) och **elpris**.
Nyttigt: eleffekt till pumpen (ger COP-anpassning), vind, molnighet, väderentitet.

### 2. Installera

```bash
git clone <detta repo> && cd machine-learning
python -m venv .venv && source .venv/bin/activate
pip install -e .

hpmpc init-config --config config/config.yaml
export HA_TOKEN="<long-lived access token från din HA-profil>"
```

Fyll i entitets-id:n, din position och värmepumpens värmekurva i `config/config.yaml`.
Sedan:

```bash
hpmpc check
```

Den listar varje entitet, dess värde och ålder, och klagar tydligt på det som saknas.

### 3. Samla data

```bash
hpmpc collect --days 45
```

Läs varningen i slutet. Om `offset_excitation.std` är nära noll saknar datan den
variation som behövs — se nästa avsnitt.

### 4. Excitera, träna, kör

```bash
hpmpc excite            # ~1 vecka under eldningssäsong
hpmpc collect --days 45
hpmpc train
hpmpc plan              # visa planen utan att skriva något
hpmpc run               # skarp drift
```

---

## Excitation — det viktigaste steget

Det här är det steg som brukar hoppas över och som avgör om modellen blir något värd.

Om offseten aldrig rör sig i historiken kan anpassningen inte veta **hur mycket** huset
reagerar på en offsetändring. Den ser bara utetemperatur och innetemperatur samvariera
och kan inte separera "huset svarar starkt på offset" från "det blev varmare ute".
Modellen blir då utmärkt på att beskriva det förflutna och oduglig på att styra.

```bash
hpmpc excite --hold-hours 6
```

Håller ett pseudoslumpmässigt offset i sextimmarsblock. Komfortvakten är aktiv hela
tiden — huset lämnar aldrig det hårda bandet. En vecka under eldningssäsong räcker
gott. Kör det gärna igen efter en säsong.

`hpmpc collect` varnar automatiskt när excitationen är för svag.

---

## Träning och vad siffrorna betyder

```
Building:  validation RMSE 0.146 C over 47 windows (persistence 1.176 C)
           48 h horizon RMSE 0.121 C
           UA 221.7 W/K, time constants {'air': 1.95, 'slab': 8.38, 'envelope': 91.57}
           identifiability: condition number 236, weakest direction ['f_sol_i', 'A_sol']
```

- **validation RMSE** — fel i flerstegsprediktion av innetemperaturen över 12 h på data
  som inte använts i anpassningen. Under 0,3 °C är bra, under 0,5 °C användbart.
- **persistence** — samma mått för "gissa att temperaturen står still". Modellen måste
  slå den med god marginal, annars har den inte lärt sig något.
- **48 h horizon RMSE** — samma sak över två dygn. Det är den siffran som säger om
  betongplattans dynamik är rätt, och alltså om lastförflyttningen kommer att fungera.
- **UA** — husets värmeförlustkoefficient i W/K. Jämför med din egen känsla: ett hus som
  drar 5 kW vid −5 ute och 21 inne har UA ≈ 190 W/K.
- **identifiability** — hur väl datan faktiskt bestämmer *varje enskild* parameter.
  Ett högt konditionstal betyder att flera parametrar handlas mot varandra: prediktionen
  kan vara utmärkt medan de individuella talen är närmast godtyckliga. `weakest_direction`
  pekar ut vilka. Citera inte ett U-värde ur den här anpassningen utan att titta här först.

Anpassningen använder två fönsteruppsättningar samtidigt — korta (12 h) som binder den
snabba luftdynamiken och långa (48 h) som binder plattan. Bara korta fönster lämnar
värmelagringskapaciteten — själva grunden för hela idén — praktiskt taget obestämd.

Varje fönster föregås av en inkörningsperiod som simuleras men inte poängsätts, så
plattans okända starttemperatur hinner glömmas bort innan felet räknas.

---

## Backtest

```bash
hpmpc backtest --days 7
```

Spelar upp historiken: vid varje styrcykel planerar MPC:n om mot det väder och de priser
som faktiskt inträffade, verkställer sitt första block, och modellen stegas framåt.

```
                               MPC    constant
electricity (kWh)             76.0        80.1
cost (SEK)                   56.34       61.04
mean indoor (C)              21.16       21.18
Kh outside comfort            0.13        2.26

Saving: 4.70 SEK (7.7 %) at equal average indoor temperature
Energy: -5.1 % kWh
```

Jämförelsen görs mot det **konstanta offset som ger samma medelinnetemperatur**. Att
jämföra mot offset 0 vore ohederligt — en del av besparingen skulle bara vara ett kallare
hus.

Läs förbehållen som skrivs ut: i uppspelningen *är* modellen verkligheten, och prognoserna
är perfekta. Siffran är alltså en övre gräns. Modellens egen träffsäkerhet är den andra
halvan av sanningen och rapporteras av `hpmpc train`.

---

## Drift

### Docker

```bash
echo "HA_TOKEN=<token>" > .env
docker compose up -d
docker compose run --rm trainer     # samla + träna på begäran
```

Sätt i så fall `paths` i `config.yaml` till volymen:

```yaml
paths:
  data_dir: /data
  model_dir: /data/models
  state_file: /data/controller_state.json
```

### systemd

```ini
[Unit]
Description=Heat pump MPC
After=network-online.target

[Service]
Type=simple
User=hpmpc
WorkingDirectory=/opt/hpmpc
Environment=HA_TOKEN=...
ExecStart=/opt/hpmpc/.venv/bin/hpmpc --config /opt/hpmpc/config/config.yaml serve
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
```

Träna om en gång i månaden under säsong:

```
0 4 1 * *  cd /opt/hpmpc && .venv/bin/hpmpc collect && .venv/bin/hpmpc train
```

### HTTP-API

| Endpoint | Beskrivning |
|---|---|
| `GET /health` | öppen; status, drifttid, senaste fel |
| `GET /status` | senaste cykelns rapport |
| `GET /plan` | löser en gång och returnerar planen utan att skriva |
| `POST /step` | kör en styrcykel nu |
| `GET /model` | parametrar, tidskonstanter, träningsmetadata |
| `GET /metrics` | Prometheus-format |

Sätt `HPMPC_API_KEY` för att kräva `X-API-Key` på allt utom `/health`.

### Home Assistant-paket

`ha/packages/heatpump_mpc.yaml` innehåller offset-hjälparen, dashboard-sensorer och —
viktigast — en **dödmansgrepp-automation**.

---

## Säkerhet

Ett fastnat extremt offset är den enda felmoden som verkligen kostar pengar eller komfort.
Fyra oberoende lager:

1. **Klamring och rampbegränsning** — `offset_min`/`offset_max` och högst
   `max_change_per_cycle` kelvin per cykel.
2. **Hårt komfortband** — under `hard_min` går regulatorn till maximal värme och över
   `hard_max` till minimal, oavsett vad optimeraren tycker.
3. **Sensorvakt** — saknad, orimlig eller för gammal data (`max_data_age_minutes`) gör att
   regulatorn *gradvis* går mot `fallback_offset` och loggar varför. Den fryser aldrig
   fast ett gammalt extremvärde.
4. **Dödmansgrepp i Home Assistant** — om regulatorn slutar rapportera nollställer HA
   offseten själv och skickar en notis. Det lagret gäller även om hela maskinen dör.

Lägg till ett femte i hårdvaran: ett relä som defaultar till den **riktiga** utegivaren
när ESP:n är strömlös. Pumpen ska alltid se en trovärdig givare, särskilt när det här
projektet inte kör.

Kör gärna `dry_run: true` (eller `hpmpc plan`) i några dygn först och läs planerna innan
du släpper det skarpt.

---

## Hårdvara: det digitala motståndet

Pumpen läser sin utegivare som ett motstånd. En NTC på 22 kΩ/25 °C ligger på ~68 kΩ vid
0 °C och ~200 kΩ vid −20 °C — ingen digital potentiometer täcker det spannet. Det behöver
den inte heller: du behöver bara några kelvins auktoritet kring den verkliga temperaturen.

Praktiskt: **en digital potentiometer i serie med ett fast motstånd**, dimensionerat så
att fönstret ligger kring dina normala utetemperaturer. `ha/esphome_digital_resistor.yaml`
är ett komplett ESPHome-exempel med en DS3502.

Kontrollera upplösningen innan du löder:

```bash
hpmpc ntc-table --step-ohm 78
```

```
  temp (C)         ohm    K per 78.0 ohm
     -20.0      199741             0.007
       0.0       68501             0.023
      10.0       42456             0.040
```

Ett steg som är värt mer än ~0,2 K börjar kvantisera styrningen märkbart.

Tre sätt att få ut offseten (`control.output_mode`):

| Läge | Skriver | Använd när |
|---|---|---|
| `offset` | kelvin | pumpen har en egen offset-parameter, eller HA räknar om själv |
| `fake_temperature` | °C | din hårdvara tar emot en temperatur |
| `resistance` | ohm | hpmpc räknar om via NTC-kurvan åt dig |

För `resistance`: ta hellre tabellen ur pumpens servicemanual (`ntc.model: table`) än
beta-modellen. Riktiga givare följer inte en tvåparametersmodell över hela spannet.

---

## Vad kan man realistiskt spara?

Backtestet ovan ger 7,7 % lägre kostnad vid samma medelinnetemperatur — och 5 % mindre
energi, eftersom optimeraren också väljer *varmare* timmar med bättre COP, inte bara
billigare timmar.

Räkna med mindre i verkligheten: prognoser är inte perfekta och modellen är inte huset.
Faktorer som avgör hur mycket du får ut:

- **Prisspridningen.** Rörligt avtal med stor dygnsvariation ger mycket; fast pris ger
  nästan ingenting (då blir vinsten enbart COP-optimering, ett par procent).
- **Bredden på komfortbandet.** Med `comfort_min` 20,3 och `comfort_max` 22,0 finns det
  1,7 K att arbeta med. Krymper du bandet krymper vinsten proportionellt.
- **Betongplattans massa.** Tunn platta på träbjälklag är ett mycket sämre värmelager.
  Kolla `time_constants_hours.slab` efter träning: under ~5 h finns lite att hämta.
- **Sätt `price_addition`.** Elöverföring och energiskatt är typiskt 0,50–0,80 kr/kWh i
  Sverige. De ingår i marginalkostnaden men dämpar den *relativa* spridningen — utan dem
  överskattar optimeraren vinsten av lastförflyttning.

---

## Begränsningar

Saker som är värda att veta innan du litar på det här:

- **Parametrarna är inte unikt bestämda.** Prediktionen kan vara utmärkt medan enskilda
  parametrar är långt från sanningen. Därför rapporteras `identifiability` — läs den.
- **Pumpen modelleras kontinuerligt.** Verkliga pumpar taktar, har startspärrar, gör
  avfrostning och prioriterar varmvatten. Avfrostningen finns som en COP-korrektion;
  taktning och varmvattenprioritering gör det inte.
- **Varmvatten ingår inte.** Om du vill lastförflytta varmvattenberedningen är det ett
  separat problem — men prisprognosen härifrån går att återanvända.
- **Elprisprognosen slutar vid morgondagens sista timme.** Bortom det extrapoleras sista
  kända priset; horisontens sista timmar är alltså en kvalificerad gissning. Fältet
  `price_extrapolated_hours` i rapporten säger hur mycket.
- **Backtestets besparing är en övre gräns.** Se avsnittet ovan.
- **Modellen åldras.** Träna om varje månad under säsong; huset ändrar sig med lövverk,
  vind, snötäcke och hur ni faktiskt bor.

---

## Utveckling

```
src/hpmpc/
├── config.py         konfiguration, env-expansion, validering
├── ha.py             Home Assistant REST-klient
├── dataset.py        historik → regelbunden träningsmatris
├── solar.py          solgeometri + klarhimmelsmodell (ersätter en molntjänst)
├── ntc.py            temperatur ↔ resistans
├── model/
│   ├── heatpump.py   värmekurva, utegivarfilter, COP, effekt
│   └── thermal.py    2R2C-modellen, batchad simulering
├── identify.py       systemidentifiering + identifierbarhetsdiagnostik
├── residual.py       lärd residual (scikit-learn)
├── forecast.py       väder- och prisprognos
├── mpc.py            optimeraren
├── controller.py     styrslinga, observatör, säkerhet
├── evaluate.py       backtest
├── train.py          träningsorkestrering, modellpersistens
├── simulator.py      syntetiskt hus (demo och tester)
├── api.py            HTTP-API
└── cli.py            kommandoradsgränssnitt
```

```bash
pip install -e ".[dev]"
pytest -q                       # hela sviten
pytest -q -k "not identify and not pipeline"   # snabb delmängd
```

Testsviten kör mot syntetisk data med känd sanning och verifierar bland annat att
identifieringen återfinner värmekurvan och husets parametrar, att MPC:n aldrig är sämre
än bästa konstanta offset, att terminalvärderingen tar bort horisont-fusket, och att
varje felväg i regulatorn leder till fallback i stället för ett fastnat offset.

MIT-licens.
