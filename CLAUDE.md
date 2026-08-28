# hpmpc — projektminne

Självhostad modellprediktiv styrning (MPC) av en luft/vatten-värmepump på
golvvärme. Systemet lär sig huset från Home Assistants historik och styr pumpen
genom att manipulera vilken utetemperatur den *tror* att den ser.

Läs `README.md` för hur det fungerar och `INSTALL.md` för hur det installeras.
Den här filen är till för att snabbt komma in i projektet igen: vad som är
bestämt, varför, och vilka fällor som redan är upptäckta.

---

## Anläggningen

| | |
|---|---|
| Värmepump | Daikin Altherma LT: **ERLQ016CAW1** utedel + **EHVH16S26CB9W** hydrobox (260 l, 9 kW elpatron) |
| Distribution | Golvvärme i betongplatta |
| Ort | Falkvägen, Norrköping (58.5877, 16.1924) |
| Elområde | SE3, rörligt pris |
| Elöverföring + energiskatt | **0,7084 kr/kWh exkl. moms** (= 0,8855 inkl.) |
| Ställdon | ESP32 + **MCP41100** (8 bitar, 100 kΩ, SPI) på pumpens utegivare |
| Effektmätning | Victron, **hela husets effekt per fas** — ingen mätare enbart på pumpen |
| Elbilsladdare | 11 kW över alla tre faser, `binary_sensor.eh6nh5cd_charging` (`Charging` / `Not charging`) |
| Körs på | NUC, Docker (Portainer), skilt från Home Assistant |
| Utegivare | **ingen** — utetemperaturen hämtas från SMHI. `entities.outdoor_temp` finns kvar för att koppla in en givare senare |

Entiteterna, som de faktiskt heter:

| roll | entity_id |
|---|---|
| innetemperatur (mitt i huset) | `sensor.hall_temperature_2` |
| husets effekt per fas | `sensor.gx_device_consumption_power_l1` / `_l2` / `_l3` |
| laddstatus | `binary_sensor.eh6nh5cd_charging` |
| wiperavläsning från ESP32 | `sensor.varmepump_proxy_mcp41100_wiper_0_255` |
| utgångar | `input_number.varmepump_offset`, `input_number.varmepump_fiktiv_utetemp` |

Användaren skriver svenska. Svara på svenska; kod och kommentarer på engelska.

---

## Arkitektur

```
   SMHI ──väder──┐                      ┌── prestandakarta (COP, kapacitet, elpatron)
elprisetjustnu ──┤                      │
                 ▼                      ▼
Home Assistant ──historik──►  dataset ──►  systemidentifiering ──►  husmodell
      │                                                                 │
      │                              MPC-optimerare  ◄──────────────────┤
      │                                    │                            │
      │◄── ohm ──  ESP32 + digitalt ◄──────┘                        observatör
      │            motstånd  ──►  värmepumpens utegivare
```

| Modul | Ansvar |
|---|---|
| `model/thermal.py` | 2R2C-husmodell (luft, platta, pumpens filtrerade utegivare), batchad simulering |
| `model/performance.py` | COP/kapacitet per maskin, elpatron, avfrostning, driftgränser |
| `model/heatpump.py` | Värmekurva, utegivarfilter, `PumpModel`, `OperatingPoint` |
| `identify.py` | Systemidentifiering + identifierbarhetsdiagnostik + verkningsgradskalibrering |
| `disaggregate.py` | Dela husets effekt i pump / laddare / baslast |
| `archive.py` | Egen kopia av recorderhistoriken, en gzippad CSV per månad |
| `residual.py` | Lärd residual (scikit-learn), enbart exogena särdrag |
| `mpc.py` | CEM-optimerare med batchade utrullningar |
| `comfort.py` | Börvärde, lägen, komfortschema över horisonten |
| `controller.py` | Styrslinga, observatör, säkerhet, lägen, utgångar |
| `ntc.py` | Givarkurva (beta/tabell) + potentiometerns geometri och räckvidd |
| `providers/` | SMHI-prognos, SE3-spotpris, geokodning |
| `settings.py` | Inställningar i drift + säker redigering av `config.yaml` |
| `evaluate.py` | Backtest |
| `simulator.py` | Syntetiskt hus med känd sanning (demo + tester) |

---

## Beslut som är lätta att råka riva upp

Det här är sådant som ser ut som godtyckliga val men som det finns skäl bakom.
Ändra gärna — men vet vad du ändrar.

**Grey-box, inte neuralt nät.** Med 3–6 veckors data är ett neuralt nät hopplöst
underbestämt och extrapolerar farligt vid temperaturer det aldrig sett. Tio
fysikaliskt tolkbara parametrar lär sig lika bra och uppför sig utanför datan.

**Prestandakartan lagrar Carnot-verkningsgrad, inte COP.** Verkningsgraden ligger
i ett smalt band (0,35–0,42) över hela driftområdet, så interpolation är stabil
och extrapolation utanför tabellen förblir fysikalisk. Interpolerar man rå COP
mellan mätpunkter får man nonsens vid små lyft.

**Terminalvärdering av lagrad energi i `mpc.py`.** Utan den avslutar optimeraren
varje horisont med kall betongplatta: billigt inuti horisonten, ångras tyst vid
nästa omplanering. Öppen slinga påstod **+13 %** besparing på data där sluten
slinga gav **−0,9 %**. Med lagrad värme prissatt till vad den kostar att köpa
försvinner incitamentet.

**Omstartsval i `identify.py` sker på det regulariserade träningsobjektivet**, inte
på validerings-RMSE. En hundradels grad är brus på en platt likelihood-ås, och
att föredra det lät en anpassning vinna just genom att smita undan
regulariseringen — den gav UA 140 W/K mot sanna 195.

**Historiken kopieras ur recordern varje styrcykel** (`archive.py`). Recordern är
ett rullande fönster som rensas av ett annat system; identifieringen vill ha sex
veckor. Att kräva `purge_keep_days: 45` gör modellen beroende av en inställning
ingen minns, i en databas som återställs från backup. Arkivet frågar bara efter
det som hänt sedan senaste lagrade raden, så recordern behöver bara hålla längre
än glappet mellan två cykler. **Bara råa signaler lagras** — solinstrålning och
offset i kelvin härleds vid läsning, annars bär gamla rader runt gamla fel efter
att NTC-tabellen rättats. Överlappet på två timmar finns för att den nyaste
resampling-luckan alltid är halvfylld när den skrivs.

**Residualmodellen får bara exogena särdrag** (klocka, sol, vind, utetemperatur).
Aldrig husets tillstånd, aldrig styrsignalen. Annars kan optimeraren utnyttja
den i en återkopplingsslinga, och den kan inte förberäknas per lösning.

**Baslastmodellen i `disaggregate.py` får bara klockan och veckodagen.** Ingen
utetemperatur. Vilken som helst variabel som korrelerar med det som driver
värmepumpen skulle låta baslasttermen suga upp pumpens signal — det enda fel
som tyst skulle korrumpera verkningsgradsuppskattningen.

**Symmetrisk robust förlust i effektuppdelningen** (`power.asymmetry: 1.0`).
Asymmetri lät logisk — hushållslaster är positiva spikar — men baslasttermen
absorberar redan medelapparatlasten, så asymmetrin bara biasar verkningsgraden
uppåt med 4 %. Huber-vikten sköter robustheten.

**Komfortbandet är relativt börvärdet.** Med absoluta gränser går det att sätta
semesterbörvärde 16 och lämna kvar ett komfortband som kräver 20,3 — huset kyls
aldrig ner och inställningen ser ut att inte göra någonting.

**Nedsänkningsband är osymmetriska.** När ingen är hemma är för kallt det enda
som spelar roll. En snäv övre gräns *förbjuder* dessutom återuppvärmning inför
hemkomst — den gjorde exakt det, med 400× hårt straff, innan profilerna vidgades.

**Semesterläget har egen `offset_max: 25`.** Värmekurvan är byggd för 21 °C. För
att glida ner till 16 °C vid −6 °C ute måste pumpen visas ca **+14 °C**, alltså
tjugo kelvins offset, inte fyra. `heat_pump.perceived_max_c` är den absoluta
spärren som håller det säkert.

**MPC-sökningen sås med hela rutnätet av konstanta offset.** Garanterar att
resultatet aldrig kan bli sämre än att bara låta offseten stå still.

**Polish-steget beräknar gradienten som *en* batchad utrullning** av n+1 störda
scheman, inte n+1 separata anrop. Det är skillnaden mellan 7 s och 0,5 s per
lösning.

**COP och kapacitet förberäknas utanför tillståndsloopen** (`OperatingPoint`).
De beror bara på kända insignaler. Tiofaldig skillnad i lösningstid.

**Backtestets besparing räknas netto efter värme som lämnas i plattan.** Annars
belönas den körning som råkar sluta med kall platta — samma fel som
terminalvärderingen fixar, fast i utvärderingen.

---

## Verifierat kontra antaget

**Verifierat** (311 tester, syntetiskt hus med känd sanning):
- Identifieringen återfinner värmekurva (lutning 0,3495 mot 0,35, R² 0,997) och
  husparametrar (UA +3 %, Ci +4 %, `k_wind` +3 %, plattans tidskonstant inom 8 %).
- Prediktionsfel 0,085 °C över 12 h, 0,066 °C över 48 h (persistensbaslinje 1,01 °C).
- Sluten loop: 5,3 % lägre kostnad vid samma medelinnetemperatur, komfort
  0,8 mot 11,0 kelvintimmar utanför bandet.
- Effektuppdelningen: verkningsgradsskala inom 2 % på 30 dygn, laddaren skattas
  till 11,2 kW mot nominella 11.
- Hemkomstplaneringen härleder framförhållningen: 20 h → startar direkt,
  30 h → väntar till t+16 h.

**Antaget / ej verifierat:**
- **Prestandakartans siffror är förankrade i publicerade mätpunkter för
  maskinklassen, inte hämtade ur Daikins databook** — den gick inte att nå från
  utvecklingsmiljön. Formen är rimlig, nivån kalibreras mot elmätaren. Byt
  tabellen när databooken finns.
- **SMHI och elprisetjustnu har aldrig anropats live** — utgående trafik dit var
  blockerad av sessionens policy. Parsning, cachning, felhantering och
  reservvägar är testade mot mockade svar. `hpmpc providers` är första
  kommandot att köra på riktig maskin.
- **`pot:`-siffrorna är MCP41100:ans typvärden** (100 kΩ ±20 %, 256 steg,
  ~100 Ω wiper), inte uppmätta på den faktiska kretsen. Mät med multimeter vid
  idrifttagning och skriv in `resistance_ohm`/`wiper_ohm`.
- **NTC-tabellen är en typisk Daikin 20 kΩ-givare**, inte uppmätt på den
  faktiska givaren. `hpmpc calibrate-ntc` finns för att rätta det, och
  `entities.pump_outdoor_temp` för att verifiera det kontinuerligt — men den
  pumpen visar bara talet på displayen, så i praktiken är `entities.pot_wiper`
  den enda återkopplingen och den når bara fram till ESP32:n.
- **Docker-imagen är inte byggd** — ingen docker-daemon i utvecklingsmiljön. Den
  motsvarande wheel-installationen är verifierad, inklusive paketdata.
- **Ingenting har körts mot en riktig Home Assistant.** HA-klienten är testad mot
  en fake som härmar REST-API:ets format.

---

## Fällor som redan är upptäckta

- **Faka inte R1T i utedelen.** Den styr också avfrostningstiming,
  driftområdesgränser och om aggregatet får gå. Daikins innedel kan ta en extern
  utegivare (KRCS01-1) och via en field setting använda *den* för kurvan. Koden
  för den inställningen skiljer mellan generationer — slå upp i installatörsmanualen.
- **MCP41100:ans problem är räckvidden, inte upplösningen.** Det var fel i den
  här filen tidigare. 392 Ω per steg ger 0,10 K vid nollan och 0,29 K vid +20 —
  gott nog. Men 100 kΩ tar slut vid **−7,4 °C** på Daikinkurvan, och kallare än
  så finns ingen wiperposition. Under −7 ute sitter wipern i ändläge, pumpen
  visas −7 när det är −15, och huset underhettar tyst. Därför är
  `heat_pump.perceived_min_c: -7` i exempelkonfigurationen, därför varnar
  `hpmpc check`/`ntc-table` när den och `pot:` inte går ihop, och därför läses
  `entities.pot_wiper` tillbaka varje cykel. Fixen är en **andra MCP41100 i
  serie** och `pot.devices: 2` → −20,3 °C vid samma steglängd. Seriekopplade
  kretsar ger räckvidd, inte upplösning.
- **Pumpen har ingen egen utegivare kvar** — potentiometern *är* givaren.
  Därför är "koppla bort emulatorn" aldrig ett säkert läge: det ger pumpen ett
  brutet givarkretslopp. Varje reservväg faller tillbaka på ett rimligt
  motstånd: HA skriver riktig utetemperatur med offset 0, ESP32:n håller sitt
  senaste värde i fyra timmar och går sedan till `FAILSAFE_WIPER` (~0 °C), och
  reläets NC-kontakt ska ha en **fast resistor** (~68 kΩ) för det strömlösa
  fallet. Det här revs upp en gång: en tidigare version slutade skriva under
  `perceived_min_c`, vilket var precis fel.
- **Under `perceived_min_c` kommenderas det kallaste hårdvaran kan visa**, och
  gapet rapporteras kvantifierat (`range_shortfall`, gap × `curve_slope` = kelvin
  framledning som fattas). `heat_pump.perceived_min_c` är ändringsbar i drift, så
  den andra MCP41100:an kan tas i bruk utan att röra filen.
- **`_limit` lägger den perceived-gränsen sist**, efter hastighetsbegränsningen.
  Annars kommenderas ett värde potentiometern klipper på egen hand, och modellen
  tror att det tillämpades.
- **`pot:` är skild från `ntc:` med flit.** `ntc:` är givarkurvan man kalibrerar,
  `pot:` är vad hårdvaran kan. En omkalibrering av kurvan får inte tyst ändra
  hårdvarans gränser.
- **Elbilsladdarens sensor säger `Charging`/`Not charging`**, inte `on`/`off`.
  `ha.BOOLEAN_STATES` mappar båda.
- **Nord Pool avräknar i kvartstimmar** — 96 priser per dygn. Ingenting i koden
  antar upplösning; den läses ur `time_start`.
- **Morgondagens priser finns inte före ~13:00.** Det är normaltillstånd, inte fel.
- **Utan excitation blir modellen värdelös för styrning.** Rör sig aldrig offseten
  i historiken kan anpassningen inte skilja "huset svarar starkt på offset" från
  "det blev varmare ute". Excitationen bryter också kopplingen mellan
  värmepumpens effekt och klockan i effektuppdelningen.
- **`hpmpc power` visar skalan *relativt* den som redan sitter i modellen.** Ett
  värde nära 1,0 betyder att förra kalibreringen fortfarande stämmer.
- **Ett jämnt fel i NTC-tabellen absorberas av kurvanpassningen.**
  `fit_heating_curve` regresserar uppmätt framledning mot *kommenderad* offset,
  så konstant fel och skalfel hamnar i `curve_offset`/`curve_slope` och
  styrningen predikterar ändå rätt. Det som *inte* absorberas är allt som hänger
  på ett absolut tröskelvärde: pumpens `heat_stop_temp`, `perceived_min_c/max_c`
  och semesterläget — som fungerar just genom att passera värmestoppet.
- **Regulatorn skriver alla konfigurerade utgångar samtidigt**, inget `output_mode`.
  Samma beslut i kelvin, grader och ohm — då kan de inte säga emot varandra. Historiken
  läses tillbaka från kelvin-entiteten eftersom den inte kräver någon omräkning.
- **Arkivet och recordern kan inte hamna i konflikt.** Raderas arkivet fyller det
  sig från det recordern har kvar; kortas recorderns retention behåller arkivet
  det redan kopierat. `training.archive: false` går direkt mot recordern som förut.
- **Utetemperaturen kommer från SMHI** eftersom `entities.outdoor_temp` är tom här.
  En givare vid huset är bättre när den finns — den mäter luften byggnaden faktiskt
  förlorar värme till — och vinner automatiskt så fort entiteten fylls i. Det ska
  inte vara ett krav att ha HA i loopen för det.
- **Kalibrera mot pumpens display, inte mot givaren.** Ett par avlästa som
  "jag skickade R, pumpen säger T" innefattar kabelresistans, kontakt och
  pumpens egen linjärisering. En bänkmätning av termistorn missar allt det.

---

## Konventioner

- Python ≥ 3.10, bara numpy / scipy / pandas / scikit-learn / PyYAML / httpx /
  FastAPI. Inga tunga beroenden, inget som kräver GPU, inga molntjänster.
- All fysik vektoriserad över en ledande batchdimension — optimeraren rullar ut
  hundratals kandidater per anrop.
- Konfiguration är en enda YAML med `${VAR:-default}`-expansion. Okända nycklar
  är fel, inte tyst ignorerade.
- Varje felväg i regulatorn leder till `fallback_offset`, aldrig till ett
  fastnat värde.
- Tester kör mot syntetisk data med känd sanning. Nya funktioner ska ha ett test
  som visar att de gör vad de påstår, inte bara att de kör.

```bash
pytest -q                                       # hela sviten, ~80 s
pytest -q -k "not identify and not pipeline"    # snabb delmängd, ~5 s
python -m pyflakes src/hpmpc tests
hpmpc demo --days 25 --backtest-days 7          # hela kedjan mot syntetiskt hus
```

---

## Vanliga kommandon

```bash
hpmpc check          # entiteter i Home Assistant
hpmpc providers      # SMHI + SE3-priser, plus faktisk marginalkostnad
hpmpc pump-table     # COP- och kapacitetstabeller
hpmpc curve --point=-15:40 --point=15:25        # Daikins tvåpunktskurva -> lutning/offset
hpmpc calibrate-ntc --point=0:66800 --point=25:20000
hpmpc ntc-table      # vad potentiometern faktiskt når, och per steg
hpmpc mode holiday   # byt komfortläge
hpmpc settings       # vad som går att ändra i drift
hpmpc set control.price_addition 0.7084
hpmpc excite         # identifieringsexperiment, ~1 vecka
hpmpc archive        # vår egen historik — omfång, hål, storlek
hpmpc collect --days 45 && hpmpc train
hpmpc power          # effektuppdelningen — laddaren ska hamna nära 11 kW
hpmpc plan           # nuvarande plan, skriver ingenting
hpmpc backtest --days 7
hpmpc run / hpmpc serve
```

---

## Nästa steg för användaren

1. `hpmpc providers` — stäm av marginalkostnaden mot elfakturan.
2. Bestäm vilken givare som ska emuleras (helst innedelens externa, inte R1T).
3. `hpmpc calibrate-ntc` på den faktiska givaren.
4. `hpmpc curve` med de två kurvpunkterna från pumpens display.
5. En vecka `hpmpc excite`.
6. `hpmpc collect && hpmpc train && hpmpc power`.
7. Några dygn med `dry_run: true`, läs planerna.
8. Skarpt.

## Möjliga vidareutvecklingar

- Varmvattenberedning: EHVH16S26 har 260 l och pausar värmedriften under
  beredning. Modellen ser inte den pausen. Prisprognosen går att återanvända.
- Taktning och startspärrar modelleras inte.
- Effekttariff (månadens toppeffekt) — `max_electric_power_kw` är en mjuk
  gräns per horisont, inte en månadsvis optimering.
