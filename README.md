# hpmpc — självhostad ML-styrning av luft/vatten-värmepump

Modellprediktiv styrning (MPC) för en luft/vatten-värmepump på golvvärme. Systemet lär
sig ditt hus från Home Assistants historik och styr pumpen genom att manipulera vilken
utetemperatur den *tror* att den ser — via ett digitalt motstånd på en ESP32.

Allt körs lokalt: numpy, scipy och scikit-learn på en Raspberry Pi eller NUC. Ingen
molntjänst, ingen AI-tjänst, ingen API-nyckel. Utgående trafik går bara till din egen
Home Assistant, till SMHI:s öppna data och till elprisetjustnu.se.

**Tar hänsyn till:** utetemperatur, innetemperatur, vindstyrka, sol/molnighet,
luftfuktighet och elpris — plus värmepumpens COP-kurva över hela driftområdet, dess
kapacitetsgräns, elpatronen, avfrostning, husets tröghet och betongplattans värmelagring.

Levereras konfigurerad för **Daikin Altherma LT** (ERLQ016CAW1 + EHVH16S26CB9W),
**Norrköping** och **elområde SE3** — men allt det är konfiguration, inte kod.

---

## Innehåll

- [Idén](#idén)
- [Arkitektur](#arkitektur)
- [Värmepumpsmodellen](#värmepumpsmodellen)
- [Datakällor](#datakällor)
- [Snabbstart utan Home Assistant](#snabbstart-utan-home-assistant)
- [Komma igång på riktigt](#komma-igång-på-riktigt)
- [Excitation — det viktigaste steget](#excitation--det-viktigaste-steget)
- [Träning och vad siffrorna betyder](#träning-och-vad-siffrorna-betyder)
- [Backtest](#backtest)
- [Drift](#drift)
- [Säkerhet](#säkerhet)
- [Hårdvara: ESP32 och det digitala motståndet](#hårdvara-esp32-och-det-digitala-motståndet)
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

> Under utvecklingen påstod den öppna slingan **+13 %** besparing innan den termen
> fanns — på data där den slutna slingan levererade **−0,9 %**. Den öppna siffran var
> ren självbedrägeri. Efter fixen ligger den öppna prognosen på blygsamma ~2 % per
> lösning och den slutna slingan levererar ~5 %, alltså mer än den lovar i stället för
> mindre.

Sökningen är en cross-entropy-metod över batchade utrullningar, efterpolerad med
L-BFGS-B (ändlig-differens-gradienten beräknas som *en* batchad utrullning av 13
störda scheman, inte 13 separata anrop — det är det som gör polisheringen prisvärd på
en Pi). Sökpopulationen sås bland annat med hela rutnätet av konstanta offset, vilket
garanterar att MPC:n aldrig kan bli sämre än att bara låta offseten stå still.

En lösning tar ~0,5–1 s. Bara första blocket verkställs; resten planeras om nästa cykel.

### 4. Värmepumpen (`model/performance.py`)

En riktig prestandakarta per maskin: COP och kapacitet över hela driftområdet,
elpatronen, avfrostning och driftgränser. Se [Värmepumpsmodellen](#värmepumpsmodellen).

### 5. Regulatorn (`controller.py`)

Betongplattans temperatur mäts inte. Den skattas med en Luenberger-observatör: modellen
propageras en cykel framåt, den uppmätta innetemperaturen adopteras rakt av, och
prediktionsfelet knuffar plattans skattning. Vid kallstart görs en 24-timmars
inkörning mot verklig historik i stället för en gissning.

---

## Värmepumpsmodellen

Det här är den del som avgör om systemet sparar pengar eller bränner dem. En generisk
Carnot-modell ser inte de två sätt på vilka en väderkompenserad pump kan bli *dyrare*
av välmenande optimering:

**1. COP är inte en jämn funktion av utetemperaturen ensam.** Att höja framledningen
för att fånga en billig timme kan kosta mer verkningsgrad än priset sparar — särskilt
när det är kallt ute och lyftet redan är stort.

**2. Kompressorn tar slut.** När effektbehovet överstiger vad kompressorn klarar
täcker hydroboxens **elpatron** resten vid COP 1,0. En optimerare som inte ser det
planerar glatt en förvärmning som tyst eldar resistiv el till tre-fyra gånger priset.

Därför bär modellen en riktig prestandakarta per maskin
(`src/hpmpc/resources/pumps/daikin_erlq016caw1.yaml`):

```
hpmpc pump-table
```
```
COP
  ambient       W30     W35     W40     W45     W50
      -20      2.21    2.02    1.86    1.74    1.61
       -7      3.18    2.82    2.54    2.32    2.11
        7      5.31    4.39    3.75    3.29    2.91

Compressor capacity (kW)
  ambient       W30     W35     W40     W45     W50
      -20       9.7     9.5     9.2     8.8     8.5
        7      16.4    16.0    15.4    14.9    14.2
```

Tabellen lagras som **Carnot-verkningsgrad**, inte som COP:

```
COP = verkningsgrad(T_ute, T_fram) · (T_fram + 273,15) / (T_fram − T_ute)
```

Det är ett medvetet val. Verkningsgraden ligger i ett smalt band (0,35–0,42) över hela
driftområdet, så interpolation är stabil och extrapolation utanför tabellen förblir
fysikaliskt rimlig. Interpolerar man rå COP mellan mätpunkter får man nonsens vid små
lyft.

Ovanpå det:

- **Kapacitetsgräns per utetemperatur och framledning**, med elpatronen som täcker
  underskottet vid COP 1,0. Det gör en för aggressiv förvärmning synligt dyr i
  målfunktionen — precis den spärr du efterfrågade.
- **Kompressorstopp** under maskinens undre driftgräns (−25 °C).
- **Avfrostning** som extra avdrag när luften är fuktig kring nollan. SMHI ger relativ
  luftfuktighet, så det här är faktiskt informerat och inte gissat. Avdraget är noll
  vid referensfuktigheten, eftersom EN14511-mätpunkterna redan innehåller normala
  avfrostningsförluster.
- **Effekttak** (`control.max_electric_power_kw`). En 16 kW-kompressor plus 9 kW
  elpatron löser ut en 25 A-servis, och en förvärmningsplan är precis när de skulle
  gå samtidigt.

### Siffrornas ursprung — läs det här

Tabellen är förankrad i publicerade mätpunkter för den här maskinklassen (A7/W35
COP 4,4; A7/W45 3,3; A7/W55 2,6; A2/W35 3,6; A−7/W35 2,8; A−7/W55 1,9; A−15/W35 2,3)
och utjämnad däremellan. Det är en **välskött prior, inte tillverkarens data.**

Två saker att göra åt det:

1. Byt ut tabellen mot den riktiga från Daikins databook (avsnittet med
   kapacitetstabeller för värmedrift; verkningsgrad räknas ut med formeln ovan).
   Peka `heat_pump.model` på din egen YAML-fil.
2. **Viktigare:** låt `hpmpc train` kalibrera den mot din egen elmätare. Efter det
   sätter den levererade tabellen bara *formen*, och din mätare sätter *nivån*.
   Träningen anpassar en enda multiplikativ skala — ett väldeterminerat tal slår tio
   dåligt bestämda — och rapporterar effektfelet per utetemperaturintervall så att en
   felaktig *form* syns istället för att absorberas in i skalan:

```
Pump:      efficiency scale 0.94, standby 71 W, observed SCOP 3.14, power RMSE 132 W
           backup heater active 3.5 h in this data
           power error by outdoor bin (W): {'-10..-5C': -18.2, '-5..0C': 5.1, '0..5C': 22.4}
```

En trend över intervallen betyder att formen är fel — då är det dags för databooken.

---

## Datakällor

Systemet hämtar väder och elpris själv. Home Assistant behövs bara för husets egna
givare och för att skriva ut styrsignalen.

### Väder: SMHI öppna data

Punktprognos ur samma modell som ligger bakom den nationella prognosen. Gratis, ingen
nyckel, ingen registrering, cirka tio dygn framåt.

```
https://opendata-download-metfcst.smhi.se/api/category/pmp3g/version/2
    /geotype/point/lon/16.192400/lat/58.587700/data.json
```

Används: `t` (temperatur), `ws` (vind), `tcc_mean` (molnighet i åttondelar) och `r`
(relativ fuktighet). Fuktigheten spelar större roll än man tror — den styr hur ofta
pumpen måste avfrosta, och avfrostning är ren förlust.

Prognosen cachas i 30 minuter (SMHI uppdaterar per timme) och en gammal cache används
hellre än ingen prognos alls — men det syns i rapporten istället för att döljas.

Koordinater: kör `hpmpc geocode "Falkvägen, Norrköping"` en gång. SMHI:s rutnät är
ungefär 2,5 km, så var i Norrköping du står spelar ingen roll — och prognosen ankras
ändå mot din egen utegivare vid horisontens början.

### Elpris: SE3 via elprisetjustnu.se

Nord Pools day ahead-priser per elområde, gratis och utan nyckel:

```
https://www.elprisetjustnu.se/api/v1/prices/2026/08-26_SE3.json
```

Morgondagens fil finns **inte** förrän Nord Pool publicerat, strax efter 13:00 svensk
tid. Det är normalt, inte ett fel: före 13 kortas horisonten av och det sista kända
priset extrapoleras. `price_extrapolated_hours` i rapporten säger hur mycket.

**Priserna är rå spot exklusive moms, överföring och energiskatt.** Det som avgör om
lastförflyttning lönar sig är marginalkostnaden:

```
marginal = (spot × price_scale + price_addition) × (1 + price_vat_pct/100)
```

Sätt `price_addition` till din överföringsavgift plus energiskatt (typiskt 0,50–0,80
kr/kWh) och `price_vat_pct: 25`. Hoppar man över det överdriver optimeraren den
*relativa* spridningen mellan billiga och dyra timmar och blir för ivrig.
`hpmpc providers` varnar om båda står på noll.

Verifiera att båda källorna svarar från din maskin:

```
hpmpc providers
```

> **Notera:** de här två endpointsen kunde inte anropas från miljön där koden skrevs —
> utgående trafik dit var blockerad av en policy. Parsning, cachning, felhantering och
> reservvägar är testade mot mockade svar, men `hpmpc providers` är det första du bör
> köra.

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
Building:  validation RMSE 0.098 C over 47 windows (persistence 1.013 C)
           48 h horizon RMSE 0.075 C
           UA 177.5 W/K, time constants {'air': 2.13, 'slab': 10.32, 'envelope': 140.03}
           identifiability: condition number 260, weakest direction ['f_sol_i', 'Hme', 'A_sol']
COP:       Carnot efficiency 0.4284, standby 71 W, observed SCOP 3.144

     parameter         true      fitted     error
     Ci              2100.0      1996.0       -5%
     Cm             26000.0     22855.1      -12%
     Hie              165.0       147.2      -11%
     Hfloor          1350.0      1393.3       +3%
     A_sol              5.5         4.1      -26%   <- flaggad som svagast identifierad
     UA (W/K)         195.0       177.5       -9%
```

Notera `A_sol`: identifierbarhetsrapporten pekar ut solaperturen som den svagast
bestämda riktningen, och det är precis den parameter som avviker mest. Diagnostiken
säger alltså sanningen om sig själv.

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

Exempelkonfigurationen är redan ifylld för Daikin Altherma LT, Norrköping och SE3.
Byt ut entitets-id:n mot dina egna.

### 3. Kontrollera att allt svarar

```bash
hpmpc check          # entiteter i Home Assistant: värde, ålder, det som saknas
hpmpc providers      # SMHI och SE3-priser, plus din faktiska marginalkostnad
hpmpc pump-table     # COP- och kapacitetstabellerna — jämför mot databooken
```

### 4. Ställ in värmekurvan och givaren

Daikin anger kurvan som två punkter. Räkna om dem istället för att göra det för hand —
det här är det enda talet som styr allt annat:

```bash
hpmpc curve --point=-15:40 --point=15:25
```

Mät din egen utegivare vid två eller tre kända temperaturer (isvatten är en gratis och
exakt nollpunkt) och anpassa NTC-modellen:

```bash
hpmpc calibrate-ntc --point=0:66800 --point=25:20000
hpmpc ntc-table --step-ohm 195     # kollar upplösningen din hårdvara ger
```

### 5. Samla data

```bash
hpmpc collect --days 45
```

Läs varningen i slutet. Om `offset_excitation.std` är nära noll saknar datan den
variation som behövs — se nästa avsnitt.

### 6. Excitera, träna, kör

```bash
hpmpc excite            # ~1 vecka under eldningssäsong
hpmpc collect --days 45
hpmpc train
hpmpc plan              # visa planen utan att skriva något
hpmpc run               # skarp drift
```

Alla kommandon:

| Kommando | Gör |
|---|---|
| `init-config` | skriv en startkonfiguration |
| `check` | verifiera Home Assistant och entiteterna |
| `providers` | verifiera SMHI och SE3-priserna, visa din marginalkostnad |
| `geocode` | slå upp koordinater för en adress |
| `curve` | räkna om Daikins tvåpunktskurva till lutning/offset |
| `pump-table` | visa COP- och kapacitetstabellerna |
| `calibrate-ntc` | anpassa NTC-modellen till dina mätningar |
| `ntc-table` | visa NTC-kurvan och upplösningen per resistanssteg |
| `collect` | hämta historik till ett dataset |
| `excite` | kör identifieringsexperimentet |
| `train` | anpassa modellen |
| `plan` | lös en gång och skriv ingenting |
| `backtest` | spela upp historiken, MPC mot konstant offset |
| `run` | styrslingan |
| `serve` | styrslingan plus lokalt HTTP-API |
| `demo` | hela kedjan mot ett syntetiskt hus |

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
electricity (kWh)            197.6       197.7
cost (SEK)                  162.37      168.45
stored heat (SEK)             1.60       -0.47
net cost (SEK)              160.76      168.92
mean indoor (C)              21.03       20.98
min indoor (C)               20.14       19.41
Kh outside comfort            0.73       11.97

Saving: 8.16 SEK (4.8 %) at equal average indoor temperature
        of which 6.08 SEK on the meter and 2.08 SEK in heat left in the slab
Energy: -0.1 % kWh
```

Två saker att lägga märke till. Det mesta av besparingen syns faktiskt på elmätaren
(6,08 av 8,16 kr) — resten är värme som ligger kvar i plattan och krediteras till vad
den skulle kosta att köpa. Och komforten blir samtidigt **mycket** bättre: 0,7 mot 12,0
kelvintimmar utanför komfortbandet, eftersom ett konstant offset inte kan förutse ett
väderomslag.

Jämförelsen görs mot det **konstanta offset som ger samma medelinnetemperatur**, och
kostnaden räknas netto efter den värme varje körning lämnar kvar i plattan. Att jämföra
brutto mot offset 0 vore ohederligt på två sätt: en del av besparingen skulle bara vara
ett kallare hus, och den som slutar perioden med kall platta skulle belönas för det.

Kör minst sju dygn. På kortare perioder kan terminaltermen bli en stor andel av
besparingen och siffran blir skör.

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
Sex oberoende lager:

1. **Klamring och rampbegränsning** — `offset_min`/`offset_max` och högst
   `max_change_per_cycle` kelvin per cykel.
2. **Hårt komfortband** — under `hard_min` går regulatorn till maximal värme och över
   `hard_max` till minimal, oavsett vad optimeraren tycker.
3. **Sensorvakt** — saknad, orimlig eller för gammal data (`max_data_age_minutes`) gör att
   regulatorn *gradvis* går mot `fallback_offset` och loggar varför. Den fryser aldrig
   fast ett gammalt extremvärde.
4. **Absolut gräns för den falska temperaturen** (`heat_pump.perceived_min_c` /
   `perceived_max_c`). Oberoende av offseten får pumpen aldrig visas ett värde utanför
   sitt normala driftområde. Modellen respekterar samma gräns, så optimeraren planerar
   aldrig något regulatorn sedan måste klippa bort.
5. **Dödmansgrepp i Home Assistant** — om regulatorn slutar rapportera släpper HA
   emulatorn och skickar en notis. Paketet innehåller också en rimlighetsspärr som
   stoppar allt om den visade temperaturen avviker mer än 8 K från den riktiga.
6. **Watchdog i ESP32:n** — släpper emulatorn efter 45 minuter utan kommando, plus en
   egen rimlighetsspärr på resistansen. Det lagret överlever att både Home Assistant
   och hpmpc dör.

Och i hårdvaran: reläet defaultar till den **riktiga** utegivaren när ESP:n är strömlös.
Pumpen ska alltid se en trovärdig givare, särskilt när det här projektet inte kör.

Utöver det finns två *ekonomiska* spärrar som skyddar mot att optimeringen slår fel:
elpatronen är prissatt korrekt i målfunktionen (COP 1,0), och `max_electric_power_kw`
hindrar planen från att stapla kompressor och elpatron ovanpå varandra.

Kör gärna `dry_run: true` (eller `hpmpc plan`) i några dygn först och läs planerna innan
du släpper det skarpt.

---

## Hårdvara: ESP32 och det digitala motståndet

### Vilken givare ska luras? Läs det här först

Det självklara målet är **R1T**, omgivningsgivaren i utedelen. Det är också fel givare.
R1T matar inte bara värmekurvan: aggregatet använder den för avfrostningstiming, för
driftområdesgränser och för att avgöra om det överhuvudtaget får gå. Ljuger man för
R1T ljuger man för allt det där.

Daikins innedel kan istället ta emot en **extern utegivare** (tillbehör KRCS01-1,
monterad utomhus) och genom en field setting använda *den* för den väderkompenserade
kurvan, medan utedelen behåller sin riktiga R1T för avfrostning och skydd. Det är den
givaren som ska emuleras: samma effekt på kurvan, inga sidoeffekter.

Slå upp koden för den field settingen i **din** installatörsmanual — den skiljer mellan
Altherma-generationer, och att gissa på den är inget man gör mot en värmepump. Sök efter
"external outdoor sensor" eller "ambient sensor".

Landar du ändå på R1T: håll `control.offset_min`/`offset_max` små (ungefär ±4 K) och
`heat_pump.perceived_min_c` med god marginal över maskinens −25 °C-gräns, så att lögnen
aldrig når ett tröskelvärde som betyder något.

### Vad som faktiskt krävs av hårdvaran

En Daikin 20 kΩ-utegivare spänner ungefär 32 kΩ vid +15 °C till 197 kΩ vid −20 °C. En
enda 8-bitars 100 kΩ digitalpotentiometer klarar varken räckvidden eller upplösningen.
Det som fungerar:

> **två 10-bitars 100 kΩ digitala reostater i serie** (t.ex. 2× AD5293-100)
> → 0–200 kΩ i steg om ~195 Ω

Kontrollera vad det ger innan du beställer:

```bash
hpmpc ntc-table --step-ohm 195
```
```
  temp (C)         ohm   K per 195.0 ohm
     -20.0      196648             0.015
       0.0       67628             0.049
      15.0       32159             0.110
```

Klart under de 0,2 K där kvantiseringen börjar synas. En 8-bitars 100 kΩ-krets hade gett
runt 0,4 K vid +15 °C och inte nått −20 °C alls.

`ha/esphome_daikin_outdoor_sensor.yaml` är en komplett ESPHome-nod med den topologin,
en oberoende rimlighetsspärr i firmware, och en watchdog som släpper emulatorn efter 45
minuter utan kommando.

**Den riktiga givaren sitter kvar** via ett reläs NC-kontakt. ESP:n måste aktivt dra
reläet för att ta över. Strömavbrott, wifi-tapp, kraschad firmware — pumpen faller
tillbaka på sin egen givare.

### Tre sätt att få ut styrsignalen

| `control.output_mode` | Skriver | Använd när |
|---|---|---|
| `offset` | kelvin | pumpen har en egen offset-parameter |
| `fake_temperature` | °C | din hårdvara tar emot en temperatur |
| `resistance` | ohm | hpmpc räknar om via NTC-kurvan åt dig — **det här är läget för ESP32-uppsättningen** |

För `resistance`: ta hellre tabellen ur pumpens servicemanual (`ntc.model: table`) än
beta-modellen. Riktiga givare följer inte en tvåparametersmodell över hela spannet —
`hpmpc calibrate-ntc` säger till när den märker det.

### Idrifttagning

1. Koppla **inte** in pumpen än. Kommendera några resistanser och mät med multimeter.
   Reostaterna har verklig wiper-resistans och några procents tolerans — mät, anta inte.
2. Mät pumpens egen givare vid kända temperaturer och kör `hpmpc calibrate-ntc`.
3. Koppla in pumpen med `dry_run: true` och offset 0. Pumpens display ska visa den
   riktiga utetemperaturen. Gör den inte det är kalibreringen fel — fixa det först.
4. Kommendera −2 K stadigt i ett dygn och kontrollera att framledningen stiger med
   ungefär `curve_slope × 2` K. Först därefter släpper du regulatorn.

## Vad kan man realistiskt spara?

Backtestet ovan ger 4,8 % lägre kostnad vid samma medelinnetemperatur, och samtidigt
klart bättre komfort. Optimeraren väljer inte bara billiga timmar utan också *varmare*
timmar med bättre COP — hur mycket av vinsten som kommer från vilket beror helt på
prisspridningen just den veckan.

Räkna med mindre i verkligheten: prognoser är inte perfekta och modellen är inte huset.
Faktorer som avgör hur mycket du får ut:

- **Prisspridningen.** Rörligt avtal med stor dygnsvariation ger mycket; fast pris ger
  nästan ingenting (då blir vinsten enbart COP-optimering, ett par procent).
- **Bredden på komfortbandet.** Med `comfort_min` 20,3 och `comfort_max` 22,0 finns det
  1,7 K att arbeta med. Krymper du bandet krymper vinsten proportionellt.
- **Betongplattans massa.** Tunn platta på träbjälklag är ett mycket sämre värmelager.
  Kolla `time_constants_hours.slab` efter träning: under ~5 h finns lite att hämta.
- **Sätt `price_addition` och `price_vat_pct`.** Elöverföring och energiskatt är typiskt
  0,50–0,80 kr/kWh i Sverige. De ingår i marginalkostnaden men dämpar den *relativa*
  spridningen — utan dem överskattar optimeraren vinsten av lastförflyttning.
- **Elpatronen.** Om `hpmpc train` rapporterar många timmar med tillsatsvärme, eller om
  planen ständigt vill ha den, är det inte optimeraren som är fel: kurvan, offsetgränserna
  eller dimensioneringen pressar systemet förbi vad kompressorn klarar. Med COP 1,0 äter
  några timmar elpatron upp en hel veckas lastförflyttning.

---

## Begränsningar

Saker som är värda att veta innan du litar på det här:

- **Parametrarna är inte unikt bestämda.** Prediktionen kan vara utmärkt medan enskilda
  parametrar är långt från sanningen. Därför rapporteras `identifiability` — läs den.
- **Prestandakartan är förankrad, inte hämtad.** Siffrorna kommer från publicerade
  mätpunkter för maskinklassen, inte ur Daikins databook — den gick inte att hämta från
  miljön där koden skrevs. Formen är rimlig och nivån kalibreras mot din elmätare, men
  byt tabellen mot den riktiga när du har den.
- **Pumpen modelleras kontinuerligt.** Verkliga pumpar taktar, har startspärrar och
  prioriterar varmvatten. Avfrostning, kapacitetsgräns och elpatron finns i modellen;
  taktning och varmvattenprioritering gör det inte.
- **Varmvatten ingår inte.** EHVH16S26 har 260 liter och pausar värmedriften under
  varmvattenberedning. Modellen ser inte den pausen. Vill du lastförflytta varmvattnet
  är det ett separat problem — men prisprognosen härifrån går att återanvända.
- **Field settings i pumpen rörs inte.** Systemet ändrar bara vad givaren visar. Vilken
  givare Daikin faktiskt lyssnar på för kurvan är något du måste ställa in själv, med
  installatörsmanualen i handen.
- **Elprisprognosen slutar vid morgondagens sista timme**, och före 13:00 vid dagens.
  Bortom det extrapoleras sista kända priset; horisontens sista timmar är alltså en
  kvalificerad gissning. `price_extrapolated_hours` i rapporten säger hur mycket.
- **SMHI och elprisetjustnu är externa tjänster.** De är gratis och nyckelfria, men de
  kan gå ner. Prognoser cachas och Home Assistant är reservväg för båda; ingenting
  däremot är AI-tjänster, och ingen husdata lämnar nätverket.
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
├── providers/
│   ├── smhi.py       SMHI punktprognos
│   ├── elpris.py     SE3 spotpris (Nord Pool via elprisetjustnu)
│   └── geocode.py    engångsuppslag av adress
├── model/
│   ├── performance.py COP-/kapacitetskarta, elpatron, avfrostning
│   ├── heatpump.py   värmekurva, utegivarfilter, effektberäkning
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
