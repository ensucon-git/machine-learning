# Givaren och potentiometern

Hur ställdonet byggs, och de två sakerna som görs sällan men måste göras rätt:
kalibrera NTC-tabellen mot pumpen, och gå över till två potentiometrar i serie
när räckvidden tar slut.

Läs `INSTALL.md` först — de numrerade avsnitten förutsätter att systemet redan
kör. Ska du bygga kortet är det [Kortet](#kortet) som gäller.

---

## Innehåll

- [Varför det här spelar roll](#varför-det-här-spelar-roll)
- [Kortet](#kortet)
- [1. Kalibrera NTC-tabellen](#1-kalibrera-ntc-tabellen)
- [2. Gå över till två MCP41100 i serie](#2-gå-över-till-två-mcp41100-i-serie)
- [Två tabeller som måste vara överens](#två-tabeller-som-måste-vara-överens)

---

## Varför det här spelar roll

Pumpen har ingen egen utegivare kvar. Potentiometern **är** givaren, och hela
kedjan är öppen slinga: hpmpc kommenderar en resistans och litar på att tabellen
säger vad den betyder.

Två saker kan gå fel, och de är olika allvarliga.

**Ett jämnt fel i tabellen absorberas.** `hpmpc train` anpassar värmekurvan
genom att regressera uppmätt framledning mot *kommenderad* offset. En konstant
förskjutning eller ett skalfel hamnar i `curve_offset`/`curve_slope`, och
styrningen predikterar ändå rätt framledning. Loopen sluts i anpassningen.

**Det som inte absorberas är allt som hänger på ett absolut tröskelvärde:**
pumpens `heat_stop_temp`, `perceived_min_c`/`perceived_max_c`, och semesterläget
— som fungerar just genom att passera värmestoppet. Där betyder två graders fel
att en inställning gör något annat än den säger.

Så: kalibrera, men få inte panik över en halv grad.

---

## Kortet

Ställdonet är ett hålmatriskort med en ESP32-C3, två digitalpotentiometrar i
serie och en nivåomvandlare. Det driver pumpens givaringång, och läser samtidigt
husets riktiga utegivare på en ADC-kanal.

`ha/esphome_daikin_outdoor_sensor.yaml` är firmware för exakt den här
uppsättningen, och dess kommentarer bär idrifttagningsordningen.

### Vad som sitter på det

| Bet. | Komponent | Roll |
|---|---|---|
| A1 | Waveshare ESP32-C3-Zero-M | 18 stift, 9 per rad |
| U1, U2 | MCP41100, 8-pol DIP | i serie → 0–200 kΩ mot pumpen |
| U3 | **74HCT125**, 14-pol DIP | 3,3 V → 5 V på SCK, SI, CS1 och CS2 |
| R1 | 20 kΩ metallfilm 1 % | referensmotstånd i givardelaren |
| R2 | 1 kΩ | serieskydd vid ADC-stiftet, värdet okritiskt |
| R3, R4 | 4,7 kΩ | pull-up på CS1 och CS2, på buffertens 3,3 V-sida |
| R7, R8 | 10 kΩ | pull-down på SCK och SI, samma sida |
| C1–C4 | 100 nF X7R | avkoppling, en per krets plus en vid ADC-stiftet |
| J1, J2, J3 | skruvplint 4-pol | givaren, pumpen, matningen |

### Två rälar, och varför

Pumpens givarterminaler mäter **4,97 V i tomgång** — det är rälen bakom dess
pull-up, och därmed taket för vad potentiometerns terminaler kan nå när banan
står högt. MCP41100:ans terminaler får aldrig gå över dess egen VDD, så:

- **U1, U2 och U3 matas med 5 V**, taget direkt från J3.1. Inte från modulens
  `5V`-stift — en del Zero-kort har en diod från USB och levererar 4,6 V, vilket
  ligger under pumpens räl. Felet skulle bara märkas i de kallaste lägena.
- **Delaren och modulen går på 3,3 V** från modulens egen regulator.

Vid VDD = 5 V kräver MCP41100 3,5 V för att garanterat läsa en etta
(0,7 × VDD), och C3:an ger 3,3 V. Därför är **74HCT125 inte valfri**: TTL-ingångar
tar allt över 2,0 V och CMOS-utgångarna svänger till 5 V. Det måste stå **HCT**
eller **AHCT** — en 74HC125 har samma 0,7 × VDD-tröskel och löser ingenting.

Åt andra hållet kommer inga 5 V tillbaka: MCP41100 saknar utgång, så bussen är
enkelriktad.

> **Innan du bonderar jordarna:** kortets GND kopplas ihop med pumpens
> givarretur, annars flyter potentiometrarnas terminaler. Mät först spänningen
> mellan pumpens båda givarterminaler och skyddsjord, och mata ESP:n från en
> **tvåpolig, ojordad** adapter — inte från en jordad laptop.

### Stiftvalet

Pinnarna är valda efter var de *sitter*, inte bara efter vad de kan. Fyra i
varje rad går bort, och ingen av dem är självklar.

| Stift | Namn | Nät | Varför |
|---|---|---|---|
| 1 | 5V | `+5V` | matning in från J3.1 |
| 2 | GND | `GND` | jordbussens startpunkt |
| 3 | 3V3 (OUT) | `+3V3` | delaren och pull-upparna |
| 4 | GP0 | `ADC_IN` | ADC1_CH0, **och granne med 3V3 och GND** |
| 5 | GP1 | `RELAY_DRV` | reserverad för steg 2 |
| 6 | GP2 | — | strapping, måste ligga högt vid boot |
| 7 | GP3 | — | ledig reserv |
| 8 | GP4 | `CS1` | nederst i vänsterraden |
| 9 | GP5 | `CS2` | granne med CS1 |
| 10 | GP6 | `SPI_CLK` | nederst i högerraden |
| 11 | GP7 | `SPI_SI` | granne med CLK |
| 12 | GP8 | — | strapping, måste ligga högt vid boot |
| 13 | GP9 | — | **BOOT-knappen sitter här** |
| 14 | GP10 | — | kortets WS2812-lysdiod |
| 15–16 | GP18, GP19 | — | USB DM och DP |
| 17–18 | GP20, GP21 | — | UART0 |

Att just GP0 blev ADC-ingång är ingen slump: den ligger direkt under 3V3 och
GND, så R1, R2 och C3 får plats på tre intilliggande hålrader utan att någon
analog tråd korsar kortet.

### Delaren åt rätt håll

```
3V3 ── J1.1 ══ utegivaren ══ J1.2 ──┬──[ R2 1k ]──┬── GP0
                                    │             │
                               [ R1 20k ]     [ C3 100n ]
                                    │             │
                                   GND           GND
```

Vändningen avgör om det går att mäta alls. Med motståndet uppåt mot 3V3 och
givaren mot jord hamnar hela vinterhalvåret över 2,5 V, där C3:ans ADC är
komprimerad och mättar. Så här ger kyla **låg** spänning:

| ute | givaren | ADC | känslighet |
|---|---|---|---|
| −30 °C | 348 kΩ | 0,18 V | 8,8 mV/K |
| −20 °C | 197 kΩ | 0,30 V | 16 mV/K |
| 0 °C | 67,6 kΩ | 0,75 V | 30 mV/K |
| +20 °C | 25,4 kΩ | 1,46 V | 39 mV/K |
| +30 °C | 16,0 kΩ | 1,83 V | 42 mV/K |

Den dominerande felkällan är inte upplösningen utan ADC:ns absoluta offset och
förstärkning, ett par procent per chip. Det felet är **inte** självläkande som
NTC-tabellens är — kurvanpassningen absorberar det inte, för den här avläsningen
går inte till pumpen. Trimma bort det en gång med `calibrate_linear` mot en
referenstermometer.

Givaren blir därmed en riktig `entities.outdoor_temp` — en mätning vid huset i
stället för SMHI:s rutpunkt. Räkna med ett steg i träningsdatan när du byter, och
kör `hpmpc train` igen när du har några veckor med den.

### Nätlista

Den auktoritativa listan. Kontinuitetsmät varje rad med tomma hållare och
modulen ur, innan något sätts i.

| Nät | Antal | Förbindelser |
|---|---|---|
| `GND` | 18 | J3.2 · A1 stift 2 · U1 stift 4 · U2 stift 4 · **U2 stift 7 (PB0)** · U3 stift 1, 4, 7, 10, 13 · J2.2 · R1 nedre · R7 nedre · R8 nedre · C1–C4 jordsida |
| `+5V` | 8 | **J3.1** · A1 stift 1 · U1 stift 8 · U2 stift 8 · U3 stift 14 · C1, C2, C4 |
| `+3V3` | 4 | A1 stift 3 · J1.1 (via L1) · R3 övre · R4 övre |
| `NTC_SENSE` | 3 | J1.2 (via L2) · R1 övre · R2 |
| `ADC_IN` | 3 | R2 · C3 signalsida · A1 stift 4 |
| `CS1` | 3 | A1 stift 8 · U3 stift 9 · R3 nedre |
| `CS1_5V` | 2 | U3 stift 8 · U1 stift 1 |
| `CS2` | 3 | A1 stift 9 · U3 stift 12 · R4 nedre |
| `CS2_5V` | 2 | U3 stift 11 · U2 stift 1 |
| `SPI_CLK` | 3 | A1 stift 10 · U3 stift 2 · R7 övre |
| `SPI_CLK_5V` | 3 | U3 stift 3 · U1 stift 2 · U2 stift 2 |
| `SPI_SI` | 3 | A1 stift 11 · U3 stift 5 · R8 övre |
| `SPI_SI_5V` | 3 | U3 stift 6 · U1 stift 3 · U2 stift 3 |
| `POT_HI` | 3 | U1 stift 6 (PW0) · U1 stift 5 (PA0) · J2.1 (via L3) |
| `POT_MID` | 3 | U1 stift 7 (PB0) · U2 stift 6 (PW0) · U2 stift 5 (PA0) |

**Alla fyra pull-motstånden sitter på buffertens 3,3 V-sida**, alltså mellan
modulen och U3 — inte på potentiometrarnas chip-select-stift. R3 och R4 håller
`CS1` och `CS2` höga medan modulen bootar och dess stift ännu är högimpedanta;
R7 och R8 gör samma sak nedåt för klocka och data. Sätter man dem efter
bufferten motarbetar de i stället dess utgång.

**U2 stift 7 är PB0**, inte en matningsanslutning — det är potentiometerkedjans
nedre ände och ska ligga på jord. Lätt att missa i en rad av VSS-stift.

`74AHCT125N` fungerar lika bra som `74HCT125N`: AHCT har samma TTL-ingångar,
allt över 2,0 V räknas som hög. Bara en ren `74HC125` vore fel. Enable är
aktiv-låg, så alla fyra `OE` går till GND.

### 74AHCT125, 14-polig DIP

| Stift | Namn | Nät |
|---|---|---|
| 1 | 1OE | `GND` |
| 2 | 1A · in | `SPI_CLK` |
| 3 | 1Y · ut | `SPI_CLK_5V` |
| 4 | 2OE | `GND` |
| 5 | 2A · in | `SPI_SI` |
| 6 | 2Y · ut | `SPI_SI_5V` |
| 7 | GND | `GND` |
| 8 | 3Y · ut | `CS1_5V` |
| 9 | 3A · in | `CS1` |
| 10 | 3OE | `GND` |
| 11 | 4Y · ut | `CS2_5V` |
| 12 | 4A · in | `CS2` |
| 13 | 4OE | `GND` |
| 14 | VCC | `+5V`, med C4 över stift 14 och 7 |

Kanal 3 och 4 är spegelvända: på högra sidan kommer *utgången* före ingången.
Stift 8 är alltså en utgång, stift 9 en ingång.

### Lägg modulen på tvären

Pinouten avgör layouten. Upprätt hamnar matningen och alla ADC-stift i ena raden
och SPI-stiften i den andra, så signalerna måste korsa kortet. **Lagd på sidan
med USB-C mot kortkanten** hamnar matning och analog klunga i modulens ena ände
och alla fyra digitala stiften i den andra.

På ett 70 × 50 mm kort:

- **J1 och J2 diagonalt** — nedre vänstra och övre högra hörnet. Det är två
  likadana tvåledarkablar och den ena får inte hamna där den andra ska sitta.
  Märk plintarna med penna medan du löder.
- **Analog klunga** (R1, R2, C3) vid modulens stift 2–4.
- **U3, U1, U2** till höger, nära stift 8–11.
- **Radavståndet mellan modulens stiftrader** är troligen 6 hål — mät, och löd
  hylslisterna med modulen isatt så de blir parallella. Det är det enda måttet
  som inte går att rätta efteråt.

Lödningen: **blank förtennad tråd på ovansidan** för `GND` och `+5V`, som är
näten med flest anslutningar; **isolerad enkeltrådig tråd på undersidan** för
signalerna, som korsar varandra; **avklippta komponentben** för hopp på två–tre
hål. Kedjor av hopklickade lödöar drar mycket tenn, ger kalla fogar i mitten och
går inte att ändra — två öar ihop går bra, tio gör det inte.

### Tre trådbyglar som håller dörren öppen

Löd de här tre som **lösa byglar i avvikande färg**, inte som fasta trådar:

| Bygel | Från | Till |
|---|---|---|
| L1 | `+3V3` | J1.1 — givarens tråd A |
| L2 | J1.2 — givarens tråd B | `NTC_SENSE`, mätnoden |
| L3 | `POT_HI`, U1 stift 5+6 | J2.1 |

Det är exakt de tre som reläväxlingen i steg 2 tar över. Klipp tre, koppla in
sex, och ingenting annat på kortet rörs.

### Steg 2: reläväxlingen, om den behövs

Säkerhetstrappan i ESPHome-filen har fem lager. Lager 1–4 klarar sig utan extra
hårdvara — de förutsätter att noden lever. Bara det femte, *ESP:n strömlös medan
pumpen går*, behöver ett relä.

Utan relä: potentiometrarnas motståndsbanor är passiva och PA0 är byglad till
PW0 på varje krets, så pumpen ser omkring 200 kΩ ≈ −20 °C. Fel, men en avläsning
och inte ett givarfel. **Verifiera det vid idrifttagningen** — dra kortets
matning och mät över J2. Blir det oändligt är byglarna fel dragna.

Tre poler växlar i steg 2, och alla tre behövs:

| Pol | Gemensam | Draget — normalt | Släppt — bypass |
|---|---|---|---|
| 1 | givarens tråd A | `+3V3` | pumpens terminal A |
| 2 | givarens tråd B | `NTC_SENSE` | `GND` |
| 3 | pumpens terminal A | `POT_HI` | bruten |

Utan pol 1 hamnar +3V3 rakt på pumpens givaringång. Utan pol 2 ser pumpen
givaren i serie med R1, och pumpens ström hittar dessutom en väg genom R2 in i
den döda modulens ESD-diod — en ~1 kΩ shunt som fäller avläsningen. Utan pol 3
hänger digipotarna kvar parallellt, vilket blir −7 °C när det är −20 ute.

Vinsten är att pumpen då får **sanningen**, inte en mindre lögn. En oktoberdag
vid +12 ute, med exempelkurvan `hpmpc curve --point=-15:40 --point=15:25`:

| | pumpen visas | framledning |
|---|---|---|
| inget relä | −20 °C | 42,5 °C |
| relä + fast 68 kΩ | 0 °C | 32,5 °C |
| relä + riktiga givaren | +12 °C | 26,5 °C |

Delar: **två DPDT-signalreläer med guldpläterade bifurkerade kontakter** och
5 V-spole (Omron G6K-2F-Y; kontrollera *single side stable*, inte latching),
en 2N7000, en 1N4148 och två motstånd. Ett vanligt effektrelä duger inte — det
här är en torrkrets på tiotals mikroampere, och silverkontakter bygger oxidfilm
vid de nivåerna och blir glappande. Driv **båda spolarna från samma drivsteg**,
annars finns ett läge där den ena polen växlat och den andra inte.

Tills dess är larmet ersättningen: `binary_sensor.varmepump_proxy_online` går
`off` så fort noden slutar svara. Just det fel reläet finns för — kortets
matning dör medan pumpen går — är per definition ett där Home Assistant och
nätet är uppe. Sätt gärna ESP:ns adapter på samma säkring som pumpen också.

---

## 1. Kalibrera NTC-tabellen

### Kalibrera mot pumpen, inte mot givaren

Det naturliga är att mäta termistorn med multimeter. Gör inte det.

Ett par avläst som *"jag skickade R ohm, pumpens display sa T grader"*
innefattar kabelresistansen, kontakten och pumpens egen linjärisering. En
bänkmätning av termistorn missar allt det, och det är summan som avgör vad
pumpen tror.

### Vad du behöver

Minst **två** par vid väl åtskilda temperaturer. Fyra är bättre — en
tvåparametersmodell (R25 och B) driver iväg i ändarna av intervallet, och med
fyra punkter kan du se om den gör det.

### Steg för steg

**1. Sätt ett känt värde manuellt.**

Slå av MPC-styrningen först så att ingen skriver emot dig:

```
input_boolean.varmepump_mpc_aktiv  →  av
```

Skriv sedan direkt till noden, i Home Assistant under
**Utvecklarverktyg → Tillstånd**, eller från instrumentpanelen:

```
number.varmepump_proxy_simulerad_utetemperatur  =  10
```

**2. Vänta en timme.**

Pumpen filtrerar sin utegivare — `heat_pump.outdoor_filter_hours` är 3 timmar i
modellen, och pumpen gör något liknande. Displayen hinner inte med direkt. Läs av
först när den slutat röra sig.

**3. Läs pumpens display, och hämta ohm-talet.**

```bash
docker compose exec hpmpc hpmpc check
```

```
Actuator
  should be showing   +10.00 C   (input_number.varmepump_fiktiv_utetemp)
  ESP32 is driving    wiper 104 = 40884 ohm = +10.01 C

  -> read the pump's display now, and that reading pairs with 40884 ohm:
       hpmpc calibrate-ntc --point=<display>:40884 --point=...
     Give the pump an hour to settle first - it filters its outdoor reading.
```

Säg att displayen visar **9,4**. Ditt par är `9.4:40884`.

Ohm-talet kommer från wiperavläsningen genom `pot:`-sektionens geometri, inte
genom NTC-kurvan — så det är giltigt även när ESP32:n äger tabellen. Har du inte
mätt potentiometern med multimeter ännu är det värt att göra det först, se
[avsnitt 2](#2-gå-över-till-två-mcp41100-i-serie).

**4. Upprepa vid tre temperaturer till.**

Sprid dem. `-5`, `0`, `+10`, `+20` täcker det du faktiskt kör i. Varje punkt tar
en timme, så räkna med en förmiddag.

**5. Anpassa.**

```bash
docker compose exec hpmpc hpmpc calibrate-ntc \
  --point=-5.4:87562 --point=0.3:67628 --point=9.4:40884 --point=19.6:25354
```

```
Fitted: R25 = 19897 ohm, B = 3940

  measured T  measured R     model R     error
        -5.4       87562       89220      1.9%
         0.3       67628       65650     -2.9%
         9.4       40884       41275      1.0%
        19.6       25354       25390      0.1%

Worst temperature error at the measured points: 0.56 K
A two-parameter beta model drifts at the ends of the range. With four or more
measurements, or the table from the service manual, prefer ntc.model: table.

Paste into config.yaml:

ntc:
  model: beta
  r25: 19897
  beta: 3940
```

**Negativa temperaturer kräver likhetstecken:** `--point=-5.4:87562`, inte
`--point -5.4:87562`. Annars tolkar argparse minustecknet som en flagga.

**6. Läs anpassningen innan du klistrar in den.**

Exemplet ovan är avsiktligt inte perfekt. Värsta felet är 0,56 K och felen byter
tecken över intervallet — betamodellen böjer sig inte riktigt som givaren gör.
Det är kommandot som säger till om, och rådet är rätt: har du fyra punkter eller
servicemanualens tabell är `model: table` bättre än en anpassad betamodell.

Med en tabell skriver du in punkterna direkt:

```yaml
ntc:
  model: table
  table_temp_c: [-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]
  table_ohm:    [347667, 260639, 196648, 149283, 114003, 87562, 67628, 52514, 40991, 32159, 25354, 20084, 15984]
```

Ligger dina uppmätta par nära den tabell som redan levereras behöver du inte
göra någonting — det är signalen att den stämmer för din givare.

**7. Uppdatera ESP32:ns tabell också** — se
[Två tabeller som måste vara överens](#två-tabeller-som-måste-vara-överens).

**8. Verifiera.**

Sätt offset 0 och kontrollera att pumpens display visar den riktiga
utetemperaturen. Gör den inte det är kalibreringen fortfarande fel.

---

## 2. Gå över till två MCP41100 i serie

### Varför

En MCP41100 spänner 0–100 kΩ, vilket på en Daikin 20 kΩ-kurva tar slut vid
**−7,4 °C**. Kallare än så finns ingen wiperposition:

```bash
docker compose exec hpmpc hpmpc ntc-table --low -20 --high 20
```

```
Pot:       1 x mcp41100 in series, 256 positions, 392 ohm per step
           reaches -7.4 to +30.0 degC

  temp (C)         ohm   wiper     K per 392 ohm
     -20.0      196648      --                    <- out of the pot's range
     -10.0      114003      --                    <- out of the pot's range
      -5.0       87562     223             0.074
       0.0       67628     172             0.098
```

**Upplösningen är inte problemet** — 0,10 K per steg kring nollan. Det är
räckvidden. Två i serie ger 0–200 kΩ och når −20,3 °C **med samma steglängd**:
seriekopplade kretsar köper räckvidd, inte upplösning.

Köp alltså ännu en **MCP41100** — samma krets, 100 **kΩ** 8-bitars
digitalpotentiometer med SPI. Inte en fast resistor.

### Koppla in

Wiper och ena änden på varje krets kopplas så att de två banorna adderas. Den
andra kretsen får ett eget chip-select-ben. `ha/esphome_daikin_outdoor_sensor.yaml`
har platsen förberedd:

```yaml
spi_device:
  - id: pot_primary
    cs_pin: GPIO5
  - id: pot_secondary      # avkommentera
    cs_pin: GPIO17
```

och spill-over-logiken i `set_wiper`: den första fylls till 255, resten går till
den andra.

### Mät innan du litar på den

MCP41100 är en ±20 %-krets. Koppla bort pumpen, kommendera några wiperpositioner
och mät med multimeter. Skriv in det du mätte:

```bash
docker compose exec hpmpc hpmpc set pot.devices 2
docker compose exec hpmpc hpmpc set pot.resistance_ohm 98400    # per krets, uppmätt
docker compose exec hpmpc hpmpc set pot.wiper_ohm 112           # uppmätt
docker compose exec hpmpc hpmpc set heat_pump.perceived_min_c -20
```

Kontrollera att räckvidden blev den du väntade dig:

```bash
docker compose exec hpmpc hpmpc ntc-table --low -20 --high 20
```

```
Pot:       2 x mcp41100 in series, 511 positions, 392 ohm per step
           reaches -20.3 to +30.0 degC

  temp (C)         ohm   wiper     K per 392 ohm
     -20.0      196648     501             0.031
     -10.0      114003     290             0.056
       0.0       67628     172             0.098
```

Ingen `WARNING` om `perceived_min_c` betyder att inställningen och hårdvaran är
överens.

### Tre ställen till som måste följa med

Antalet kretsar står på fler ställen än i `pot:`, och de sitter i olika filer.
Missar du något av dem märks det inte förrän mitt i vintern.

| var | vad |
|---|---|
| `config/config.yaml` | `pot.devices: 2` |
| `ha/esphome_daikin_outdoor_sensor.yaml` | `DEVICES` i de två lambdorna, och `STEP_MAX` i `set_wiper` |
| `ha/packages/heatpump_mpc.yaml` | `{% set pots = 1 %}` i mallen `Utegivare wiper` |
| samma fil | automationen `MPC potentiometer at end stop` triggar på `"255"` — med två kretsar är ändläget 510, och 255 passeras varje gång wipern går förbi mitten |

Den medskickade ESPHome-filen står redan på två kretsar. De två i HA-paketet
gör det inte, eftersom exempelkonfigurationen fortfarande levereras med
`devices: 1`.

### Låt hpmpc äga wiperkurvan

Med räckvidden på plats är det värt att flytta hela omräkningen till hpmpc. Då
finns bara **en** tabell, och ställdonskontrollen i `hpmpc check` blir meningsfull
i stället för att jämföra två tabeller mot varandra.

**1. Konfigurera wiperutgången.** I `config/config.yaml`:

```yaml
entities:
  offset_output: input_number.varmepump_offset
  fake_temperature_output: input_number.varmepump_fiktiv_utetemp
  wiper_output: input_number.varmepump_wiper           # ny hjälpare, se nedan
```

Behåll `fake_temperature_output` — den är läsbar på en instrumentpanel och det
är den dödmansgreppet skriver till.

**2. Lägg till hjälparen** i `ha/packages/heatpump_mpc.yaml`:

```yaml
input_number:
  varmepump_wiper:
    name: MPC wiper target
    min: 0
    max: 511          # 255 per krets
    step: 1
    mode: box
```

**3. Byt automation.** I samma fil: kommentera bort `MPC to sensor emulator` (A)
och avkommentera `MPC to digital resistor` (B), men låt den läsa den nya
hjälparen i stället för mallen:

```yaml
  - alias: "MPC to digital resistor"
    id: hpmpc_to_resistor
    mode: single
    trigger:
      - platform: state
        entity_id: input_number.varmepump_wiper
    condition:
      - condition: state
        entity_id: input_boolean.varmepump_mpc_aktiv
        state: "on"
    action:
      - service: number.set_value
        target:
          entity_id: number.varmepump_proxy_wiper_target
        data:
          value: "{{ states('input_number.varmepump_wiper') | int }}"
```

**4. Exponera wiperingången på noden.** ESPHome-filen har den redan — `Wiper
target` → `number.varmepump_proxy_wiper_target`. Flasha om noden.

**5. Verifiera i torrläge.**

```bash
docker compose exec hpmpc hpmpc plan
```

```
Offset now: -1.50 K   [mpc]
    input_number.varmepump_offset                        -1.5 K
    input_number.varmepump_fiktiv_utetemp                -8.5 degC
    input_number.varmepump_wiper                          270 step
```

Kontrollera att wipertalet stämmer med `hpmpc ntc-table` för den temperaturen
innan du slår på `input_boolean.varmepump_mpc_aktiv`.

### Sänk inte offsetgränserna för snabbt

Med räckvidd ner till −20 °C kan optimeraren plötsligt be om mycket mer värme än
förut. `control.offset_min` är fortfarande den styrande gränsen — låt den ligga
kvar på −6 en vecka och titta på planerna innan du vidgar den.

---

## Två tabeller som måste vara överens

Så länge ESP32:n äger omräkningen finns NTC-tabellen på **två** ställen:

| var | vad den används till |
|---|---|
| `ntc:` i `config/config.yaml` | hpmpc:s egen omräkning och räckviddsvarningar |
| tabellen i `esphome_daikin_outdoor_sensor.yaml` | vad noden faktiskt driver |

De levereras identiska — samma 13 punkter. **Ändrar du den ena måste du ändra
den andra.**

Gör du inte det syns det i `hpmpc check` som ett stående ställdonsfel:

```
Actuator
  should be showing   +15.70 C
  ESP32 is driving    wiper 66 = 25982 ohm = +19.54 C
  wiper commanded     79   (error +3.84 K)

  Note: this checks the chain as far as the ESP32 only. ... The wiper is read
  back through hpmpc's own ntc:/pot: sections, so if the node owns the
  conversion instead, a standing error here means those two tables disagree -
  not that anything is failing to arrive.
```

Ett konstant fel som inte ändrar sig med temperaturen betyder att tabellerna är
oense. Ett fel som *växer* mot ändarna av intervallet betyder att en av dem har
fel form — då är det kalibrering som behövs, inte synkronisering.

Går du över till wiper-vägen i avsnitt 2 försvinner problemet: då finns tabellen
bara i `config.yaml`, och ställdonskontrollen jämför hpmpc mot verkligheten i
stället för mot sig själv.
