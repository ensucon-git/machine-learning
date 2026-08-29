# Givaren och potentiometern

Två saker som görs sällan men måste göras rätt: kalibrera NTC-tabellen mot
pumpen, och gå över till två potentiometrar i serie när räckvidden tar slut.

Läs `INSTALL.md` först — det här förutsätter att systemet redan kör.

---

## Innehåll

- [Varför det här spelar roll](#varför-det-här-spelar-roll)
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
