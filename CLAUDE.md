# hpmpc — projektminne

Självhostad modellprediktiv styrning (MPC) av en luft/vatten-värmepump på
golvvärme. Systemet lär sig huset från Home Assistants historik och styr pumpen
genom att manipulera vilken utetemperatur den *tror* att den ser.

Läs `README.md` för hur det fungerar, `INSTALL.md` för hur det installeras och
`docs/HARDWARE.md` för NTC-kalibrering och bytet till två potentiometrar.
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
| Ställdon | **ESP32-C3-Zero-M** + **2× MCP41100** i serie (8 bitar, 100 kΩ styck, SPI) + **74HCT125** som nivåomvandlare, på hydroboxens **KRCS01-1-ingång** — inte utedelens R1T. Bygget: `docs/HARDWARE.md#kortet` |
| Effektmätning | Victron, **hela husets effekt per fas** — ingen mätare enbart på pumpen |
| Elbilsladdare | 11 kW över alla tre faser, `binary_sensor.eh6nh5cd_charging` (`Charging` / `Not charging`) |
| Körs på | NUC, Docker (Portainer), skilt från Home Assistant |
| Utegivare | **ingen** — utetemperaturen hämtas via `weather.smhi_home` i Home Assistant (direkt-SMHI gav 404 hos användaren). `entities.outdoor_temp` finns kvar för att koppla in en givare senare |

Entiteterna, som de faktiskt heter:

| roll | entity_id |
|---|---|
| innetemperatur (mitt i huset) | `sensor.hall_temperature_2` |
| husets effekt per fas | `sensor.gx_device_consumption_power_l1` / `_l2` / `_l3` |
| laddstatus | `binary_sensor.eh6nh5cd_charging` |
| ESP32:ns ingång (grader) | `number.varmepump_proxy_simulerad_utetemperatur` — noden äger NTC-tabellen |
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

**Kontrollern startar utan modell** (`load_model_if_trained`, läget `collecting`).
En färsk installation har ingen modell, och att vägra starta vore bakvänt: pumpen
har ingen annan givare än den vi driver, och historiken anpassningen behöver
samlas in av samma slinga. `adopt_model` växlar över utan omstart så fort
`hpmpc train` körts.

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

**Verifierat** (364 tester, syntetiskt hus med känd sanning):
- Identifieringen återfinner värmekurva (lutning 0,3495 mot 0,35, R² 0,997) och
  husparametrar (UA +3 %, Ci +4 %, `k_wind` +3 %, plattans tidskonstant inom 8 %).
  `hpmpc train` skriver ut vad den lärt sig om sol och vind som area och
  procent per m/s, inte som råa koefficienter.
- Prediktionsfel 0,085 °C över 12 h, 0,066 °C över 48 h (persistensbaslinje 1,01 °C).
- Sluten loop: 5,3 % lägre kostnad vid samma medelinnetemperatur, komfort
  0,8 mot 11,0 kelvintimmar utanför bandet.
- Effektuppdelningen: verkningsgradsskala inom 2 % på 30 dygn, laddaren skattas
  till 11,2 kW mot nominella 11.
- Hemkomstplaneringen härleder framförhållningen: 20 h → startar direkt,
  30 h → väntar till t+16 h.
- **Potentiometerkedjan är uppmätt på det byggda kortet** (2026-09): wiper 123 Ω
  per krets, banorna 96,45 och 98,30 kΩ, 381,9 Ω per steg, 195,0 kΩ i ändläget
  = **−19,83 °C**. Fem mätpunkter, två av dem kontroller som föll inom 0,2 % av
  modellen — vilket samtidigt bevisar båda chip select, bufferten och SPI-vägen.
  Siffrorna står i `docs/HARDWARE.md#mät-innan-du-litar-på-den` och i de två
  lambdorna i ESPHome-filen. `perceived_min_c` ska vara **−19,5**, inte −20:
  −20,3 är det nominella talet för 2 × 100 kΩ, inte det här kortet.
- **Hela ställdonskedjan är verifierad mot pumpen** (2026-09). Kortet byggt,
  firmware flashad och körd, givaren på J1, pumpen på J2: när ESP32:n får ström
  bootar noden till `BOOT_WIPER = 172` = 65 928 Ω, och **pumpens display visar
  0,0 °C**. Vår tabell säger +0,56 °C för det motståndet, alltså −0,56 K
  avvikelse. Det bevisar ESP32 → 74AHCT125 → 2× MCP41100 → pumpens givaringång →
  pumpens egen linjärisering, hela vägen. Det är också den **första riktiga
  kalibreringspunkten**: 65 928 Ω ↔ 0,0 °C på displayen. `hpmpc calibrate-ntc`
  behöver två, så kommendera en till i den varma änden (wiper 52 ≈ 20 103 Ω,
  förväntat +25 °C) — den änden får pumpen att sluta värma i stället för att
  börja.
- **NTC-kurvan är uppmätt mot pumpens display** (2026-09, tio punkter från
  wiper 502 till 52). Interpolationen återger varje punkt på **0,000 K**. De tre
  yttersta värdena (−30, −25, +30) är extrapolerade: betaanpassningens *form*,
  förankrad i den yttersta uppmätta punkten så skarven blir kontinuerlig.
  En ren betamodell prövades och förkastades — värsta fel 0,50 K med U-formade
  residualer, alltså precis vad en tvåparameters-Arrhenius gör mot en givare med
  verklig Steinhart-Hart-krökning. Den typiska Daikintabellen den ersätter låg
  0,92 K fel i den kalla änden. **Räckvidden blev −20,43 °C**, inte −19,83, så
  `perceived_min_c` är nu **−20,0** med 0,4 K marginal mot den extrapolerade
  änden. Kurvan finns på **fem** ställen: `ntc:`, de två lambdorna i ESPHome-
  filen, nodens egen `calibration:` för J1-givaren, och `r25`/`beta` i mallen
  `Utegivare mAlresistans` (den sista som betaanpassning, 0,5 K sämre — den är
  inte i styrvägen).

**Antaget / ej verifierat:**
- **Prestandakartans siffror är förankrade i publicerade mätpunkter för
  maskinklassen, inte hämtade ur Daikins databook** — den gick inte att nå från
  utvecklingsmiljön. Formen är rimlig, nivån kalibreras mot elmätaren. Byt
  tabellen när databooken finns.
- **SMHI och elprisetjustnu har aldrig anropats live** — utgående trafik dit var
  blockerad av sessionens policy. Parsning, cachning, felhantering och
  reservvägar är testade mot mockade svar. `hpmpc providers` är första
  kommandot att köra på riktig maskin.
- **Docker-imagen är inte byggd** — ingen docker-daemon i utvecklingsmiljön. Den
  motsvarande wheel-installationen är verifierad, inklusive paketdata.
- **Ingenting har körts mot en riktig Home Assistant.** HA-klienten är testad mot
  en fake som härmar REST-API:ets format.
- **Delarens siffror är räknade, inte uppmätta.** 20 kΩ mot en Daikin 20 kΩ-kurva
  ger 0,18–1,83 V över −30…+30 °C. Kontrollera med kända motstånd över J1 vid
  idrifttagning: 20 kΩ → 1,650 V, 68 kΩ → 0,750 V, 200 kΩ → 0,300 V.

---

## Fällor som redan är upptäckta

- **Faka inte R1T i utedelen.** Den styr också avfrostningstiming,
  driftområdesgränser och om aggregatet får gå. Daikins innedel kan ta en extern
  utegivare (KRCS01-1) och via en field setting använda *den* för kurvan. Koden
  för den inställningen skiljer mellan generationer — slå upp i installatörsmanualen.
  **Här är det KRCS01-1-ingången som emuleras** (bekräftat 2026-09), alltså rätt
  givare: utedelens R1T är orörd och sköter fortfarande avfrostning och skydd.
- **MCP41100:ans problem är räckvidden, inte upplösningen.** Det var fel i den
  här filen tidigare. 392 Ω per steg ger 0,10 K vid nollan och 0,29 K vid +20 —
  gott nog. Men 100 kΩ tar slut vid **−7,4 °C** på Daikinkurvan, och kallare än
  så finns ingen wiperposition. Under −7 ute sitter wipern i ändläge, pumpen
  visas −7 när det är −15, och huset underhettar tyst. Därför är
  `heat_pump.perceived_min_c: -7` i exempelkonfigurationen, därför varnar
  `hpmpc check`/`ntc-table` när den och `pot:` inte går ihop, och därför läses
  `entities.pot_wiper` tillbaka varje cykel. Fixen är en **andra MCP41100 i
  serie** och `pot.devices: 2` → −20,3 °C nominellt, uppmätt −19,8 på det
  byggda kortet, vid samma steglängd. Seriekopplade
  kretsar ger räckvidd, inte upplösning.
- **Pumpen exciterar givaringången med 4,97 V, uppmätt i tomgång.** Det är taket
  för vad potentiometerns terminaler kan nå, så **MCP-kretsarna måste matas med
  ≥ 4,97 V** — annars klämmer deras ESD-dioder banan i just de kallaste lägena.
  Följden: vid VDD = 5 V kräver MCP41100 3,5 V för en etta (0,7 × VDD) och C3:an
  ger 3,3 V, alltså behövs en **74HCT125** på SCK, SI, CS1 och CS2. Inte HC —
  den har samma tröskel. Matningen tas från skruvplinten, inte från modulens
  `5V`-stift: en del Zero-kort har en diod från USB och ger 4,6 V.
- **Fyra stift per rad går bort på C3-Zero:n, och inget av dem är självklart.**
  GP2 och GP8 är strapping (måste ligga högt vid boot), **GP9 sitter på
  BOOT-knappen**, GP10 på WS2812-lysdioden, GP18/19 på USB och GP20/21 på UART0.
  Kvar: GP0, GP1, GP3, GP4, GP5, GP6, GP7 — sju stift, bygget behöver sex.
  ADC hamnade på **GP0** för att den är granne med 3V3 och GND, så hela den
  analoga klungan får plats utan att korsa kortet.
- **MCP41100 har ingen återläsning.** 8-polsvarianten saknar utgång, och
  `sensor.varmepump_proxy_mcp41100_wiper_0_255` rapporterar nodens egen variabel
  — vad ESP:n *tror* att den skickade. En felläst bit vore alltså ett tyst,
  bestående fel, och med spill-over-logiken är en trasig databyte på andra
  kretsen värd 100 kΩ. Därför skriver firmware om samma wipervärde var 30:e
  sekund: felet blir övergående i stället för permanent, och pumpen filtrerar
  ändå sin utegivare över timmar.
- **Pumpen har ingen egen utegivare kvar** — potentiometern *är* givaren.
  Därför är "koppla bort emulatorn" aldrig ett säkert läge: det ger pumpen ett
  brutet givarkretslopp. Varje reservväg faller tillbaka på ett rimligt
  motstånd: HA skriver riktig utetemperatur med offset 0, ESP32:n håller sitt
  senaste värde i fyra timmar och går sedan till `FAILSAFE_WIPER` (~0 °C).
  Det här revs upp en gång: en tidigare version slutade skriva under
  `perceived_min_c`, vilket var precis fel.
- **Ett lågt värde i provet "strömlöst kort" är en diod, inte ett motstånd.**
  Uppmätt 61,6 kΩ i stället för 195 en gång. Det kan inte vara en parallell
  läcka — hade en sådan funnits hade *det spänningssatta* provet vid wiper 510
  läst 195 ∥ läckan, alltså just 61,6 k, och det läste 195,0. En resistans som
  bara finns när matningen är borta är ingen resistans. Mätarens testspänning
  leder in i POT_HI:s ESD-diod, upp i den flytande +5V-rälen och vidare till jord
  genom modulens 5 V-stift. Kvittot: **värdet ändras med multimeterns
  mätområde**, eftersom varje område har sin egen testspänning — ett verkligt
  motstånd läser lika på alla områden som räcker till. Dra USB-C också (en modul
  på USB backmatar rälen genom U3:s ingångsdioder). **Att lyfta A1 löser det
  inte** — prövat: 61,6 kΩ med modulen i, 41,6 kΩ med den ur, alltså lägre och
  inte högre, så modulen var aldrig avloppet. Jaga det inte med ohmmeter alls:
  en ospänningssatt CMOS-krets är ett nät av övergångar vars skenbara resistans
  beror på mätarens testspänning. **Mät i stället som pumpen gör, med mätaren på
  volt** — 5 V genom ett känt R_ref till J2.1, jord på J2.2, och R = V·R_ref/(5−V).
  Definitiva versionen är pumpens egen display med kortet spänningslöst.
  Ingenting av det blockerar idrifttagningen: det spänningssatta uppförandet är
  redan bevisat, det som är oklart är bara hur bra reservläget är.
- **Flussmedel spelar roll på det här kortet.** I drift är kedjan en torrkrets på
  några tiotals mikroampere, så en läckbana på hundratals kΩ är ett förstahandsfel
  och inte kosmetik. Tvätta med 99 % isopropanol och borste, båda sidor, och mät
  först när det är helt torrt — blöt IPA leder själv. Har man ingen aning om
  mätaren: mät ett känt motstånd (20 kΩ) på två områden. Skiljer de sig mer än
  någon procent är det mätaren eller batteriet, inte kortet.
- **Bygeln PA0↔PW0 räddar INTE ett strömlöst kort. Prövat mot pumpen, den föll.**
  Tanken var: motståndsbanan i en MCP41100 är passiv, så med PA0 byglad till
  wipern ligger hela banan kvar spänningslös och pumpen ser ≈ 195 kΩ ≈ −20 °C —
  fel, men en avläsning i stället för givarfel. Så blir det inte. Med kortet
  strömlöst och pumpen inkopplad visar pumpens display **`--,--`**, alltså
  givarfel, och J2 mäter **0,68 V** — ett dioddropp, inte en resistans.
  Mekanismen: POT_HI:s ESD-diod upp i +5V-rälen leder, och rälen *flyter inte* —
  den ligger på jord genom matningen och kretsarnas VDD-stift. Pumpen klampas
  därmed till 0,68 V, vilket vid dess pull-up motsvarar ett par kΩ, långt under
  NTC-kurvans varma ände (15,98 kΩ vid +30 °C) — alltså utanför området.
  Det förklarar också bänkmätningarna som skiftade med multimeterns mätområde:
  samma diod, probad vid olika testströmmar.
  **Följd:** ett dött kort ger pumpen givarfel, inte −20 °C. Det är på sätt och
  vis ett snällare fel — pumpen larmar och slutar elda i stället för att värma
  för fullt — men det är inte det som står i konstruktionen. En seriediod i
  matningen är *fel* fix: den kostar spänning, och VDD måste vara ≥ 4,97 V.
  **Valt 2026-09: acceptera givarfel som reservläge**, med
  `binary_sensor.varmepump_proxy_online` som hela skyddsnätet. Larmet måste då
  gå åt rätt håll — faran är att pumpen *slutar* värma, inte att den värmer för
  mycket, så ingenting ska stänga av värmen; det har pumpen redan gjort. Ett
  dött kort i januari är tyst i alla kanaler utom pumpens egen display.
- **Reläväxlingen i steg 2 kräver tre växlande poler, inte två**, och guldkontakter.
  Poler: givarens tråd A (3V3 ↔ pumpens terminal A), givarens tråd B (mätnoden ↔
  GND), och pumpens terminal A (POT_HI ↔ bruten). Utan den första hamnar 3V3 på
  pumpens ingång; utan den andra ser pumpen givaren i serie med R1 *och* en
  ~1 kΩ shunt genom R2 in i den döda modulens ESD-diod; utan den tredje hänger
  digipotarna kvar parallellt. Kontakterna måste vara **guldpläterade och
  bifurkerade** — det här är en torrkrets på tiotals mikroampere, och ett vanligt
  effektreläs silverkontakter bygger oxidfilm och blir glappande. Båda spolarna
  på samma drivsteg, annars kan polerna hamna i otakt.
- **Under `perceived_min_c` kommenderas det kallaste hårdvaran kan visa**, och
  gapet rapporteras kvantifierat (`range_shortfall`, gap × `curve_slope` = kelvin
  framledning som fattas). `heat_pump.perceived_min_c` är ändringsbar i drift, så
  den andra MCP41100:an kan tas i bruk utan att röra filen.
- **`_limit` lägger den perceived-gränsen sist**, efter hastighetsbegränsningen.
  Annars kommenderas ett värde potentiometern klipper på egen hand, och modellen
  tror att det tillämpades.
- **`pot:`-geometrin går att ändra i drift** (`pot.devices`, `resistance_ohm`,
  `wiper_ohm`, `series_ohm` finns i `OVERRIDABLE`). Det är precis de tal man mäter
  vid idrifttagning och ändrar igen dagen den andra kretsen sitter, så de ska inte
  kräva att man redigerar filen. `pot.devices` tvingas till heltal via
  `INTEGER_FIELDS` — annars står det `2.0` i filen.
- **`pot:` är skild från `ntc:` med flit.** `ntc:` är givarkurvan man kalibrerar,
  `pot:` är vad hårdvaran kan. En omkalibrering av kurvan får inte tyst ändra
  hårdvarans gränser.
- **`homeassistant.local` fungerar inte i Docker.** `.local` är mDNS, och en
  container har varken Bonjour eller Avahi — samma URL som fungerar i webbläsaren
  ger `Temporary failure in name resolution` inne i containern. Därför är
  exempelkonfigurationens `base_url` en IP-adress, och därför översätter
  `HomeAssistant.diagnose()` anslutningsfel till något handlingsbart.
- **`HA_TOKEN` och `HPMPC_API_KEY` är olika saker.** Den första är HA:s
  long-lived token; den andra skyddar hpmpc:s eget API på 8129 och skickas
  aldrig till Home Assistant. Felmeddelandena säger det explicit, för det är den
  förväxling som faktiskt sker.
- **Elbilsladdarens sensor säger `Charging`/`Not charging`**, inte `on`/`off`.
  `ha.BOOLEAN_STATES` mappar båda.
- **Utan utegivare måste kontrollern arkivera vädret själv** (`record_resolved`).
  SMHI-temperaturen hämtas i stunden och passerar aldrig Home Assistant, så
  recordern har ingen historik och `hpmpc collect` föll på
  `missing required signals: t_outdoor`. Bara signaler utan konfigurerad entitet
  skrivs — finns en givare är recorderns historik tätare och närmare huset.
  **Spotpriset ingår** av samma skäl: husmodellen bryr sig inte om vad el kostar,
  men ett backtest mot ett påhittat platt pris är värdelöst, och det är just den
  siffran det graderar besparingen på. Sparas som spot — påslag och moms läggs
  på vid läsning, så arkivet och en Nord Pool-entitet är utbytbara.
- **En väderentitet är inte en temperaturgivare, men bär alla talen.**
  `weather.smhi_home` har tillståndet `cloudy`; mätvärdena ligger som *attribut*
  (`temperature`, `wind_speed`, `humidity`) och prognosen i tjänsten
  `weather.get_forecasts`. `weather_current()` läser attributen, `WEATHER_FIELDS`
  täcker både visningsnamn och `native_`-varianten, och nu-värdena ankrar
  prognosens första steg precis som en riktig givare. Den hör hemma i
  `entities.weather`, och konfigurationen avvisar den i `outdoor_temp`.
- **SMHI faller tillbaka på `entities.weather` av sig själv** i `weather_points`.
  Det är därför utebliven utgående trafik inte behöver stoppa något. **Exempel-
  konfigurationen står ändå på `weather_source: home_assistant`** — direktvägen
  gav 404 med HTML hos användaren (elpriserna fungerade, så det är inte nätet),
  och URL:en är fortfarande overifierad. Direktvägen är bättre när den fungerar:
  den bär luftfuktighet för varje prognossteg.
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
- **Utgångar kan vara `sensor.*` som hpmpc skapar själv** via states-API:t, eller
  hjälpare som redan finns. Skillnaden är hållbarhet: en publicerad sensor lever
  bara i minnet och glöms vid HA-omstart tills nästa cykel. Därför ska ställdonet
  drivas från en `input_number`, som återställer sitt värde — sensorer är för
  instrumentpaneler. `hpmpc check` säger `pending`, inte `MISSING`, för en sensor
  som ännu inte skapats. `EntityConfig.self_published()` är det enda stället som
  vet vilka entiteter hpmpc skapar själv — `status_entity` ingår, och glömdes
  först, vilket gav falskt `MISSING` på en färsk installation.
- **Normalfallet: hpmpc skapar inga entiteter i HA** — det skriver in i hjälpare som
  `ha/packages/heatpump_mpc.yaml` skapar (`input_number.varmepump_offset`,
  `input_number.varmepump_fiktiv_utetemp`). De står på `unknown` tills första
  cykeln kört; därför faller resistansmallen tillbaka på den riktiga
  utetemperaturen, annars har ESP:n inget att skicka efter en HA-omstart.
  `sensor.utegivare_verklig` överst i paketet är den enda rad användaren måste
  peka mot sina egna entiteter — alla skyddsnät läser den.
- **Regulatorn skriver alla konfigurerade utgångar samtidigt**, inget `output_mode`.
  Samma beslut i kelvin, grader och ohm — då kan de inte säga emot varandra. Historiken
  läses tillbaka från kelvin-entiteten eftersom den inte kräver någon omräkning.
- **Arkivet och recordern kan inte hamna i konflikt.** Raderas arkivet fyller det
  sig från det recordern har kvar; kortas recorderns retention behåller arkivet
  det redan kopierat. `training.archive: false` går direkt mot recordern som förut.
- **Utetemperaturen kommer från `weather.smhi_home` via Home Assistant**, inte
  direkt-SMHI, eftersom `entities.outdoor_temp` är tom här. Direktvägen gav 404
  hos användaren (se ovan); väderentitetens *attribut* (`temperature`,
  `wind_speed`, `humidity`) läses och används på samma sätt en riktig givare
  skulle vara — dess *tillstånd* är bara ett väderomdöme (`cloudy`) och bär
  ingen siffra. En givare vid huset är bättre om den tillkommer — den mäter
  luften byggnaden faktiskt förlorar värme till — och vinner automatiskt så
  fort entiteten fylls i.
- **Automatisk omträning finns bara i `hpmpc serve`, inte i `hpmpc run`.**
  `maybe_retrain()` (i `api.py`) kör var `training.retrain_days` (30 som
  standard) vid `retrain_hour` (natten), bygger dataset ur arkivet och byter
  modell utan omstart. Containern kör `serve`, så det gäller normalt — men
  körs `hpmpc run` fristående för felsökning sker ingen periodisk omträning
  där, bara `hpmpc train` manuellt eller att en modell som redan finns på disk
  plockas upp.
- **En `input_number` utan `initial:` startar på sitt MINIMUM.** Paketets
  `varmepump_fiktiv_utetemp` läste alltså −40 innan hpmpc skrivit första gången,
  och −40 °C är ~800 kΩ = wiper 255 = kallast pumpen kan visas = maximal värme.
  Mallen behandlar nu värden utanför ±35 som "aldrig satt" och faller tillbaka på
  den riktiga utetemperaturen. `applied_offset` gör samma sak vid inläsning: ett
  orimligt perceived-värde blir NaN, inte en offset som ser hårdklippt ut.
  (`initial:` går inte att sätta — då tappar hjälparen sitt värde vid omstart.)
- **`collect` kan tyst kasta nästan hela arkivet.** Rader utan `t_outdoor`
  faller bort i `dropna(subset=REQUIRED)`, och historik från före
  `record_resolved` fanns saknar den kolumnen helt. Både loggen och `collect`
  säger nu hur många rader som föll och varför.
- **ESP32:n tar en temperatur, inte en wiperposition.** Kedjan är
  `input_number.varmepump_fiktiv_utetemp` → automationen `MPC to sensor emulator`
  → `number.varmepump_proxy_simulerad_utetemperatur`, och noden räknar om själv.
  Wiper-vägen finns kvar utkommenterad i paketet. Följd: `hpmpc check` läser
  tillbaka wipern genom *hpmpc:s* `ntc:`/`pot:`, så ett stående ställdonsfel där
  betyder att de två tabellerna är oense — inte att något inte kommer fram.
- **En öppen J1 är ingen temperatur, och noden får inte låtsas annat.** Utan
  givare på J1 ligger delaren i botten, resistansen räknas till megaohm och
  NTC-kurvan extrapolerar till något absurt men **ändligt** — ett tal som
  Home Assistant tar på allvar, eftersom `sensor.utegivare_verklig` föredrar
  nodens givare framför väderentiteten och varje skyddsnät i paketet läser den.
  Ett kort på bänken utan givare skulle alltså kunna påstå −40 ute. Därför
  filtrerar firmware bort allt utanför −45…+60 och publicerar då *ingenting*,
  så mallen faller igenom som det var tänkt. Bandet är med flit vidare än
  Norrköping någonsin blir: det är ett brott-i-kabeln-prov, inte ett
  rimlighetsfilter på vädret.
- **`input_boolean.varmepump_mpc_aktiv` är av tills någon slår på den**, och då
  gör hela kedjan ingenting. En `input_boolean` utan `initial:` startar av, och
  automationen `MPC to sensor emulator` har den som villkor. hpmpc skriver då
  troget till `input_number.varmepump_fiktiv_utetemp` varje cykel, men ingenting
  förs vidare till noden — som ligger kvar på sitt bootvärde. Symptomet är exakt
  "pumpen får bara det satta värdet hela tiden". Slå på den i HA. Att den *måste*
  slås på manuellt är med flit: den är också nödstoppet, och ett nödstopp som
  återställer sig självt vid omstart vore inget nödstopp.
- **`git pull` uppdaterar inte containern.** Containern kör koden som bakades in
  i imagen; en pull rör bara källan på värden. Symptomet är förvirrande: ett
  `hpmpc set` svarar *"'heat_pump.perceived_min_c' is not changeable at runtime"*
  fast fältet står i `OVERRIDABLE` i repot — det står inte i imagens `OVERRIDABLE`.
  Meddelandet handlar alltså inte om att systemet är igång, utan om att fältet
  saknas i den kod som faktiskt kör. `hpmpc settings` listar vad imagen tror är
  ändringsbart och avgör saken på en sekund. Fix: `docker compose up -d --build`.
- **`ntc-table`:s "reaches" är räknad, inte inställd.** Den kommer ur `pot:` och
  `ntc:` och läser aldrig `perceived_min_c`. Att ändra `perceived_min_c` kan
  alltså inte flytta den siffran — det är den kontrollen jämförs *mot*. Två tal
  som ser förvillande lika ut: med den gamla typiska Daikintabellen och den
  uppmätta potentiometern blir räckvidden −19,83 → skrivs "−19.8", och den
  ändras först när `ntc:`-tabellen byts (då till −20,4).
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

## Var vi står

Kortet är **byggt, lött och uppmätt.** Potentiometerkedjan är verifierad (se
"Verifierat" ovan) och hela konfigurationen — `pot:`, `perceived_min_c`, ESPHome-
filens två lambdor och HA-paketets wipermall — står på de uppmätta talen. Kvar
innan pumpen kopplas in: tvätta bort flussmedlet, delarprovet mot kända motstånd
över J1, och pumpens thermistor flyttad till kortet. Reservlägets faktiska värde
(det strömlösa provet) är fortfarande obesvarat och avgörs bäst av pumpens display.

Designen är låst efter en genomgång som gav fem beslut värda att inte riva upp:

1. **Två MCP41100 i serie** direkt från start — en enda tar slut vid −7,4 °C.
2. **5 V-matning till U1/U2/U3, tagen från skruvplinten**, eftersom pumpen
   exciterar med uppmätta 4,97 V.
3. **74HCT125** som följd av det — C3:ans 3,3 V når inte 0,7 × VDD.
4. **Bygeln PA0↔wiper** i stället för fast failsafe-motstånd på ett relä.
5. **Reläväxlingen uppskjuten** till steg 2, med larm som ersättning och tre lösa
   trådbyglar (L1–L3) på kortet så tillägget inte rör något annat. **Beslut
   2026-09: den byggs tills vidare inte alls.** Efter att bygeln PA0↔PW0 visat
   sig inte hålla (se fällorna) är valet medvetet att acceptera givarfel som
   reservläge — pumpen larmar och slutar elda i stället för att värma för fullt,
   vilket är det snällare felet. Priset är att `binary_sensor.varmepump_proxy_online`
   nu är hela skyddsnätet, inte en bekvämlighet.

Byggblad med schema, nätlista, zonplan och idrifttagningsordning finns som artefakt
i den session där kortet togs fram; källan till samma innehåll är
`docs/HARDWARE.md#kortet` och kommentarerna i ESPHome-filen.

## Nästa steg för användaren

**A. Bygg och verifiera kortet** (`docs/HARDWARE.md#kortet`, och
idrifttagningslistan sist i ESPHome-filen). Kort: matningen med tomma hållare,
bufferten ensam, potentiometrarna utan pumpen, det strömlösa provet över J2,
delaren mot kända motstånd. Först därefter pumpen.

**B. Koppla ihop ESP32:n med hpmpc — gjort.** `pot.devices: 2`,
`resistance_ohm: 97377`, `wiper_ohm: 123`, `perceived_min_c: -19.5`, HA-paketets
`pots = 2`, ändlägeslarmet på `510` och wipermallen på de uppmätta talen.
`entities.pot_wiper` pekade redan rätt. Kvar i den tråden:
- `entities.outdoor_temp` — givaren sitter på J1 sedan 2026-09, så den kan fyllas
  i med `sensor.varmepump_proxy_verklig_utetemperatur`. **Gör det tidigt, inte
  sent.** Bytet ger ett steg i träningsdatan mot den SMHI-härledda historiken,
  och just nu är arkivet i princip tomt — byter man nu blir hela datasetet
  homogent från dag ett, byter man om en månad ligger diskontinuiteten mitt i
  det anpassningen ska läsa. Kontrollera först att nodens värde ser rimligt ut
  mot SMHI ett dygn; det ersätter delarprovet mot kända motstånd som aldrig
  gjordes.
- `control.offset_min` lämnad på −6 med flit. Räckvidden tillåter mer nu, men
  läs en vecka planer först (`docs/HARDWARE.md#sänk-inte-offsetgränserna-för-snabbt`).
- `binary_sensor.varmepump_proxy_online` — **gjort**, automationerna
  `MPC proxy node offline` / `... back` finns i paketet. Notera att larmet går åt
  *andra* hållet än den ursprungliga planen: ett dött kort ger givarfel, alltså
  ingen värme alls, inte för mycket. Ingenting ska stänga av värmen — pumpen har
  redan gjort det. Peka `notify.notify` mot något som faktiskt når dig.

**C. Sedan den ursprungliga listan:**
1. `hpmpc providers` — stäm av marginalkostnaden mot elfakturan.
2. ~~Bestäm vilken givare som emuleras~~ — gjort: KRCS01-1-ingången.
3. ~~`hpmpc calibrate-ntc` mot pumpens display~~ — gjort, tio punkter.
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
