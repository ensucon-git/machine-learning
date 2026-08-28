# Installation på en NUC med Docker

Guide för att köra hpmpc som en tjänst på en egen maskin, skild från Home
Assistant. Räkna med en timmes arbete första gången, plus en vecka excitation
innan modellen är värd att lita på.

Kontrollern behöver inte mycket: den använder ungefär 300 MB RAM och en
CPU-kärna i någon sekund var femtonde minut. Träningen tar en minut eller två
en gång i månaden.

---

## 1. Förberedelser i Home Assistant

### Long-lived access token

Profil → Säkerhet → *Long-lived access tokens* → **Skapa token**. Kopiera den
direkt; den visas bara en gång.

### Paketet med hjälpare och skyddsnät

Kopiera `ha/packages/heatpump_mpc.yaml` till `config/packages/` i Home Assistant
och lägg till i `configuration.yaml`:

```yaml
homeassistant:
  packages: !include_dir_named packages
```

Starta om HA. Du får då hjälparna för läge, börvärde, semesterläge och
hemkomsttid — plus dödmansgreppet som skriver den riktiga utetemperaturen med
offset 0 om kontrollern tystnar. (Det finns ingen termistor kvar att falla
tillbaka på — potentiometern *är* pumpens givare — så säkert läge betyder
"visa sanningen", inte "koppla bort".)

**Redigera en rad i paketet:** mallsensorn `Utegivare verklig` överst är den
riktiga utetemperaturen, och den enda plats du behöver peka mot dina egna
entiteter. Alla skyddsnät faller tillbaka på den.

### Vad hpmpc skriver — och vad du kopplar ESP:n till

hpmpc skapar inga egna entiteter. Det **skriver in i två hjälpare** som paketet
skapar, varje styrcykel:

| entitet | enhet | vad det är |
|---|---|---|
| `input_number.varmepump_offset` | K | beslutet självt |
| `input_number.varmepump_fiktiv_utetemp` | °C | riktig utetemp + offset — **temperaturen att visa pumpen** |

Den andra är den du agerar på. Den är redan inkopplad i exempelkonfigurationen:

```yaml
entities:
  offset_output: input_number.varmepump_offset
  fake_temperature_output: input_number.varmepump_fiktiv_utetemp
```

**Båda står på `unknown` tills hpmpc kört sin första cykel.** Det är normalt på
en färsk installation — `hpmpc plan` visar vad som *skulle* skrivas innan dess.
Under tiden faller resistansmallen tillbaka på den riktiga utetemperaturen, så
ESP:n har alltid ett värde att skicka.

Sedan finns två vägar vidare till ESP:n, och du väljer den som passar din
firmware:

- **Din nod tar en wiperposition** (som den ESPHome-fil som ligger med): mallarna
  `sensor.utegivare_malresistans` → `sensor.utegivare_wiper` gör omräkningen, och
  automationen `MPC to digital resistor` skickar den. NTC-kurvan bor då i Home
  Assistant där du kan trimma den mot pumpens display.
- **Din nod tar en temperatur**: skicka `input_number.varmepump_fiktiv_utetemp`
  rakt av. Automation **B)** längst ner i paketet gör precis det — avkommentera
  den och ta bort A). ESPHome-filen har en `number.varmepump_proxy_target_temperature`
  som tar emot grader och räknar om till wiper i firmware.

Kolla entitets-id:t under **Utvecklarverktyg → Tillstånd** — det beror på vad din
nod heter, inte på vad som står här.

### Recorder

**hpmpc sparar sin egen historik.** Varje styrcykel kopierar det recordern har
fått sedan förra cykeln till `data/history/` i containern, och där ligger det
kvar i `training.archive_keep_days` dygn (400 som standard). Recordern behöver
alltså bara hålla längre än *glappet mellan två styrcykler* — timmar, inte
veckor. Standardinställningen tio dagar räcker gott.

```bash
docker compose exec hpmpc hpmpc archive     # vad vi själva har sparat
```

Det enda recorderns retention fortfarande styr är hur mycket historik du **ärver
vid första installationen**. Vill du kunna träna direkt istället för att vänta
en månad på att arkivet fyller sig, höj retentionen *innan* du installerar —
antingen generellt:

```yaml
recorder:
  purge_keep_days: 45
```

eller bara för de entiteter modellen läser:

```yaml
recorder:
  purge_keep_days: 10
  include:
    entities:
      - sensor.hall_temperature_2
      - sensor.gx_device_consumption_power_l1
      - sensor.gx_device_consumption_power_l2
      - sensor.gx_device_consumption_power_l3
      - binary_sensor.eh6nh5cd_charging
      - sensor.varmepump_proxy_mcp41100_wiper_0_255
      - input_number.varmepump_offset
      - input_number.varmepump_fiktiv_utetemp
```

Sedan kan du sänka den igen — arkivet behåller det som redan kopierats. De två
kan aldrig hamna i konflikt: raderar du arkivet fyller det sig på nytt från det
recordern har kvar.

Utan tillräcklig historik blir första träningen tunn — och det märks mest på
effektuppdelningen, som vill ha en månad.

---

## 2. Installera på NUC:en

### Alternativ A: kommandoraden (rekommenderas första gången)

```bash
sudo mkdir -p /opt/hpmpc && sudo chown "$USER" /opt/hpmpc
git clone <repo-url> /opt/hpmpc
cd /opt/hpmpc

mkdir -p config
cat > .env <<'EOF'
HA_TOKEN=<din long-lived token>
HPMPC_API_KEY=<valfri, skyddar API:et>
TZ=Europe/Stockholm
EOF
chmod 600 .env

docker compose build
docker compose run --rm hpmpc init-config
```

Det sista skriver `config/config.yaml` — redan ifylld för den här anläggningen:
Daikin Altherma LT, Norrköping, SE3, Victrons faseffekter och MCP41100-proxyn.
Öppna den och kontrollera entitets-id:na.

Två saker där som är värda en extra titt:

`entities.outdoor_temp` är **tom med flit**. Utetemperaturen hämtas då från SMHI
inne i kontrollern, så ingen utegivare behövs och Home Assistant är inte
mellanhand för den. Har du en givare vid huset senare: skriv in dess entitets-id
där, så vinner den automatiskt över SMHI — inget annat behöver ändras.

`heat_pump.perceived_min_c` är satt till **−7**, inte −20, för att det är så
långt en enda MCP41100 räcker. Se avsnittet om potentiometern nedan.

```bash
docker compose up -d
docker compose exec hpmpc hpmpc check
```

### Alternativ B: Portainer

Portainer → **Stacks** → **Add stack** → **Repository**:

| Fält | Värde |
|---|---|
| Repository URL | ditt repo |
| Compose path | `docker-compose.yml` |
| Environment variables | `HA_TOKEN`, `TZ`, ev. `HPMPC_API_KEY` |

Portainer checkar ut repot på NUC:en och bygger imagen. `./config` blir en
katalog i utcheckningen — och den är tom första gången, så skapa konfigurationen
innan du startar stacken:

```bash
docker run --rm -v /var/lib/docker/volumes/<stack>_config:/config hpmpc:latest init-config
```

I praktiken är det enklare att göra alternativ A först och sedan peka Portainer
på `/opt/hpmpc` som en **local** stack. Portainer blir då fönstret mot loggar
och omstarter, inte det som äger installationen.

---

## 3. Verifiera innan du släpper den lös

Kör allt det här innan kontrollern får skriva något skarpt. Varje kommando
svarar på en fråga du annars får svar på senare, dyrare.

```bash
docker compose exec hpmpc hpmpc check        # entiteter, värden, ålder
docker compose exec hpmpc hpmpc providers    # SMHI + SE3-priser
docker compose exec hpmpc hpmpc ntc-table    # vad potentiometern räcker till
docker compose exec hpmpc hpmpc pump-table   # COP och kapacitet
docker compose exec hpmpc hpmpc settings     # vad du kan ändra i efterhand
docker compose exec hpmpc hpmpc mode         # aktivt komfortläge
```

**Stäm av `hpmpc providers` mot din elfaktura.** Den skriver ut
marginalkostnaden med ditt tillägg och din moms — det är den siffran
optimeraren faktiskt planerar mot, och den enda som är värd att kontrollräkna.

**Läs `hpmpc ntc-table` noga.** Den skriver ut vilket temperaturband
potentiometern faktiskt når, och varnar om `heat_pump.perceived_min_c` ligger
utanför det. Med en enda MCP41100:

```
Pot:       1 x mcp41100 in series, 256 positions, 392 ohm per step
           reaches -7.4 to +30.0 degC
```

Upplösningen är gott och väl tillräcklig — 0,10 K per steg kring nollan. Det är
**räckvidden** som tar slut. Under cirka −7 °C ute sitter wipern i sitt ändläge
och pumpen visas −7 när det är −15, utan att något säger ifrån. Två saker gör
det ofarligt:

1. `heat_pump.perceived_min_c: -7` gör att optimeraren aldrig planerar en offset
   den inte kan leverera. Det står redan så i exempelkonfigurationen.
2. Innan riktig kyla: löd in en **andra MCP41100 i serie** och sätt
   `pot.devices: 2`. Det är ännu en likadan krets — en 100 **kΩ** 8-bitars
   digitalpotentiometer, inte en fast resistor. Då når du −20 °C med samma
   steglängd; seriekopplade kretsar ger räckvidd, inte upplösning. Höj sedan
   gränsen utan att röra filen:

   ```bash
   hpmpc set heat_pump.perceived_min_c -20
   ```
3. Under gränsen fortsätter regulatorn styra — den måste, potentiometern är
   pumpens enda givare. Den kommenderar det kallaste hårdvaran kan visa och
   rapporterar hur mycket framledning som fattas:

   ```
   LIMITED: -12.0 C out, pump held at -7.0 C - 1.5 K of supply temperature short
   ```

   Huset kryper då nedåt de timmarna, långsamt. Det går att kompensera under
   tiden genom att höja värmekurvan lite (`hpmpc set heat_pump.curve_offset`)
   och sänka den igen när den andra kretsen sitter.

**Löd också en fast resistor på reläets NC-kontakt** — ungefär 68 kΩ, 1 %
metallfilm, vilket Daikinkurvan läser som 0 °C. Det är det enda skyddet som
överlever att ESP32:n blir strömlös; utan den betyder ett strömavbrott ett
givarfel på pumpen, eftersom det inte finns någon termistor kvar bakom
emulatorn.

Automationen `MPC potentiometer at end stop` i HA-paketet larmar om wipern
står kvar i ett ändläge i en halvtimme.

Sätt `dry_run: true` i konfigurationen och låt den gå ett par dygn. Läs
planerna med `hpmpc plan`. Först därefter `hpmpc set control.dry_run 0`.

---

## 4. Modellen

```bash
docker compose exec hpmpc hpmpc excite       # ~1 vecka, kör i förgrunden
docker compose exec hpmpc hpmpc archive      # hur mycket historik vi har sparat
docker compose exec hpmpc hpmpc collect --days 45
docker compose exec hpmpc hpmpc train
docker compose exec hpmpc hpmpc power        # kontrollera effektuppdelningen
docker compose exec hpmpc hpmpc backtest --days 7
```

`hpmpc excite` blockerar terminalen. Kör det hellre som en egen container under
veckan:

```bash
docker compose run -d --name hpmpc-excite hpmpc excite --hold-hours 6
# och när veckan är slut:
docker stop hpmpc-excite && docker rm hpmpc-excite
docker compose up -d
```

Två saker att titta på i utskriften: **validation RMSE** under 0,3 °C i
byggnadsanpassningen, och **laddarens effekt** nära 11 kW i `hpmpc power`. Den
andra är den bästa kontrollen på att fasentiteterna är rätt kopplade.

Därefter sköter sig omträningen själv: `training.retrain_days: 30` gör att
tjänsten tränar om när modellen blivit en månad gammal, vid `retrain_hour`
lokal tid. Sätt till 0 om du hellre gör det för hand.

---

## 5. Drift

```bash
docker compose logs -f hpmpc            # vad den gör, cykel för cykel
docker compose exec hpmpc hpmpc plan    # nuvarande plan, skriver ingenting
curl localhost:8129/health              # öppet, för övervakning
```

Ändringar i `config/config.yaml` läses om automatiskt vid nästa cykel — ingen
omstart behövs. Detsamma gäller de inställningar du mappat till
HA-hjälpentiteter.

### Uppdatera

```bash
cd /opt/hpmpc && git pull
docker compose build && docker compose up -d
```

Modellen och historiken ligger i volymen `hpmpc-data` och överlever bygget.
**Radera inte den volymen** — där ligger arkivet under `history/`, och det är den
enda kopian av historik som recordern redan har rensat bort. En backup av volymen
är en backup av allt modellen vet om huset.

### Säkerhetskopiera

Två saker är värda att spara — resten går att räkna fram igen:

```bash
cp config/config.yaml ~/backup/
docker compose cp hpmpc:/data/models ~/backup/models
```

`config.yaml` innehåller din kalibrering av givaren och värmekurvan, och
`models/` innehåller en månads inlärning. Historiken kan alltid hämtas om från
Home Assistant.

---

## 6. Om något inte fungerar

| Symptom | Trolig orsak |
|---|---|
| `hpmpc check` når inte HA | fel `base_url`, eller token utan behörighet. Testa `curl -H "Authorization: Bearer $HA_TOKEN" http://ha:8123/api/` |
| `hpmpc providers` misslyckas mot SMHI | koordinater utanför Norden, eller ingen utgående nät från containern |
| Morgondagens priser saknas | normalt före 13:00. `price_extrapolated_hours` säger hur mycket som gissas |
| Kontrollern faller tillbaka till offset 0 | sensordata saknas eller är för gammal; `problems` i loggen säger vilken |
| Planen vill ha elpatron | kurvan, offsetgränserna eller dimensioneringen pressar systemet förbi kompressorn — se `hpmpc pump-table` |
| Semesterläget gör ingenting | offsetgränsen räcker inte för att kyla ner. Kontrollern säger det själv i sina `notes` |
| Fel tidszon i loggarna | `TZ` saknas i `.env` |

Om kontrollern verkar göra något oväntat: `hpmpc plan` skriver ut hela
resonemanget — prognos, priser, planerad offset per timme, förväntad
innetemperatur och vad den tror att den sparar.
