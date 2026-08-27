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
hemkomsttid — plus dödmansgreppet som släpper givaremulatorn om kontrollern
tystnar.

### Recorder

Modellen behöver minst tre till fyra veckors historik för de entiteter den
använder. Standard är tio dagar. Antingen höj generellt:

```yaml
recorder:
  purge_keep_days: 45
```

eller spara bara det som behövs länge:

```yaml
recorder:
  purge_keep_days: 10
  include:
    entities:
      - sensor.vardagsrum_temperatur
      - sensor.ute_temperatur
      - sensor.daikin_framledning
      - sensor.victron_ac_consumption_l1_power
      - sensor.victron_ac_consumption_l2_power
      - sensor.victron_ac_consumption_l3_power
      - binary_sensor.eh6nh5cd_charging
      - number.utegivare_resistans
```

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

Det sista skriver `config/config.yaml` — redan ifylld för Daikin Altherma LT,
Norrköping och SE3. Öppna den och byt entitets-id:n mot dina egna.

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
docker compose exec hpmpc hpmpc pump-table   # COP och kapacitet
docker compose exec hpmpc hpmpc settings     # vad du kan ändra i efterhand
docker compose exec hpmpc hpmpc mode         # aktivt komfortläge
```

**Stäm av `hpmpc providers` mot din elfaktura.** Den skriver ut
marginalkostnaden med ditt tillägg och din moms — det är den siffran
optimeraren faktiskt planerar mot, och den enda som är värd att kontrollräkna.

Sätt `dry_run: true` i konfigurationen och låt den gå ett par dygn. Läs
planerna med `hpmpc plan`. Först därefter `hpmpc set control.dry_run 0`.

---

## 4. Modellen

```bash
docker compose exec hpmpc hpmpc excite       # ~1 vecka, kör i förgrunden
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
