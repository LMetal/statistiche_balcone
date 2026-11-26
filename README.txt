SUONO BALCONE - livello dB + YAMNet -> Prometheus -> Grafana

DA PORTAINER (consigliato)
1. Metti questa cartella in un repository Git (GitHub/GitLab, anche privato).
2. Portainer > Stacks > Add stack > Repository
   - Repository URL: il tuo repo
   - Compose path: docker-compose.yml
   - Environment variables:
       STREAM_URL        = http://IP:PORTA/stream   (obbligatoria)
       GRAFANA_PASSWORD  = la tua password          (obbligatoria)
       CAL_OFFSET_DB, DETECT_THRESHOLD, WINDOW_SECONDS (opzionali)
3. Deploy the stack (la prima build sul Pi richiede qualche minuto).
4. Grafana: http://IP_DEL_PI:3000  (admin / GRAFANA_PASSWORD)

Aggiornamenti: modifichi il repo, poi in Portainer "Pull and redeploy".

DA TERMINALE (alternativa)
   export STREAM_URL=... GRAFANA_PASSWORD=...
   docker compose up -d --build

Note:
- I dB sono relativi (dBFS) finche' non calibri con CAL_OFFSET_DB.
- Classi YAMNet tracciate: dizionario TRACKED in analyzer/app.py.
