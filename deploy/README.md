# QuantAgent v2 — Production Deployment

Single Hetzner server. No Kubernetes. systemd + nginx + PostgreSQL.

## Server Requirements

- Ubuntu 22.04+ or Debian 12+
- Python 3.12+
- PostgreSQL 15+
- nginx
- 4GB+ RAM (8GB recommended)

## First-Time Setup

```bash
# 1. Create user
sudo useradd -m -s /bin/bash quantagent
sudo mkdir -p /opt/quantagent
sudo chown quantagent:quantagent /opt/quantagent

# 2. Clone repo
sudo -u quantagent git clone https://github.com/Alishark14/quantagent_v2.git /opt/quantagent
cd /opt/quantagent

# 3. Python environment
sudo -u quantagent python3.12 -m venv .venv
sudo -u quantagent .venv/bin/pip install -e ".[dev]"

# 4. Configure environment
sudo -u quantagent cp deploy/.env.production.template .env
sudo -u quantagent nano .env  # fill in all values

# 5. Database setup
sudo -u postgres createuser quantagent
sudo -u postgres createdb -O quantagent quantagent
sudo -u quantagent .venv/bin/python -m quantagent migrate

# 6. Seed initial data (optional)
sudo -u quantagent .venv/bin/python -m quantagent seed

# 7. Install systemd service
sudo cp deploy/quantagent.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable quantagent
sudo systemctl start quantagent

# 8. Install nginx config
sudo cp deploy/nginx.conf /etc/nginx/sites-available/quantagent
sudo ln -sf /etc/nginx/sites-available/quantagent /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# 9. Verify
curl http://localhost/v1/health | python3 -m json.tool
```

## Deploy Updates

From your local machine:

```bash
./scripts/deploy.sh quantagent@your-server.com main
```

This pulls, installs, migrates, restarts, and health-checks in one command.

## Commands

```bash
# Service management
sudo systemctl start quantagent
sudo systemctl stop quantagent
sudo systemctl restart quantagent
sudo systemctl status quantagent

# Logs (live tail)
sudo journalctl -u quantagent -f

# Logs (last 100 lines)
sudo journalctl -u quantagent -n 100 --no-pager

# Health check
./scripts/health_check.sh
./scripts/health_check.sh your-server.com:80

# Manual migration
cd /opt/quantagent && .venv/bin/python -m quantagent migrate

# Manual seed
cd /opt/quantagent && .venv/bin/python -m quantagent seed
```

## SSL (Let's Encrypt)

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
# Then uncomment the SSL lines in deploy/nginx.conf
```

## Troubleshooting

| Symptom | Check |
|---------|-------|
| Service won't start | `journalctl -u quantagent -n 50` |
| 502 Bad Gateway | `systemctl status quantagent` — is it running? |
| DB connection error | Verify DATABASE_URL in .env, check `pg_isready` |
| API returns 401 | Check API_KEYS in .env |
| High memory | Check bot count, consider reducing concurrent symbols |
