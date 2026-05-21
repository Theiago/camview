# CamView

Monitor de câmera IP via RTSP com detecção de movimento, pessoa e portão.

## Funcionalidades

- Stream ao vivo no browser
- Detecção de movimento (MOG2 + YOLOv11)
- Alerta de permanência (loitering) com tempo configurável
- Detecção de portão aberto/fechado via comparação de imagens de referência
- Modo noturno automático — detecta câmera IR e ajusta sensibilidade
- Gravação automática em MP4 com buffer pré-evento
- Notificações via Telegram (foto no alerta, `/foto` e `/video` sob demanda)
- Integração com Alexa via Sinric Pro (dispara rotinas ao detectar permanência)
- Toggle de alertas — silencia Alexa e Telegram sem desativar a detecção
- Página de gravações com player, data/hora e exclusão
- Configuração de 3 áreas pelo browser: movimento, permanência e portão

## Requisitos

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)
- ffmpeg

## Instalação

```bash
uv sync
```

## Configuração

Crie um arquivo `.env`:

```
RTSP_URL=rtsp://ip:porta/caminho
TELEGRAM_TOKEN=seu_token
TELEGRAM_CHAT_ID=seu_chat_id
SINRIC_APP_KEY=seu_app_key
SINRIC_APP_SECRET=seu_app_secret
SINRIC_API_KEY=sua_api_key
SINRIC_DEVICE_ID=seu_device_id
```

As variáveis do Sinric Pro são opcionais — sem elas a integração com Alexa é desativada.

## Uso

```bash
uv run python main.py
```

Acesse `http://localhost:5000`.

Para configurar as áreas monitoradas acesse `http://localhost:5000/setup`.
