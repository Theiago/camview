# CamView

Monitor de câmera IP via RTSP com detecção de movimento e pessoa.

## Funcionalidades

- Stream ao vivo no browser
- Detecção de movimento (MOG2 + YOLO)
- Alerta de permanência (loitering)
- Gravação automática em MP4
- Notificação e foto via Telegram
- Comando `/foto` no Telegram para captura sob demanda
- Seleção de área monitorada (ROI) pelo browser
- Página de gravações com player e exclusão

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
```

## Uso

```bash
uv run python main.py
```

Acesse `http://localhost:5000`.

Para definir a área monitorada acesse `http://localhost:5000/setup`.
